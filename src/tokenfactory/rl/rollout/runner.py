import concurrent.futures
import logging
import queue
import threading
import time
import traceback
from dataclasses import asdict
from functools import partial
from typing import Generic, Protocol, TypeVar

from tokenfactory.rl.api_client import NotFoundError, TokenFactory
from tokenfactory.rl.api_client.models import Sample as APISample
from tokenfactory.rl.rollout.config import RolloutConfig
from tokenfactory.rl.rollout.context import RolloutContext
from tokenfactory.rl.rollout.dataset import DatasetProtocol
from tokenfactory.rl.rollout.exceptions import JobInitializationTimeout
from tokenfactory.rl.rollout.models import (
    JobStatusUpdate,
    RolloutException,
    RolloutResult,
    Sample,
    SampleGroup,
    Task,
    TaskID,
)
from tokenfactory.rl.schedulers import (
    BaseRolloutScheduler,
    RolloutSchedulerConfig,
    create_rollout_scheduler,
)


DEFAULT_QUEUE_GET_TIMEOUT = 0.1


logger = logging.getLogger(__name__)

TaskSpec_contra = TypeVar("TaskSpec_contra", contravariant=True)

RolloutRunnerInputs = (
    JobStatusUpdate | RolloutResult[TaskSpec_contra] | RolloutException
)


class RolloutFn(Protocol[TaskSpec_contra]):
    def __call__(
        self, task: TaskSpec_contra, context: RolloutContext
    ) -> SampleGroup: ...


class RolloutRunner(Generic[TaskSpec_contra]):
    def __init__(
        self,
        api_client: TokenFactory,
        config: RolloutConfig,
        rollout_fn: RolloutFn[TaskSpec_contra],
        dataset: DatasetProtocol[TaskSpec_contra],
    ):
        self._api_client = api_client
        self._jobs_api = api_client.v1alpha1.fine_tuning.jobs
        self._config = config
        self._rollout_fn = rollout_fn
        self._dataset = dataset
        self._context = RolloutContext(
            config=config,
            api_client=api_client,
        )
        self._scheduler: BaseRolloutScheduler[TaskID, Sample] = (
            create_rollout_scheduler(
                config=RolloutSchedulerConfig(
                    batch_size=self._config.batch_size,
                    num_batches=self._config.num_batches,
                    num_samples_per_task=self._config.num_samples_per_task,
                    parallel_rollouts=self._config.max_concurrency,
                    allowed_staleness=self._config.allowed_staleness,
                    drop_stale_trajectories=self._config.drop_stale_trajectories,
                )
            )
        )
        self._dispatcher_queue: queue.Queue[Task[TaskSpec_contra]] = queue.Queue()
        self._input_queue: queue.Queue[RolloutRunnerInputs[TaskSpec_contra]] = (
            queue.Queue()
        )

        self._job_status_tracker = JobStatusTracker(
            job_id=self._config.job_id,
            api_client=self._api_client,
            output_queue=self._input_queue,
            poll_interval=self._config.status_polling_interval,
        )
        self._rollout_dispatcher = RolloutDispatcher(
            rollout_fn=self._rollout_fn,
            context=self._context,
            input_queue=self._dispatcher_queue,
            output_queue=self._input_queue,
            concurrent_workers=self._config.max_concurrency,
            num_rollout_retries=self._config.num_rollout_retries,
            raise_on_rollout_failure=self._config.raise_on_rollout_failure,
        )
        self._batch_idx = 0
        self._batch_n_samples = 0

    def run(self):
        self._wait_for_job_initialization()

        self._batch_idx, self._batch_n_samples = self._load_initial_state()
        self._scheduler.set_initial_batch_index(
            batch_index=self._batch_idx, n_samples=self._batch_n_samples
        )

        self._job_status_tracker.start()
        self._rollout_dispatcher.start()

        while True:
            update = self._input_queue.get()
            match update:
                case JobStatusUpdate(inference_version=inference_version):
                    logger.info(
                        f"Received job status update: "
                        f"inference_version={inference_version}"
                    )
                    self._scheduler.set_inference_version(
                        inference_version=inference_version
                    )
                    self._spawn_tasks()
                case RolloutResult(task=task, sample_group=sample_group):
                    logger.debug(
                        f"Received rollout result for "
                        f"{task.id=} {sample_group.is_rejected=}"
                    )
                    self._process_rollout_result(task=task, sample_group=sample_group)
                    self._submit_samples()
                    self._spawn_tasks()

                    if self._scheduler.is_all_finished:
                        self._job_status_tracker.stop()
                        self._rollout_dispatcher.stop()
                        self._job_status_tracker.join()
                        self._rollout_dispatcher.join()
                        break

                case RolloutException(exception=e, traceback=tb):
                    logger.error(f"Received rollout exception: {e}")
                    logger.error(f"Original traceback:\n{tb}")
                    raise e

    def _wait_for_job_initialization(self) -> None:
        deadline = (
            None
            if self._config.job_init_timeout is None
            else time.monotonic() + self._config.job_init_timeout
        )
        last_exception: Exception | None = None

        while True:
            try:
                self._jobs_api.get_runtime_status(job_id=self._config.job_id)
            except Exception as e:
                last_exception = e
                logger.info(f"Error polling job status during initialization: {e}")
            else:
                logger.info(
                    f"Job {self._config.job_id} responded during initialization"
                )
                return

            if deadline is not None and time.monotonic() > deadline:
                break
            time.sleep(self._config.status_polling_interval)

        assert self._config.job_init_timeout is not None
        error = JobInitializationTimeout(
            job_id=self._config.job_id,
            timeout_seconds=self._config.job_init_timeout,
            last_exception=last_exception,
        )
        if last_exception is not None:
            raise error from last_exception
        raise error

    def _load_initial_state(self) -> tuple[int, int]:
        job_status = self._jobs_api.get_runtime_status(job_id=self._config.job_id)
        try:
            current_batch = self._jobs_api.batches.get(
                job_id=self._config.job_id, batch_index=job_status.filled_batches
            )
        except NotFoundError:
            logger.info(
                f"No batch found for job {self._config.job_id} "
                f"at index {job_status.filled_batches}."
            )
            self._jobs_api.batches.create(
                job_id=self._config.job_id, index=job_status.filled_batches
            )
            return job_status.filled_batches, 0
        assert isinstance(current_batch.sample_count, int)
        logger.info(
            f"Starting batch index {current_batch.index}: "
            f"{current_batch.sample_count}/{self._config.batch_size}"
        )
        return job_status.filled_batches, current_batch.sample_count

    def _spawn_tasks(self):
        n_rollouts = self._scheduler.get_num_rollouts_to_spawn()
        if n_rollouts <= 0:
            return
        assert self._scheduler.inference_version is not None
        tasks = [
            Task(spec=t, starting_inference_version=self._scheduler.inference_version)
            for t in self._dataset.sample(
                n_rollouts // self._config.num_samples_per_task
            )
        ]
        for task in tasks:
            logger.debug(
                f"Spawning task {task.id} with inference version "
                f"{task.starting_inference_version}"
            )
            self._dispatcher_queue.put(task)
        self._scheduler.add_tasks(tasks=tasks)

    def _process_rollout_result(
        self, task: Task[TaskSpec_contra], sample_group: SampleGroup
    ) -> None:
        if sample_group.is_rejected:
            self._scheduler.remove_task(task_id=task.id)
            return
        for sample in sample_group.samples:
            sample.debug_info["starting_inference_version"] = (
                task.starting_inference_version
            )
        self._scheduler.add_sample_group(task_id=task.id, samples=sample_group.samples)

    def _submit_samples(self) -> None:
        ready_samples = self._scheduler.get_ready_samples()
        offset = 0
        while offset < len(ready_samples):
            batch_remaining = self._config.batch_size - self._batch_n_samples
            chunk = ready_samples[offset : offset + batch_remaining]
            offset += len(chunk)
            self._batch_n_samples += len(chunk)

            logger.info(
                f"Submitting {len(chunk)} samples to {self._batch_idx}: "
                f"({self._batch_n_samples}/{self._config.batch_size})"
            )
            self._jobs_api.batches.submit_samples(
                job_id=self._config.job_id,
                batch_index=self._batch_idx,
                samples=[APISample.model_validate(asdict(s)) for s in chunk],
            )

            if self._batch_n_samples == self._config.batch_size:
                self._batch_idx += 1
                self._batch_n_samples = 0
                self._jobs_api.batches.create(
                    job_id=self._config.job_id, index=self._batch_idx
                )


class JobStatusTracker(threading.Thread):
    """Thread that polls the status of a rollout job at regular intervals
    and sends updates to the RolloutRunner via a queue."""

    def __init__(
        self,
        job_id: str,
        api_client: TokenFactory,
        output_queue: queue.Queue[RolloutRunnerInputs[TaskSpec_contra]],
        poll_interval: float = 1.0,
    ):
        super().__init__(name=self.__class__.__name__, daemon=True)
        self._api_client = api_client
        self._jobs_api = api_client.v1alpha1.fine_tuning.jobs
        self._job_id = job_id
        self._output_queue = output_queue
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                job_status = self._jobs_api.get_runtime_status(job_id=self._job_id)
            except Exception as e:
                logger.info(f"Error polling job status: {e}")
                self._stop_event.wait(timeout=self._poll_interval)
                continue
            if job_status.inference_version is not None:
                self._output_queue.put(
                    JobStatusUpdate(inference_version=job_status.inference_version)
                )
            self._stop_event.wait(timeout=self._poll_interval)

    def stop(self):
        self._stop_event.set()


class RolloutDispatcher(threading.Thread, Generic[TaskSpec_contra]):
    """Thread that dispatches rollout tasks to a pool
    based on the current job status."""

    def __init__(
        self,
        rollout_fn: RolloutFn[TaskSpec_contra],
        context: RolloutContext,
        input_queue: queue.Queue[Task[TaskSpec_contra]],
        output_queue: queue.Queue[RolloutRunnerInputs[TaskSpec_contra]],
        concurrent_workers: int,
        num_rollout_retries: int,
        raise_on_rollout_failure: bool,
    ):
        super().__init__(name=self.__class__.__name__, daemon=True)
        self._rollout_fn = rollout_fn
        self._context = context
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._stop_event = threading.Event()
        self._concurrent_workers = concurrent_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_workers
        )
        self._num_rollout_retries = num_rollout_retries
        self._raise_on_rollout_failure = raise_on_rollout_failure

    def run(self):
        try:
            while not self._stop_event.is_set():
                try:
                    task = self._input_queue.get(timeout=DEFAULT_QUEUE_GET_TIMEOUT)
                    future = self._executor.submit(
                        self._rollout_fn, task=task.spec, context=self._context
                    )
                    future.add_done_callback(
                        partial(self._handle_result, task=task, attempt=1)
                    )
                except queue.Empty:
                    continue
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _handle_result(
        self,
        future: concurrent.futures.Future[SampleGroup],
        task: Task[TaskSpec_contra],
        attempt: int,
    ):
        try:
            result = future.result()
            self._output_queue.put(RolloutResult(task=task, sample_group=result))
        except Exception as e:
            if attempt <= self._num_rollout_retries and not self._stop_event.is_set():
                logger.info(
                    "Rollout failed "
                    f"(attempt {attempt}/{self._num_rollout_retries + 1}), "
                    f"retrying: {e}"
                )
                new_future = self._executor.submit(
                    self._rollout_fn, task=task.spec, context=self._context
                )
                new_future.add_done_callback(
                    partial(self._handle_result, task=task, attempt=attempt + 1)
                )
            else:
                logger.exception(f"Rollout failed after {attempt} attempts")
                if self._raise_on_rollout_failure:
                    logger.warning(
                        f"Raising exception to main thread due to rollout failure: {e}"
                    )
                    self._output_queue.put(
                        RolloutException(exception=e, traceback=traceback.format_exc())
                    )
                else:
                    logger.warning(f"Marking rollout as rejected due to failure: {e}")
                    self._output_queue.put(
                        RolloutResult(
                            task=task, sample_group=SampleGroup(is_rejected=True)
                        )
                    )

    def stop(self):
        self._stop_event.set()
