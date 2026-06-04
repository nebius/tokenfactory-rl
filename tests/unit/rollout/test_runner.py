import concurrent.futures
import queue
import time
from dataclasses import dataclass
from typing import Any, Literal
from unittest.mock import MagicMock, call

import pytest

from tokenfactory.rl.api_client import NotFoundError, TokenFactory
from tokenfactory.rl.api_client.models import Batch, JobStatus
from tokenfactory.rl.rollout import JobInitializationTimeout
from tokenfactory.rl.rollout.config import ExecutorType, RolloutConfig
from tokenfactory.rl.rollout.context import RolloutContext
from tokenfactory.rl.rollout.models import (
    JobStatusUpdate,
    RolloutException,
    RolloutResult,
    Sample,
    SampleGroup,
    SampleID,
    Task,
    TaskID,
)
from tokenfactory.rl.rollout.runner import JobStatusTracker, RolloutDispatcher, RolloutRunner, RolloutRunnerInputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTaskSpec:
    question: str = "test"


@dataclass
class ProcessSmokeTask:
    question: str


class FakeDataset:
    def sample(self, n_tasks: int) -> list[FakeTaskSpec]:
        return [FakeTaskSpec(question=f"q{i}") for i in range(n_tasks)]


TrackerQueue = queue.Queue[RolloutRunnerInputs[FakeTaskSpec]]
DispatcherInputQueue = queue.Queue[Task[FakeTaskSpec]]
DispatcherOutputQueue = queue.Queue[RolloutRunnerInputs[FakeTaskSpec]]


def _make_sample(
    *,
    sample_id: SampleID | None = None,
    token_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
    mask: list[int] | None = None,
    normalized_reward: float = 0.5,
    metadata: dict[str, str] | None = None,
    debug_info: dict[str, Any] | None = None,
) -> Sample:
    return Sample(
        id=sample_id or SampleID.generate(),
        token_ids=token_ids or [1, 2, 3],
        logprobs=logprobs or [0.1, 0.2, 0.3],
        mask=mask or [0, 1, 1],
        normalized_reward=normalized_reward,
        metadata=metadata or {},
        debug_info=debug_info or {},
    )


def _process_smoke_rollout_fn(task: ProcessSmokeTask, context: RolloutContext) -> SampleGroup:
    return SampleGroup(
        samples=[
            Sample(
                token_ids=[1],
                logprobs=[0.0],
                mask=[1],
                normalized_reward=1.0,
                debug_info={
                    "question": task.question,
                    "executor_type": str(context.config.executor_type),
                },
            )
        ]
    )


def _make_config(
    *,
    job_id: str = "test-job",
    model_name: str = "test-model",
    max_concurrency: int = 4,
    executor_type: Literal[ExecutorType.THREAD, ExecutorType.PROCESS] = ExecutorType.THREAD,
    batch_size: int = 2,
    num_samples_per_task: int = 1,
    num_batches: int | None = 1,
    allowed_staleness: int = 0,
    drop_stale_trajectories: bool = False,
    num_rollout_retries: int = 0,
    raise_on_rollout_failure: bool = False,
    status_polling_interval: float = 1.0,
    job_init_timeout: int | None = None,
) -> RolloutConfig:
    return RolloutConfig(
        job_id=job_id,
        model_name=model_name,
        max_concurrency=max_concurrency,
        executor_type=executor_type,
        batch_size=batch_size,
        num_samples_per_task=num_samples_per_task,
        num_batches=num_batches,
        allowed_staleness=allowed_staleness,
        drop_stale_trajectories=drop_stale_trajectories,
        num_rollout_retries=num_rollout_retries,
        raise_on_rollout_failure=raise_on_rollout_failure,
        status_polling_interval=status_polling_interval,
        job_init_timeout=job_init_timeout,
    )


def _make_api_client() -> MagicMock:
    client = MagicMock()
    client.v1alpha1.fine_tuning.jobs.get_runtime_status.return_value = JobStatus(
        object="fine_tuning.job.status",
        inference_version=0,
        total_batches=None,
        filled_batches=0,
    )
    client.v1alpha1.fine_tuning.jobs.batches.get.side_effect = NotFoundError("not found")
    client.v1alpha1.fine_tuning.jobs.batches.create.return_value = Batch(
        object="fine_tuning.batch",
        index=0,
        sample_count=0,
    )
    client.v1alpha1.fine_tuning.jobs.batches.submit_samples.return_value = Batch(
        object="fine_tuning.batch",
        index=0,
        sample_count=0,
    )
    return client


def _make_runner(
    api_client=None,
    config=None,
    rollout_fn=None,
    dataset=None,
) -> RolloutRunner[FakeTaskSpec]:
    return RolloutRunner(  # ty: ignore[invalid-return-type]
        api_client=api_client or _make_api_client(),
        config=config or _make_config(),
        rollout_fn=rollout_fn or MagicMock(return_value=SampleGroup(samples=[_make_sample()])),
        dataset=dataset or FakeDataset(),  # ty: ignore[invalid-argument-type]
    )


# ===========================================================================
# RolloutRunner._init_batch
# ===========================================================================


class TestInitBatch:
    def test_creates_new_batch_when_not_found(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.return_value = JobStatus(
            object="fine_tuning.job.status",
            inference_version=0,
            total_batches=None,
            filled_batches=0,
        )
        client.v1alpha1.fine_tuning.jobs.batches.get.side_effect = NotFoundError("nope")

        runner = _make_runner(api_client=client)
        batch_idx, n_samples = runner._load_initial_state()

        assert batch_idx == 0
        assert n_samples == 0
        client.v1alpha1.fine_tuning.jobs.batches.create.assert_called_once_with(job_id="test-job", index=0)

    def test_resumes_existing_batch(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.return_value = JobStatus(
            object="fine_tuning.job.status",
            inference_version=2,
            total_batches=None,
            filled_batches=3,
        )
        client.v1alpha1.fine_tuning.jobs.batches.get.side_effect = None
        client.v1alpha1.fine_tuning.jobs.batches.get.return_value = Batch(
            object="fine_tuning.batch",
            index=3,
            sample_count=5,
        )

        runner = _make_runner(api_client=client)
        batch_idx, n_samples = runner._load_initial_state()

        assert batch_idx == 3
        assert n_samples == 5
        client.v1alpha1.fine_tuning.jobs.batches.create.assert_not_called()

    def test_creates_batch_at_filled_batches_index(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.return_value = JobStatus(
            object="fine_tuning.job.status",
            inference_version=0,
            total_batches=None,
            filled_batches=7,
        )
        client.v1alpha1.fine_tuning.jobs.batches.get.side_effect = NotFoundError("nope")

        runner = _make_runner(api_client=client)
        batch_idx, n_samples = runner._load_initial_state()

        assert batch_idx == 7
        client.v1alpha1.fine_tuning.jobs.batches.create.assert_called_once_with(job_id="test-job", index=7)


# ===========================================================================
# RolloutRunner._wait_for_job_initialization
# ===========================================================================


class TestWaitForJobInitialization:
    def test_waits_until_success_when_timeout_is_none(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.side_effect = [
            ConnectionError("network error"),
            JobStatus(
                object="fine_tuning.job.status",
                inference_version=7,
                total_batches=None,
                filled_batches=0,
            ),
        ]
        runner = _make_runner(
            api_client=client,
            config=_make_config(job_init_timeout=None, status_polling_interval=0),
        )

        assert runner._wait_for_job_initialization() is None
        assert client.v1alpha1.fine_tuning.jobs.get_runtime_status.call_count == 2

    def test_returns_after_first_successful_status_response(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.side_effect = [
            JobStatus(
                object="fine_tuning.job.status",
                inference_version=None,
                total_batches=None,
                filled_batches=0,
            ),
            JobStatus(
                object="fine_tuning.job.status",
                inference_version=7,
                total_batches=None,
                filled_batches=0,
            ),
        ]
        runner = _make_runner(
            api_client=client,
            config=_make_config(job_init_timeout=1, status_polling_interval=0.01),
        )

        assert runner._wait_for_job_initialization() is None
        assert client.v1alpha1.fine_tuning.jobs.get_runtime_status.call_count == 1

    def test_recovers_from_transient_errors_before_success(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.side_effect = [
            ConnectionError("network error"),
            JobStatus(
                object="fine_tuning.job.status",
                inference_version=None,
                total_batches=None,
                filled_batches=0,
            ),
        ]
        runner = _make_runner(
            api_client=client,
            config=_make_config(job_init_timeout=1, status_polling_interval=0.01),
        )

        assert runner._wait_for_job_initialization() is None
        assert client.v1alpha1.fine_tuning.jobs.get_runtime_status.call_count == 2

    def test_raises_when_no_successful_status_response_arrives(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.side_effect = ConnectionError("network error")
        runner = _make_runner(
            api_client=client,
            config=_make_config(job_init_timeout=0, status_polling_interval=0.01),
        )

        with pytest.raises(
            JobInitializationTimeout,
            match="did not become ready within 0 seconds",
        ):
            runner._wait_for_job_initialization()

    def test_raises_with_original_polling_error_as_cause(self):
        client = _make_api_client()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.side_effect = ConnectionError("network error")
        runner = _make_runner(
            api_client=client,
            config=_make_config(job_init_timeout=0, status_polling_interval=0.01),
        )

        with pytest.raises(JobInitializationTimeout) as exc_info:
            runner._wait_for_job_initialization()

        assert isinstance(exc_info.value.__cause__, ConnectionError)
        assert exc_info.value.last_exception is exc_info.value.__cause__


# ===========================================================================
# RolloutRunner._process_rollout_result
# ===========================================================================


class TestProcessRolloutResult:
    def test_accepted_result_adds_to_scheduler(self):
        runner = _make_runner()
        runner._scheduler = MagicMock()

        sample = _make_sample()
        group = SampleGroup(samples=[sample])
        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)

        runner._process_rollout_result(task=task, sample_group=group)

        runner._scheduler.add_sample_group.assert_called_once_with(task_id=task.id, samples=[sample])
        assert sample.debug_info["starting_inference_version"] == 0

    def test_rejected_result_removes_task(self):
        runner = _make_runner()
        runner._scheduler = MagicMock()

        group = SampleGroup(is_rejected=True)
        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)

        runner._process_rollout_result(task=task, sample_group=group)

        runner._scheduler.remove_task.assert_called_once_with(task_id=task.id)
        runner._scheduler.add_sample_group.assert_not_called()

    def test_debug_info_set_on_all_samples(self):
        runner = _make_runner(config=_make_config(batch_size=6, num_samples_per_task=3))
        runner._scheduler = MagicMock()

        samples = [_make_sample() for _ in range(3)]
        group = SampleGroup(samples=samples)
        task = Task(spec=FakeTaskSpec(), starting_inference_version=5)

        runner._process_rollout_result(task=task, sample_group=group)

        for s in samples:
            assert s.debug_info["starting_inference_version"] == 5


# ===========================================================================
# RolloutRunner._submit_samples
# ===========================================================================


class TestSubmitSamples:
    def test_submits_ready_samples_to_api(self):
        client = _make_api_client()
        runner = _make_runner(api_client=client)
        runner._batch_idx = 2
        runner._batch_n_samples = 0

        sample = _make_sample()
        runner._scheduler = MagicMock()
        runner._scheduler.get_ready_samples.return_value = [sample]

        runner._submit_samples()

        client.v1alpha1.fine_tuning.jobs.batches.submit_samples.assert_called_once()
        call_kwargs = client.v1alpha1.fine_tuning.jobs.batches.submit_samples.call_args
        assert call_kwargs.kwargs["job_id"] == "test-job"
        assert call_kwargs.kwargs["batch_index"] == 2
        assert len(call_kwargs.kwargs["samples"]) == 1
        assert runner._batch_n_samples == 1

    def test_no_submission_when_no_ready_samples(self):
        client = _make_api_client()
        runner = _make_runner(api_client=client)
        runner._scheduler = MagicMock()
        runner._scheduler.get_ready_samples.return_value = []

        runner._submit_samples()

        client.v1alpha1.fine_tuning.jobs.batches.submit_samples.assert_not_called()

    def test_batch_boundary_creates_new_batch(self):
        client = _make_api_client()
        config = _make_config(batch_size=2, num_batches=3)
        runner = _make_runner(api_client=client, config=config)
        runner._batch_idx = 0
        runner._batch_n_samples = 1  # 1 already submitted, batch_size=2

        runner._scheduler = MagicMock()
        runner._scheduler.get_ready_samples.return_value = [_make_sample()]

        runner._submit_samples()

        assert runner._batch_idx == 1
        assert runner._batch_n_samples == 0
        client.v1alpha1.fine_tuning.jobs.batches.create.assert_called_once_with(job_id="test-job", index=1)

    def test_final_batch_does_not_create_next_batch(self):
        client = _make_api_client()
        config = _make_config(batch_size=2, num_batches=3)
        runner = _make_runner(api_client=client, config=config)
        runner._batch_idx = 2  # filling the last batch (num_batches=3)
        runner._batch_n_samples = 1

        runner._scheduler = MagicMock()
        runner._scheduler.get_ready_samples.return_value = [_make_sample()]

        runner._submit_samples()

        assert runner._batch_idx == 3
        assert runner._batch_n_samples == 0
        client.v1alpha1.fine_tuning.jobs.batches.create.assert_not_called()

    def test_unbounded_num_batches_always_creates_next_batch(self):
        client = _make_api_client()
        config = _make_config(batch_size=2, num_batches=None)
        runner = _make_runner(api_client=client, config=config)
        runner._batch_idx = 0
        runner._batch_n_samples = 1

        runner._scheduler = MagicMock()
        runner._scheduler.get_ready_samples.return_value = [_make_sample()]

        runner._submit_samples()

        assert runner._batch_idx == 1
        client.v1alpha1.fine_tuning.jobs.batches.create.assert_called_once_with(job_id="test-job", index=1)

    def test_no_new_batch_when_not_full(self):
        client = _make_api_client()
        config = _make_config(batch_size=4)
        runner = _make_runner(api_client=client, config=config)
        runner._batch_idx = 0
        runner._batch_n_samples = 0

        runner._scheduler = MagicMock()
        runner._scheduler.get_ready_samples.return_value = [_make_sample()]

        runner._submit_samples()

        assert runner._batch_idx == 0
        assert runner._batch_n_samples == 1
        client.v1alpha1.fine_tuning.jobs.batches.create.assert_not_called()


# ===========================================================================
# RolloutRunner._spawn_tasks
# ===========================================================================


class TestSpawnTasks:
    def test_spawns_tasks_when_scheduler_allows(self):
        runner = _make_runner(config=_make_config(num_samples_per_task=1))
        runner._scheduler = MagicMock()
        runner._scheduler.get_num_rollouts_to_spawn.return_value = 3
        runner._scheduler.inference_version = 0

        runner._spawn_tasks()

        assert runner._dispatcher_queue.qsize() == 3
        runner._scheduler.add_tasks.assert_called_once()
        tasks = runner._scheduler.add_tasks.call_args.kwargs["tasks"]
        assert len(tasks) == 3

    def test_no_spawn_when_zero_rollouts(self):
        runner = _make_runner()
        runner._scheduler = MagicMock()
        runner._scheduler.get_num_rollouts_to_spawn.return_value = 0

        runner._spawn_tasks()

        assert runner._dispatcher_queue.qsize() == 0
        runner._scheduler.add_tasks.assert_not_called()

    def test_tasks_have_correct_inference_version(self):
        runner = _make_runner(config=_make_config(num_samples_per_task=1))
        runner._scheduler = MagicMock()
        runner._scheduler.get_num_rollouts_to_spawn.return_value = 2
        runner._scheduler.inference_version = 42

        runner._spawn_tasks()

        tasks = runner._scheduler.add_tasks.call_args.kwargs["tasks"]
        for t in tasks:
            assert t.starting_inference_version == 42

    def test_samples_dataset_with_correct_n(self):
        dataset = MagicMock()
        dataset.sample.return_value = [FakeTaskSpec(), FakeTaskSpec()]
        runner = _make_runner(
            dataset=dataset,
            config=_make_config(num_samples_per_task=2),
        )
        runner._scheduler = MagicMock()
        runner._scheduler.get_num_rollouts_to_spawn.return_value = 4
        runner._scheduler.inference_version = 0

        runner._spawn_tasks()

        # 4 rollouts // 2 samples_per_task = 2 tasks to sample
        dataset.sample.assert_called_once_with(2)


# ===========================================================================
# RolloutRunner.run (integration with mocked threads)
# ===========================================================================


class TestRunLoop:
    def _run_with_events(self, runner, events):
        """Feed events into the input queue and run the loop.
        Patches the threads so they don't actually start."""
        job_status_tracker = MagicMock()
        rollout_dispatcher = MagicMock()
        runner._job_status_tracker = job_status_tracker
        runner._rollout_dispatcher = rollout_dispatcher

        for event in events:
            runner._input_queue.put(event)

        runner.run()
        return job_status_tracker, rollout_dispatcher

    def test_full_happy_path(self):
        config = _make_config(batch_size=2, num_batches=1, num_samples_per_task=1)
        client = _make_api_client()
        runner = _make_runner(api_client=client, config=config)

        # Inject a version update followed by two rollout results
        task1 = Task(id=TaskID("task-001"), spec=FakeTaskSpec(), starting_inference_version=0)
        task2 = Task(id=TaskID("task-002"), spec=FakeTaskSpec(), starting_inference_version=0)

        # We need to control the scheduler to complete in 2 results.
        # Easiest: let the real scheduler run but seed events properly.
        # The JobStatusUpdate sets the inference version, _spawn_tasks
        # will sample from dataset and enqueue — but we also need
        # RolloutResults for those tasks. Instead, mock the scheduler.

        mock_scheduler = MagicMock()
        # First call (after JobStatusUpdate): spawn 2 rollouts
        # Second call (after RolloutResult 1): spawn 0
        # Third call (after RolloutResult 2): spawn 0
        mock_scheduler.get_num_rollouts_to_spawn.return_value = 0
        mock_scheduler.inference_version = 0
        mock_scheduler.get_ready_samples.side_effect = [
            [_make_sample()],  # after result 1
            [_make_sample()],  # after result 2
        ]
        mock_scheduler.is_all_finished = False

        def set_finished_after_two(*args, **kwargs):
            # call_count includes the current call (incremented before side_effect)
            if mock_scheduler.add_sample_group.call_count >= 2:
                mock_scheduler.is_all_finished = True

        mock_scheduler.add_sample_group.side_effect = set_finished_after_two

        runner._scheduler = mock_scheduler

        events = [
            JobStatusUpdate(inference_version=0),
            RolloutResult(
                task=task1,
                sample_group=SampleGroup(samples=[_make_sample()]),
            ),
            RolloutResult(
                task=task2,
                sample_group=SampleGroup(samples=[_make_sample()]),
            ),
        ]

        job_status_tracker, rollout_dispatcher = self._run_with_events(runner, events)

        assert mock_scheduler.set_inference_version.call_count == 1
        assert mock_scheduler.add_sample_group.call_count == 2
        assert client.v1alpha1.fine_tuning.jobs.batches.submit_samples.call_count == 2
        job_status_tracker.stop.assert_called_once()
        rollout_dispatcher.stop.assert_called_once()

    def test_rollout_exception_propagates(self):
        runner = _make_runner()
        runner._scheduler = MagicMock()

        error = ValueError("boom")
        events = [
            RolloutException(exception=error, traceback="traceback text"),
        ]

        with pytest.raises(ValueError, match="boom"):
            self._run_with_events(runner, events)

    def test_rejected_result_does_not_submit(self):
        config = _make_config(batch_size=2, num_batches=1, num_samples_per_task=1)
        client = _make_api_client()
        runner = _make_runner(api_client=client, config=config)

        mock_scheduler = MagicMock()
        mock_scheduler.get_num_rollouts_to_spawn.return_value = 0
        mock_scheduler.inference_version = 0
        # After rejected: get_ready_samples returns nothing
        # After accepted: returns sample, then mark finished
        mock_scheduler.get_ready_samples.side_effect = [
            [],  # after rejected
            [_make_sample(), _make_sample()],  # after accepted (fills batch)
        ]
        mock_scheduler.is_all_finished = False

        def mark_finished(*args, **kwargs):
            # Mark finished after the accepted result
            if mock_scheduler.add_sample_group.call_count >= 1:
                mock_scheduler.is_all_finished = True

        mock_scheduler.add_sample_group.side_effect = mark_finished
        runner._scheduler = mock_scheduler

        task_rejected = Task(id=TaskID("task-r"), spec=FakeTaskSpec(), starting_inference_version=0)
        task_ok = Task(id=TaskID("task-ok"), spec=FakeTaskSpec(), starting_inference_version=0)

        events = [
            RolloutResult(
                task=task_rejected,
                sample_group=SampleGroup(is_rejected=True),
            ),
            RolloutResult(
                task=task_ok,
                sample_group=SampleGroup(samples=[_make_sample()]),
            ),
        ]

        self._run_with_events(runner, events)

        mock_scheduler.remove_task.assert_called_once_with(task_id=task_rejected.id)
        # First call was after rejected (empty), second after accepted
        assert client.v1alpha1.fine_tuning.jobs.batches.submit_samples.call_count == 1

    def test_job_status_update_sets_version_and_spawns(self):
        runner = _make_runner()
        mock_scheduler = MagicMock()
        mock_scheduler.get_num_rollouts_to_spawn.return_value = 0
        mock_scheduler.inference_version = 0
        runner._scheduler = mock_scheduler

        # Feed two status updates, then a result that finishes
        task = Task(id=TaskID("task-1"), spec=FakeTaskSpec(), starting_inference_version=0)
        mock_scheduler.get_ready_samples.return_value = [_make_sample(), _make_sample()]
        mock_scheduler.is_all_finished = False

        def finish_on_add(*a, **kw):
            mock_scheduler.is_all_finished = True

        mock_scheduler.add_sample_group.side_effect = finish_on_add

        events = [
            JobStatusUpdate(inference_version=0),
            JobStatusUpdate(inference_version=1),
            RolloutResult(
                task=task,
                sample_group=SampleGroup(samples=[_make_sample()]),
            ),
        ]
        self._run_with_events(runner, events)

        calls = mock_scheduler.set_inference_version.call_args_list
        assert calls == [call(inference_version=0), call(inference_version=1)]


# ===========================================================================
# Bug: pending drain crosses batch boundary
# ===========================================================================


class TestDroplessPendingDrainBatchBoundary:
    """When DroplessScheduler drains buffered (pending) samples after a
    straggler completes, get_ready_samples() can return more samples than
    fit in the runner's current batch.  The runner submits them all under
    one batch_index and the == batch_size transition check is skipped,
    so subsequent batches are never created on the API.

    Scenario (batch_size=1, num_samples_per_task=1, allowed_staleness=0):

      v0: t0          v1: t1, t2

    1. t1 completes first → buffered (v0 task t0 reserves the only slot)
    2. t2 completes      → accepted into batch 0, runner transitions to batch 1
    3. t0 completes      → accepted, AND pending t1 drains too
       get_ready_samples() returns [t0, t1] (2 samples) to the runner
    4. Runner submits both to batch_idx=1, _batch_n_samples jumps to 2.
       2 != batch_size(1) → no transition → batch 2 is never created.
    """

    def test_samples_spanning_batch_boundary_are_split(self):
        config = _make_config(
            batch_size=1,
            num_batches=4,
            num_samples_per_task=1,
            max_concurrency=4,
            allowed_staleness=0,
            drop_stale_trajectories=False,
        )
        client = _make_api_client()
        runner = _make_runner(api_client=client, config=config)

        # Initialise scheduler and runner batch state as _init_batch would.
        runner._batch_idx = 0
        runner._batch_n_samples = 0
        runner._scheduler.set_initial_batch_index(batch_index=0)

        # --- register tasks with the real DroplessRolloutScheduler ---
        runner._scheduler.set_inference_version(inference_version=0)
        t0 = Task(id=TaskID("task-t0"), spec=FakeTaskSpec(), starting_inference_version=0)
        runner._scheduler.add_tasks(tasks=[t0])

        runner._scheduler.set_inference_version(inference_version=1)
        t1 = Task(id=TaskID("task-t1"), spec=FakeTaskSpec(), starting_inference_version=1)
        t2 = Task(id=TaskID("task-t2"), spec=FakeTaskSpec(), starting_inference_version=1)
        runner._scheduler.add_tasks(tasks=[t1, t2])

        # --- complete tasks in an order that triggers pending buffering ---

        # t1 completes (v1) → buffered because t0 (v0) reserves the slot
        runner._process_rollout_result(task=t1, sample_group=SampleGroup(samples=[_make_sample()]))
        runner._submit_samples()
        assert client.v1alpha1.fine_tuning.jobs.batches.submit_samples.call_count == 0

        # t2 completes (v1) → accepted into batch 0
        runner._process_rollout_result(task=t2, sample_group=SampleGroup(samples=[_make_sample()]))
        runner._submit_samples()
        assert runner._batch_idx == 1, "batch 0 should be full, runner on batch 1"

        # t0 completes (v0) → accepted, AND pending t1 drains.
        # The scheduler returns both samples in one get_ready_samples() call.
        runner._process_rollout_result(task=t0, sample_group=SampleGroup(samples=[_make_sample()]))
        runner._submit_samples()

        # --- assertions ---
        submit_calls = client.v1alpha1.fine_tuning.jobs.batches.submit_samples.call_args_list
        batch_indices_submitted = [c.kwargs["batch_index"] for c in submit_calls]
        sample_counts = [len(c.kwargs["samples"]) for c in submit_calls]

        # All 3 samples should have been submitted.
        assert sum(sample_counts) == 3

        # Samples after batch 0 should be split: 1 to batch 1, 1 to batch 2.
        # The bug: both land on batch 1 in a single call, _batch_n_samples
        # overshoots, and batch 2 is never reached.
        assert 2 in batch_indices_submitted, (
            f"Expected a submission to batch 2, but submissions went to "
            f"batches {batch_indices_submitted} with counts {sample_counts}. "
            f"Runner batch_idx stuck at {runner._batch_idx}."
        )


# ===========================================================================
# JobStatusTracker
# ===========================================================================


class TestJobStatusTracker:
    def test_posts_update_when_inference_version_present(self):
        client = MagicMock()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.return_value = JobStatus(
            object="fine_tuning.job.status",
            inference_version=5,
            total_batches=None,
            filled_batches=0,
        )
        output_queue: TrackerQueue = queue.Queue()

        tracker = JobStatusTracker(
            job_id="j1",
            api_client=client,
            output_queue=output_queue,
            poll_interval=0.01,
        )
        tracker.start()
        time.sleep(0.1)
        tracker.stop()
        tracker.join(timeout=1)

        assert not output_queue.empty()
        update = output_queue.get_nowait()
        assert isinstance(update, JobStatusUpdate)
        assert update.inference_version == 5

    def test_skips_when_inference_version_none(self):
        client = MagicMock()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.return_value = JobStatus(
            object="fine_tuning.job.status",
            inference_version=None,
            total_batches=None,
            filled_batches=0,
        )
        output_queue: TrackerQueue = queue.Queue()

        tracker = JobStatusTracker(
            job_id="j1",
            api_client=client,
            output_queue=output_queue,
            poll_interval=0.01,
        )
        tracker.start()
        time.sleep(0.1)
        tracker.stop()
        tracker.join(timeout=1)

        assert output_queue.empty()

    def test_continues_after_api_error(self):
        client = MagicMock()
        call_count = 0

        def flaky_status(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("network error")
            return JobStatus(
                object="fine_tuning.job.status",
                inference_version=1,
                total_batches=None,
                filled_batches=0,
            )

        client.v1alpha1.fine_tuning.jobs.get_runtime_status.side_effect = flaky_status
        output_queue: TrackerQueue = queue.Queue()

        tracker = JobStatusTracker(
            job_id="j1",
            api_client=client,
            output_queue=output_queue,
            poll_interval=0.01,
        )
        tracker.start()
        time.sleep(0.3)
        tracker.stop()
        tracker.join(timeout=1)

        # Should have recovered and posted at least one update
        assert not output_queue.empty()

    def test_stops_cleanly(self):
        client = MagicMock()
        client.v1alpha1.fine_tuning.jobs.get_runtime_status.return_value = JobStatus(
            object="fine_tuning.job.status",
            inference_version=0,
            total_batches=None,
            filled_batches=0,
        )
        output_queue: TrackerQueue = queue.Queue()

        tracker = JobStatusTracker(
            job_id="j1",
            api_client=client,
            output_queue=output_queue,
            poll_interval=0.01,
        )
        tracker.start()
        assert tracker.is_alive()
        tracker.stop()
        tracker.join(timeout=2)
        assert not tracker.is_alive()


# ===========================================================================
# RolloutDispatcher
# ===========================================================================


class TestRolloutDispatcher:
    def _make_dispatcher(
        self,
        rollout_fn=None,
        num_rollout_retries=0,
        raise_on_rollout_failure=False,
        concurrent_workers=2,
        executor_type: str = ExecutorType.THREAD,
        context: RolloutContext = MagicMock()
    ) -> tuple[RolloutDispatcher[FakeTaskSpec], DispatcherInputQueue, DispatcherOutputQueue]:
        context.config.executor_type = executor_type
        input_queue: DispatcherInputQueue = queue.Queue()
        output_queue: DispatcherOutputQueue = queue.Queue()

        dispatcher = RolloutDispatcher(
            rollout_fn=rollout_fn or MagicMock(return_value=SampleGroup(samples=[])),
            context=context,
            input_queue=input_queue,
            output_queue=output_queue,
            concurrent_workers=concurrent_workers,
            num_rollout_retries=num_rollout_retries,
            raise_on_rollout_failure=raise_on_rollout_failure,
        )
        return dispatcher, input_queue, output_queue

    def test_successful_rollout_posts_result(self):
        sample_group = SampleGroup(samples=[_make_sample()])
        rollout_fn = MagicMock(return_value=sample_group)

        dispatcher, in_q, out_q = self._make_dispatcher(rollout_fn=rollout_fn)
        dispatcher.start()

        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)
        in_q.put(task)
        time.sleep(0.3)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        result = out_q.get_nowait()
        assert isinstance(result, RolloutResult)
        assert result.task.id == task.id
        assert result.sample_group is sample_group

    def test_failed_rollout_no_retries_rejected(self):
        rollout_fn = MagicMock(side_effect=RuntimeError("fail"))

        dispatcher, in_q, out_q = self._make_dispatcher(
            rollout_fn=rollout_fn,
            num_rollout_retries=0,
            raise_on_rollout_failure=False,
        )
        dispatcher.start()

        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)
        in_q.put(task)
        time.sleep(0.3)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        result = out_q.get_nowait()
        assert isinstance(result, RolloutResult)
        assert result.sample_group.is_rejected is True

    def test_failed_rollout_raises_exception(self):
        rollout_fn = MagicMock(side_effect=RuntimeError("fail"))

        dispatcher, in_q, out_q = self._make_dispatcher(
            rollout_fn=rollout_fn,
            num_rollout_retries=0,
            raise_on_rollout_failure=True,
        )
        dispatcher.start()

        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)
        in_q.put(task)
        time.sleep(0.3)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        result = out_q.get_nowait()
        assert isinstance(result, RolloutException)
        assert isinstance(result.exception, RuntimeError)

    def test_retries_on_failure_then_succeeds(self):
        sample_group = SampleGroup(samples=[_make_sample()])
        call_count = 0

        def flaky_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient")
            return sample_group

        dispatcher, in_q, out_q = self._make_dispatcher(
            rollout_fn=flaky_fn,
            num_rollout_retries=2,
        )
        dispatcher.start()

        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)
        in_q.put(task)
        time.sleep(0.5)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        result = out_q.get_nowait()
        assert isinstance(result, RolloutResult)
        assert result.sample_group is sample_group
        assert call_count == 3

    def test_retries_exhausted_then_rejected(self):
        rollout_fn = MagicMock(side_effect=RuntimeError("always fail"))

        dispatcher, in_q, out_q = self._make_dispatcher(
            rollout_fn=rollout_fn,
            num_rollout_retries=2,
            raise_on_rollout_failure=False,
        )
        dispatcher.start()

        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)
        in_q.put(task)
        time.sleep(0.5)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        result = out_q.get_nowait()
        assert isinstance(result, RolloutResult)
        assert result.sample_group.is_rejected is True
        # 1 initial + 2 retries = 3 total calls
        assert rollout_fn.call_count == 3

    def test_retries_exhausted_then_raises(self):
        rollout_fn = MagicMock(side_effect=ValueError("permanent"))

        dispatcher, in_q, out_q = self._make_dispatcher(
            rollout_fn=rollout_fn,
            num_rollout_retries=1,
            raise_on_rollout_failure=True,
        )
        dispatcher.start()

        task = Task(spec=FakeTaskSpec(), starting_inference_version=0)
        in_q.put(task)
        time.sleep(0.5)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        result = out_q.get_nowait()
        assert isinstance(result, RolloutException)
        assert isinstance(result.exception, ValueError)
        assert rollout_fn.call_count == 2  # 1 initial + 1 retry

    def test_dispatches_multiple_tasks(self):
        rollout_fn = MagicMock(return_value=SampleGroup(samples=[]))

        dispatcher, in_q, out_q = self._make_dispatcher(rollout_fn=rollout_fn, concurrent_workers=4)
        dispatcher.start()

        for i in range(5):
            task = Task(spec=FakeTaskSpec(question=f"q{i}"), starting_inference_version=0)
            in_q.put(task)

        time.sleep(0.5)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        results = []
        while not out_q.empty():
            results.append(out_q.get_nowait())

        assert len(results) == 5
        assert all(isinstance(r, RolloutResult) for r in results)

    def test_stops_cleanly(self):
        dispatcher, in_q, out_q = self._make_dispatcher()
        dispatcher.start()
        assert dispatcher.is_alive()
        dispatcher.stop()
        dispatcher.join(timeout=2)
        assert not dispatcher.is_alive()

    def test_passes_task_spec_and_context(self):
        rollout_fn = MagicMock(return_value=SampleGroup(samples=[]))
        context = MagicMock()
        context.config.executor_type = ExecutorType.THREAD
        in_q: DispatcherInputQueue = queue.Queue()
        out_q: DispatcherOutputQueue = queue.Queue()

        dispatcher = RolloutDispatcher(
            rollout_fn=rollout_fn,
            context=context,
            input_queue=in_q,
            output_queue=out_q,
            concurrent_workers=1,
            num_rollout_retries=0,
            raise_on_rollout_failure=False,
        )
        dispatcher.start()

        spec = FakeTaskSpec(question="hello")
        task = Task(spec=spec, starting_inference_version=0)
        in_q.put(task)
        time.sleep(0.3)
        dispatcher.stop()
        dispatcher.join(timeout=2)

        rollout_fn.assert_called_once_with(task=spec, context=context)

    def test_uses_thread_pool_executor_for_thread_executor_type(self):
        dispatcher, in_q, out_q = self._make_dispatcher()

        try:
            assert isinstance(dispatcher._executor, concurrent.futures.ThreadPoolExecutor)
        finally:
            dispatcher._executor.shutdown(wait=False, cancel_futures=True)

    def test_uses_process_pool_executor_for_process_executor_type(self):
        dispatcher, in_q, out_q = self._make_dispatcher(executor_type=ExecutorType.PROCESS)

        try:
            assert isinstance(dispatcher._executor, concurrent.futures.ProcessPoolExecutor)
        finally:
            dispatcher._executor.shutdown(wait=False, cancel_futures=True)

    def test_process_executor_posts_result(self):
        dispatcher, in_q, out_q = self._make_dispatcher(
            rollout_fn=_process_smoke_rollout_fn,
            raise_on_rollout_failure=True,
            concurrent_workers=1,
            executor_type=ExecutorType.PROCESS,
            context=RolloutContext(
                config=_make_config(executor_type=ExecutorType.PROCESS),
                api_client=TokenFactory(api_key="test-key"),
            )
        )

        dispatcher.start()
        try:
            task = Task(spec=ProcessSmokeTask(question="hello"), starting_inference_version=0)
            in_q.put(task)
            result = out_q.get(timeout=10)
        finally:
            dispatcher.stop()
            dispatcher.join(timeout=5)

        assert isinstance(result, RolloutResult)
        assert result.task.id == task.id
        assert result.sample_group.samples[0].debug_info == {
            "question": "hello",
            "executor_type": "process",
        }
