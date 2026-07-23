import heapq
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, NonNegativeInt, PositiveInt, model_validator


logger = logging.getLogger(__name__)


class RolloutSchedulerConfig(BaseModel):
    batch_size: PositiveInt
    num_batches: NonNegativeInt | None = None
    num_samples_per_task: PositiveInt
    parallel_rollouts: PositiveInt
    allowed_staleness: NonNegativeInt | None
    drop_stale_trajectories: bool

    @model_validator(mode="after")
    def _check_batch_size_divisible(self) -> "RolloutSchedulerConfig":
        if self.batch_size % self.num_samples_per_task != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"num_samples_per_task ({self.num_samples_per_task})"
            )
        return self


def create_rollout_scheduler(
    config: RolloutSchedulerConfig,
) -> "BaseRolloutScheduler[Any, Any]":
    if config.allowed_staleness is None:
        return AnyStalenessRolloutScheduler(config=config)
    if config.drop_stale_trajectories and config.allowed_staleness is not None:
        return DropStaleRolloutScheduler(config=config)
    if not config.drop_stale_trajectories and config.allowed_staleness is not None:
        return DroplessRolloutScheduler(config=config)
    raise NotImplementedError("Unsupported rollout scheduler configuration")


TaskID = TypeVar("TaskID")


class WithID(Protocol[TaskID]):
    id: TaskID


Sample = TypeVar("Sample")


class BaseRolloutScheduler(Generic[TaskID, Sample], ABC):  # ruff:ignore[generic-not-last-base-class]
    """Rollout scheduler is responsible for deciding how many new rollouts it is
    possible to spawn at each moment, and for returning completed samples in
    such an order that doesn't break constraints of a scheduler.

    Scheduler keeps track of started tasks and their inference versions,
    and uses this information to decide whether samples from a completed
    task can be added to the ready samples or should be dropped or delayed.

    To initiate a scheduler, set starting batch index and inference version:

        scheduler.set_initial_batch_index(batch_index=0)
        scheduler.set_inference_version(inference_version=0)

    Get the number of rollouts that can be spawned at the moment:

        num_rollouts = scheduler.get_num_rollouts_to_spawn()

    Call this method when inference version is updated or completed samples received.

    As new rollouts are started, add corresponding tasks to the scheduler:

        scheduler.add_tasks(tasks=[task1, task2, ...])

    When samples are ready, add them to the scheduler:

        scheduler.add_samples(samples=[sample1, sample2, ...])

    After adding processed samples, get samples that are ready to be sent to training:

        ready_samples = scheduler.get_ready_samples()
    """

    def __init__(self, config: RolloutSchedulerConfig):
        self._config = config

        self._batch_index: int | None = None
        self._batch_remainder = config.batch_size
        self._inference_version: int | None = None
        self._initial_batch_index: int | None = None

        self._tasks_by_inference_version: dict[int, set[TaskID]] = defaultdict(set)
        self._task_to_inference_version: dict[TaskID, int] = {}
        self._total_spawned_tasks: int = 0

        self._ready_samples: list[Sample] = []

    @abstractmethod
    def _consume_sample_group(self, task_id: TaskID, sample_group: Sequence[Sample]):
        """Process a group of samples corresponding to a single task. Concrete
        implementation of this method should populate `_ready_samples` and remove
        the task from internal tracking structures as needed.
        """

    @abstractmethod
    def _get_num_rollouts_to_spawn(self) -> int:
        """Calculate the number of rollouts that can be spawned at the current moment,
        based on internal state and limitations of a concrete scheduler.
        """

    @property
    def is_spawning_finished(self) -> bool:
        return self._total_remaining_rollouts is not None and self._total_remaining_rollouts == 0

    @property
    def is_all_finished(self) -> bool:
        return (
            self._config.num_batches is not None
            and self._batch_index is not None
            and self._batch_index == self._config.num_batches
        )

    @property
    def inference_version(self) -> int | None:
        return self._inference_version

    @property
    def current_batch_index(self) -> int | None:
        return self._batch_index

    def set_initial_batch_index(self, batch_index: int, n_samples: int | None = None) -> None:
        if self._batch_index is None:
            self._batch_index = batch_index
            self._initial_batch_index = batch_index
            if n_samples is not None:  # todo unit test
                if n_samples > self._config.batch_size:
                    raise ValueError("n_samples cannot be greater than batch size")
                self._batch_remainder = self._config.batch_size - n_samples
        else:
            raise RuntimeError("Batch index can be set only once")

    def set_inference_version(self, inference_version: int) -> None:
        if self._inference_version is not None and self._inference_version > inference_version:
            raise RuntimeError("New inference version must be greater or equal the current one")
        self._inference_version = inference_version

    def get_num_rollouts_to_spawn(self) -> int:
        limit = self._total_remaining_rollouts
        desired = self._get_num_rollouts_to_spawn()
        if limit is not None:
            return min(desired, limit)
        return desired

    def add_tasks(self, tasks: Iterable[WithID[TaskID]]) -> None:
        if self._inference_version is None:
            raise RuntimeError("Inference version must be set before adding tasks")
        self._tasks_by_inference_version[self._inference_version].update([task.id for task in tasks])
        for task in tasks:
            self._task_to_inference_version[task.id] = self._inference_version
            self._total_spawned_tasks += 1
            remaining = self._total_remaining_rollouts
            if remaining is not None and remaining < 0:
                raise RuntimeError("Added more tasks than allowed by num_batches and batch_size")

    def remove_task(self, task_id: TaskID) -> None:
        ver = self._task_to_inference_version.pop(task_id)  # type: ignore[reportArgumentType]
        self._tasks_by_inference_version[ver].remove(task_id)  # type: ignore[reportArgumentType]
        if len(self._tasks_by_inference_version[ver]) == 0:
            self._tasks_by_inference_version.pop(ver)

    def add_sample_group(self, task_id: TaskID, samples: Sequence[Sample]) -> None:
        assert len(samples) == self._config.num_samples_per_task
        self._consume_sample_group(task_id=task_id, sample_group=samples)

    def get_ready_samples(self) -> list[Sample]:
        ret = self._ready_samples
        self._ready_samples = []
        return ret

    @property
    def _task_num_in_progress(self) -> int:
        return sum(len(tasks) for tasks in self._tasks_by_inference_version.values())

    @property
    def _total_remaining_rollouts(self) -> int | None:
        if self._config.num_batches is not None and self._initial_batch_index is not None:
            return (
                self._config.num_batches - self._initial_batch_index
            ) * self._config.batch_size - self._total_spawned_tasks * self._config.num_samples_per_task
        return None

    def _add_ready_sample_group(self, sample_group: Sequence[Sample]):
        assert len(sample_group) == self._config.num_samples_per_task

        if self._config.num_batches is not None and self._batch_index == self._config.num_batches:
            raise RuntimeError("Cannot add more samples, all batches are processed")

        self._ready_samples.extend(sample_group)
        self._batch_remainder -= len(sample_group)
        if self._batch_remainder == 0:
            assert isinstance(self._batch_index, int)
            self._batch_index += 1
            self._batch_remainder = self._config.batch_size


class AnyStalenessRolloutScheduler(BaseRolloutScheduler[TaskID, Sample], Generic[TaskID, Sample]):
    """Scheduler without staleness checks: all samples go to batch in order
    of appearance. Always spawns the maximum number of trajectories
    that allows `parallel_rollouts`.
    """

    def _consume_sample_group(self, task_id: TaskID, sample_group: Sequence[Sample]):
        self._add_ready_sample_group(sample_group=sample_group)
        self.remove_task(task_id=task_id)

    def _get_num_rollouts_to_spawn(self) -> int:
        return self._config.parallel_rollouts - self._task_num_in_progress * self._config.num_samples_per_task


class DropStaleRolloutScheduler(BaseRolloutScheduler[TaskID, Sample], Generic[TaskID, Sample]):
    """Spawns the maximum number of trajectories that allows `parallel_rollouts`
    and drops samples that exceed `allowed_staleness`.
    """

    def _consume_sample_group(self, task_id: TaskID, sample_group: Sequence[Sample]):
        assert isinstance(self._batch_index, int)
        assert self._config.allowed_staleness is not None
        if self._batch_index - 1 - self._task_to_inference_version[task_id] <= self._config.allowed_staleness:
            self._add_ready_sample_group(sample_group=sample_group)
        else:
            logger.info(f"Dropping sample group for task {task_id} due to staleness")
        self.remove_task(task_id=task_id)

    def _get_num_rollouts_to_spawn(self) -> int:
        return self._config.parallel_rollouts - self._task_num_in_progress * self._config.num_samples_per_task


class DroplessRolloutScheduler(BaseRolloutScheduler[TaskID, Sample], Generic[TaskID, Sample]):
    """Never drops samples. Waits for stragglers when they are about to exceed
    `allowed_staleness` and buffers samples that arrive during that time.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending_sample_groups: list[tuple[int, int, TaskID, Sequence[Sample]]] = []

    def _consume_sample_group(self, task_id: TaskID, sample_group: Sequence[Sample]):
        added = self._process_sample_group(task_id=task_id, sample_group=sample_group)
        self.remove_task(task_id=task_id)
        if added:
            while len(self._pending_sample_groups):
                _, _, pending_task_id, pending_sample_group = self._pending_sample_groups[0]
                added = self._process_sample_group(
                    task_id=pending_task_id,
                    sample_group=pending_sample_group,
                    is_pending=True,
                )
                if added:
                    heapq.heappop(self._pending_sample_groups)
                else:
                    break

    def _get_num_rollouts_to_spawn(self) -> int:
        assert isinstance(self._inference_version, int)
        assert self._config.allowed_staleness is not None
        max_batch = self._inference_version + 1 + self._config.allowed_staleness
        assert isinstance(self._batch_index, int)
        max_budget = (
            (max_batch - self._batch_index) * self._config.batch_size
            + self._batch_remainder
            - len(self._pending_sample_groups) * self._config.num_samples_per_task
        )
        res = (
            min(self._config.parallel_rollouts, max_budget)
            - self._task_num_in_progress * self._config.num_samples_per_task
        )
        logger.debug(
            f"inference_version={self._inference_version} "
            f"batch_index={self._batch_index} "
            f"batch_remainder={self._batch_remainder} {max_budget=} "
            f"num_rollouts_to_spawn={res} "
            f"task_num1={len(self._task_to_inference_version)} "
            f"task_num2={self._task_num_in_progress} "
            f"total_spawned_tasks={self._total_spawned_tasks} "
            f"pending_sample_groups={len(self._pending_sample_groups)} "
        )
        return res

    def _process_sample_group(self, task_id: TaskID, sample_group: Sequence[Sample], is_pending: bool = False) -> bool:
        """For the current batch, we calculate the number of reserved places
        for samples from trajectories started in different inference versions.

        Returns:
            True if samples were added to the list `self._ready_samples`, that is, to the current batch.
            False if samples were added to the heap queue `self._pending_sample_groups` for pending samples.
        """
        reserved_places: deque[tuple[int, int]] = deque()
        if len(self._tasks_by_inference_version) > 0:
            inf_versions = self._tasks_by_inference_version.keys()
            max_inf_ver = next(reversed(inf_versions))
            assert isinstance(self._batch_index, int)
            assert self._config.allowed_staleness is not None
            min_inf_ver = self._batch_index - 1 - self._config.allowed_staleness
            assert next(iter(inf_versions)) >= min_inf_ver
            for ver in range(max_inf_ver, min_inf_ver - 1, -1):
                # remove batch_size of tasks from reserved_places
                cnt = 0
                while cnt < self._config.batch_size and len(reserved_places) > 0:
                    batch_index, task_num = reserved_places.pop()
                    if cnt + task_num > self._config.batch_size:
                        task_num_to_return = cnt + task_num - self._config.batch_size
                        reserved_places.append((batch_index, task_num_to_return))
                        break
                    cnt += task_num
                if ver in self._tasks_by_inference_version:
                    reserved_places.appendleft((
                        ver,
                        len(self._tasks_by_inference_version[ver]) * self._config.num_samples_per_task,
                    ))

        allowed_versions = {x[0] for x in reserved_places}
        reserved_places_num = sum(x[1] for x in reserved_places)

        if (
            reserved_places_num < self._batch_remainder
            or self._task_to_inference_version.get(task_id) in allowed_versions
        ):
            self._add_ready_sample_group(sample_group=sample_group)
            return True
        if not is_pending:
            heapq.heappush(
                self._pending_sample_groups,
                (
                    self._task_to_inference_version[task_id],
                    len(self._pending_sample_groups),
                    task_id,
                    sample_group,
                ),
            )
        return False
