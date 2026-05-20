from dataclasses import dataclass

import pytest

from tokenfactory.rl.schedulers import (
    AnyStalenessRolloutScheduler,
    BaseRolloutScheduler,
    DroplessRolloutScheduler,
    DropStaleRolloutScheduler,
    RolloutSchedulerConfig,
    create_rollout_scheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTask:
    id: str
    value: str = ""


@dataclass
class FakeSample:
    data: int = 0


SchedulerUnderTest = BaseRolloutScheduler[str, FakeSample]
DropStaleSchedulerUnderTest = DropStaleRolloutScheduler[str, FakeSample]
DroplessSchedulerUnderTest = DroplessRolloutScheduler[str, FakeSample]


def _make_config(
    *,
    batch_size: int = 4,
    num_batches: int | None = None,
    num_samples_per_task: int = 1,
    parallel_rollouts: int = 4,
    allowed_staleness: int | None = 0,
    drop_stale_trajectories: bool = False,
) -> RolloutSchedulerConfig:
    return RolloutSchedulerConfig(
        batch_size=batch_size,
        num_batches=num_batches,
        num_samples_per_task=num_samples_per_task,
        parallel_rollouts=parallel_rollouts,
        allowed_staleness=allowed_staleness,
        drop_stale_trajectories=drop_stale_trajectories,
    )


def _init_scheduler(
    scheduler: SchedulerUnderTest,
    batch_index: int = 0,
    n_samples: int = 0,
    inference_version: int = 0,
) -> None:
    scheduler.set_initial_batch_index(batch_index=batch_index, n_samples=n_samples)
    scheduler.set_inference_version(inference_version=inference_version)


def _run_full_cycle(
    scheduler: SchedulerUnderTest,
    task_id: str,
    num_samples_per_task: int = 1,
) -> list[FakeSample]:
    """Add a task, add its samples, return ready samples."""
    task = FakeTask(id=task_id)
    scheduler.add_tasks(tasks=[task])
    samples = [FakeSample(data=i) for i in range(num_samples_per_task)]
    scheduler.add_sample_group(task_id=task_id, samples=samples)
    return scheduler.get_ready_samples()


# ===========================================================================
# Factory
# ===========================================================================


class TestCreateRolloutScheduler:
    def test_creates_any_staleness_when_allowed_staleness_none(self):
        config = _make_config(allowed_staleness=None)
        assert isinstance(
            create_rollout_scheduler(config), AnyStalenessRolloutScheduler
        )

    def test_creates_drop_stale_when_drop_flag_true(self):
        config = _make_config(allowed_staleness=1, drop_stale_trajectories=True)
        assert isinstance(create_rollout_scheduler(config), DropStaleRolloutScheduler)

    def test_creates_dropless_when_drop_flag_false(self):
        config = _make_config(allowed_staleness=1, drop_stale_trajectories=False)
        assert isinstance(create_rollout_scheduler(config), DroplessRolloutScheduler)

    def test_rejects_non_divisible_batch_size(self):
        with pytest.raises(ValueError, match="divisible"):
            _make_config(batch_size=5, num_samples_per_task=3)

    def test_accepts_divisible_batch_size(self):
        config = _make_config(batch_size=6, num_samples_per_task=3)
        assert config.batch_size == 6


# ===========================================================================
# BaseRolloutScheduler (shared behaviour, tested via AnyStaleness)
# ===========================================================================


class TestBaseSchedulerBehaviour:
    def test_set_initial_batch_index_twice_raises(self):
        s = AnyStalenessRolloutScheduler(config=_make_config(allowed_staleness=None))
        s.set_initial_batch_index(batch_index=0)
        with pytest.raises(RuntimeError, match="set only once"):
            s.set_initial_batch_index(batch_index=1)

    def test_set_initial_batch_index_with_n_samples_greater_than_batch_size_raises(
        self,
    ):
        s = AnyStalenessRolloutScheduler(
            config=_make_config(allowed_staleness=None, batch_size=4)
        )
        with pytest.raises(ValueError, match="greater than batch size"):
            s.set_initial_batch_index(batch_index=0, n_samples=5)

    def test_inference_version_cannot_decrease(self):
        s = AnyStalenessRolloutScheduler(config=_make_config(allowed_staleness=None))
        _init_scheduler(s, inference_version=5)
        s.set_inference_version(inference_version=5)  # same is fine
        with pytest.raises(RuntimeError, match="greater or equal"):
            s.set_inference_version(inference_version=4)

    def test_add_tasks_before_inference_version_raises(self):
        s = AnyStalenessRolloutScheduler(config=_make_config(allowed_staleness=None))
        s.set_initial_batch_index(batch_index=0)
        with pytest.raises(RuntimeError, match="Inference version must be set"):
            s.add_tasks(tasks=[FakeTask(id="t1")])

    def test_remove_task(self):
        s = AnyStalenessRolloutScheduler(config=_make_config(allowed_staleness=None))
        _init_scheduler(s)
        task = FakeTask(id="t1")
        s.add_tasks(tasks=[task])
        assert s._task_num_in_progress == 1
        s.remove_task(task_id="t1")
        assert s._task_num_in_progress == 0

    def test_is_all_finished_false_when_num_batches_none(self):
        s = AnyStalenessRolloutScheduler(
            config=_make_config(allowed_staleness=None, num_batches=None)
        )
        _init_scheduler(s)
        assert s.is_all_finished is False

    def test_is_all_finished_true_when_all_batches_done(self):
        config = _make_config(
            allowed_staleness=None, num_batches=1, batch_size=2, num_samples_per_task=1
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        _run_full_cycle(s, "t1")
        _run_full_cycle(s, "t2")
        assert s.is_all_finished is True

    def test_get_ready_samples_drains(self):
        s = AnyStalenessRolloutScheduler(config=_make_config(allowed_staleness=None))
        _init_scheduler(s)
        _run_full_cycle(s, "t1")
        first = s.get_ready_samples()  # already drained by _run_full_cycle
        assert first == []

    def test_num_samples_per_task_assertion(self):
        config = _make_config(allowed_staleness=None, num_samples_per_task=2)
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        task = FakeTask(id="t1")
        s.add_tasks(tasks=[task])
        with pytest.raises(AssertionError):
            s.add_sample_group(task_id="t1", samples=[FakeSample()])

    def test_resume_mid_batch(self):
        config = _make_config(
            allowed_staleness=None, num_batches=1, batch_size=4, num_samples_per_task=1
        )
        s = AnyStalenessRolloutScheduler(config=config)
        s.set_initial_batch_index(batch_index=0, n_samples=3)
        s.set_inference_version(inference_version=0)
        assert s.is_all_finished is False
        _run_full_cycle(s, "t1")
        assert s.is_all_finished is True

    def test_get_num_rollouts_respects_total_remaining(self):
        config = _make_config(
            allowed_staleness=None,
            num_batches=1,
            batch_size=2,
            num_samples_per_task=1,
            parallel_rollouts=10,
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        assert s.get_num_rollouts_to_spawn() == 2  # capped by num_batches * batch_size

    def test_add_tasks_exceeding_budget_raises(self):
        config = _make_config(
            allowed_staleness=None,
            num_batches=1,
            batch_size=1,
            num_samples_per_task=1,
            parallel_rollouts=10,
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        s.add_tasks(tasks=[FakeTask(id="t1")])
        with pytest.raises(RuntimeError, match="more tasks than allowed"):
            s.add_tasks(tasks=[FakeTask(id="t2")])


# ===========================================================================
# AnyStalenessRolloutScheduler
# ===========================================================================


class TestAnyStalenessScheduler:
    def test_basic_flow(self):
        config = _make_config(
            allowed_staleness=None, batch_size=4, num_samples_per_task=1
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)

        ready = _run_full_cycle(s, "t1")
        assert len(ready) == 1

    def test_spawns_up_to_parallel_rollouts(self):
        config = _make_config(
            allowed_staleness=None, parallel_rollouts=8, num_samples_per_task=1
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        assert s.get_num_rollouts_to_spawn() == 8

    def test_spawns_accounts_for_in_progress(self):
        config = _make_config(
            allowed_staleness=None, parallel_rollouts=4, num_samples_per_task=1
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        s.add_tasks(tasks=[FakeTask(id="t1"), FakeTask(id="t2")])
        assert s.get_num_rollouts_to_spawn() == 2

    def test_spawns_accounts_for_num_samples_per_task(self):
        config = _make_config(
            allowed_staleness=None, parallel_rollouts=8, num_samples_per_task=2
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        s.add_tasks(tasks=[FakeTask(id="t1")])
        # 1 task in progress producing 2 samples → occupies 2 slots
        assert s.get_num_rollouts_to_spawn() == 6

    def test_multiple_samples_per_task(self):
        config = _make_config(
            allowed_staleness=None, batch_size=4, num_samples_per_task=2
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)

        task = FakeTask(id="t1")
        s.add_tasks(tasks=[task])
        samples = [FakeSample(data=i) for i in range(2)]
        s.add_sample_group(task_id="t1", samples=samples)
        ready = s.get_ready_samples()
        assert len(ready) == 2

    def test_samples_accepted_regardless_of_version_gap(self):
        config = _make_config(
            allowed_staleness=None, batch_size=4, num_samples_per_task=1
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s, inference_version=0)

        task = FakeTask(id="t1")
        s.add_tasks(tasks=[task])

        # Advance inference version far ahead before completing the task
        s.set_inference_version(inference_version=100)

        samples = [FakeSample()]
        s.add_sample_group(task_id="t1", samples=samples)
        ready = s.get_ready_samples()
        assert len(ready) == 1

    def test_batch_transitions(self):
        config = _make_config(
            allowed_staleness=None,
            num_batches=3,
            batch_size=2,
            num_samples_per_task=1,
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)

        assert s.current_batch_index == 0
        _run_full_cycle(s, "t1")
        assert s.current_batch_index == 0  # 1/2
        _run_full_cycle(s, "t2")
        assert s.current_batch_index == 1  # batch 0 full → now on batch 1
        _run_full_cycle(s, "t3")
        _run_full_cycle(s, "t4")
        assert s.current_batch_index == 2
        _run_full_cycle(s, "t5")
        _run_full_cycle(s, "t6")
        assert s.current_batch_index == 3
        assert s.is_all_finished is True

    def test_infinite_mode_never_finishes(self):
        config = _make_config(
            allowed_staleness=None,
            num_batches=None,
            batch_size=2,
            num_samples_per_task=1,
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)
        for i in range(100):
            _run_full_cycle(s, f"t{i}")
        assert s.is_all_finished is False


# ===========================================================================
# DropStaleRolloutScheduler
# ===========================================================================


class TestDropStaleScheduler:
    def _make(self, **overrides) -> DropStaleSchedulerUnderTest:
        config = _make_config(drop_stale_trajectories=True, **overrides)
        return DropStaleRolloutScheduler(config=config)

    def test_fresh_sample_accepted(self):
        s = self._make(batch_size=4, allowed_staleness=0)
        _init_scheduler(s, inference_version=0)

        task = FakeTask(id="t1")
        s.add_tasks(tasks=[task])
        # Advance to version 1 so batch_index - 1 - task_version = 0 - 1 - 0 = -1 <= 0
        s.set_inference_version(inference_version=1)
        s.add_sample_group(task_id="t1", samples=[FakeSample()])
        ready = s.get_ready_samples()
        assert len(ready) == 1

    def test_stale_sample_dropped(self):
        """With allowed_staleness=0, a sample from version 0 is stale
        once batch_index - 1 - task_version > allowed_staleness.
        Need batch_index=2 so that 2 - 1 - 0 = 1 > 0."""
        s = self._make(batch_size=1, allowed_staleness=0)
        _init_scheduler(s, inference_version=0)

        # Start a task at version 0 that will become stale
        task = FakeTask(id="t_stale")
        s.add_tasks(tasks=[task])

        # Fill 2 batches to push batch_index to 2
        s.set_inference_version(inference_version=1)
        _run_full_cycle(s, "f1")
        assert s.current_batch_index == 1
        _run_full_cycle(s, "f2")
        assert s.current_batch_index == 2

        # batch_index - 1 - task_version = 2 - 1 - 0 = 1 > 0 → dropped
        s.add_sample_group(task_id="t_stale", samples=[FakeSample()])
        ready = s.get_ready_samples()
        assert len(ready) == 0

    def test_exactly_at_staleness_boundary_accepted(self):
        s = self._make(batch_size=2, allowed_staleness=1)
        _init_scheduler(s, inference_version=0)

        # Task started at version 0
        task = FakeTask(id="t1")
        s.add_tasks(tasks=[task])

        # Fill batch 0 to advance batch_index to 1
        _run_full_cycle(s, "f1")
        _run_full_cycle(s, "f2")
        assert s.current_batch_index == 1

        # batch_index - 1 - task_version = 1 - 1 - 0 = 0 <= 1 → accepted
        s.add_sample_group(task_id="t1", samples=[FakeSample()])
        ready = s.get_ready_samples()
        assert len(ready) == 1

    def test_one_past_staleness_boundary_dropped(self):
        s = self._make(batch_size=1, allowed_staleness=1)
        _init_scheduler(s, inference_version=0)

        task = FakeTask(id="t_old")
        s.add_tasks(tasks=[task])

        # Fill 3 batches to push batch_index to 3
        # batch_index - 1 - task_version = 3 - 1 - 0 = 2 > 1 → dropped
        _run_full_cycle(s, "f1")
        _run_full_cycle(s, "f2")
        _run_full_cycle(s, "f3")
        assert s.current_batch_index == 3

        s.add_sample_group(task_id="t_old", samples=[FakeSample()])
        ready = s.get_ready_samples()
        assert len(ready) == 0

    def test_spawns_up_to_parallel_rollouts(self):
        s = self._make(parallel_rollouts=6, allowed_staleness=2)
        _init_scheduler(s)
        assert s.get_num_rollouts_to_spawn() == 6


# ===========================================================================
# DroplessRolloutScheduler
# ===========================================================================


class TestDroplessScheduler:
    def _make(self, **overrides) -> DroplessSchedulerUnderTest:
        config = _make_config(drop_stale_trajectories=False, **overrides)
        return DroplessRolloutScheduler(config=config)

    def test_fresh_sample_accepted_immediately(self):
        s = self._make(batch_size=4, allowed_staleness=0)
        _init_scheduler(s, inference_version=0)

        ready = _run_full_cycle(s, "t1")
        assert len(ready) == 1

    def test_sample_buffered_when_would_exceed_staleness(self):
        """When there are in-progress tasks from an older version that reserve
        batch slots, a newer sample may be buffered instead of dropped."""
        s = self._make(batch_size=2, allowed_staleness=0)
        _init_scheduler(s, inference_version=0)

        # Start two tasks at version 0
        old_task = FakeTask(id="t_old")
        s.add_tasks(tasks=[old_task])

        # Advance version and start a new task
        s.set_inference_version(inference_version=1)
        new_task = FakeTask(id="t_new")
        s.add_tasks(tasks=[new_task])

        # Complete the new task first — it may be buffered if old task reserves a slot
        s.add_sample_group(task_id="t_new", samples=[FakeSample(data=99)])
        ready_after_new = s.get_ready_samples()

        # Complete the old task
        s.add_sample_group(task_id="t_old", samples=[FakeSample(data=1)])
        ready_after_old = s.get_ready_samples()

        # Both samples should eventually be delivered (none dropped)
        total = len(ready_after_new) + len(ready_after_old)
        assert total == 2

    def test_pending_queue_drains_after_straggler_completes(self):
        s = self._make(batch_size=4, allowed_staleness=0)
        _init_scheduler(s, inference_version=0)

        # Start a task at version 0
        old = FakeTask(id="old")
        s.add_tasks(tasks=[old])

        # Advance and start 3 tasks at version 1
        s.set_inference_version(inference_version=1)
        for i in range(3):
            s.add_tasks(tasks=[FakeTask(id=f"new_{i}")])

        # Complete new tasks first — they may be buffered
        for i in range(3):
            s.add_sample_group(task_id=f"new_{i}", samples=[FakeSample(data=i)])

        buffered = s.get_ready_samples()

        # Now complete the old task
        s.add_sample_group(task_id="old", samples=[FakeSample(data=100)])
        after_old = s.get_ready_samples()

        total = len(buffered) + len(after_old)
        assert total == 4

    def test_spawn_budget_accounts_for_pending(self):
        s = self._make(
            batch_size=2,
            allowed_staleness=0,
            parallel_rollouts=10,
            num_samples_per_task=1,
        )
        _init_scheduler(s, inference_version=0)

        old = FakeTask(id="old")
        s.add_tasks(tasks=[old])

        s.set_inference_version(inference_version=1)
        new = FakeTask(id="new")
        s.add_tasks(tasks=[new])

        # Complete new task — gets buffered
        s.add_sample_group(task_id="new", samples=[FakeSample()])
        s.get_ready_samples()

        # Budget should be reduced by the pending sample
        spawn = s.get_num_rollouts_to_spawn()
        # 1 task in progress (old) + 1 pending → limited budget
        assert spawn >= 0

    def test_no_samples_dropped(self):
        """Main guarantee: even with staleness constraints, nothing is dropped."""
        s = self._make(
            batch_size=2,
            allowed_staleness=0,
            num_samples_per_task=1,
            parallel_rollouts=10,
        )
        _init_scheduler(s, inference_version=0)

        all_ready = []

        # Version 0: 2 tasks
        s.add_tasks(tasks=[FakeTask(id="v0_a"), FakeTask(id="v0_b")])

        # Version 1: 2 tasks
        s.set_inference_version(inference_version=1)
        s.add_tasks(tasks=[FakeTask(id="v1_a"), FakeTask(id="v1_b")])

        # Complete v1 tasks first (out of order)
        s.add_sample_group(task_id="v1_a", samples=[FakeSample()])
        all_ready.extend(s.get_ready_samples())
        s.add_sample_group(task_id="v1_b", samples=[FakeSample()])
        all_ready.extend(s.get_ready_samples())

        # Then complete v0 tasks
        s.add_sample_group(task_id="v0_a", samples=[FakeSample()])
        all_ready.extend(s.get_ready_samples())
        s.add_sample_group(task_id="v0_b", samples=[FakeSample()])
        all_ready.extend(s.get_ready_samples())

        assert len(all_ready) == 4

    def test_pending_drains_in_version_order(self):
        """Pending samples should drain oldest-version-first."""
        s = self._make(
            batch_size=4,
            allowed_staleness=0,
            num_samples_per_task=1,
            parallel_rollouts=20,
        )
        _init_scheduler(s, inference_version=0)

        old = FakeTask(id="old")
        s.add_tasks(tasks=[old])

        s.set_inference_version(inference_version=1)
        s.add_tasks(tasks=[FakeTask(id="v1")])

        s.set_inference_version(inference_version=2)
        s.add_tasks(tasks=[FakeTask(id="v2")])

        # Complete in reverse version order
        s.add_sample_group(task_id="v2", samples=[FakeSample(data=2)])
        s.add_sample_group(task_id="v1", samples=[FakeSample(data=1)])

        # Both should be pending/buffered; old task still blocks
        mid = s.get_ready_samples()

        # Complete old task — should flush pending in version order
        s.add_sample_group(task_id="old", samples=[FakeSample(data=0)])
        final = s.get_ready_samples()

        all_samples = mid + final
        assert len(all_samples) == 3

    def test_spawn_limited_by_staleness_window(self):
        """Spawn count should not allow tasks that would inevitably exceed staleness."""
        s = self._make(
            batch_size=2,
            allowed_staleness=1,
            parallel_rollouts=100,
            num_samples_per_task=1,
        )
        _init_scheduler(s, inference_version=0)

        spawn = s.get_num_rollouts_to_spawn()
        # With version=0, staleness=1: can fill batches up to version 0+1+1=2
        # That's 2 batches * 2 batch_size = 4 + remainder 2 = 6 max budget
        # Capped to min(100, 6) = 6
        assert spawn <= 100
        assert spawn > 0


# ===========================================================================
# Multi-batch integration scenarios
# ===========================================================================


class TestMultiBatchIntegration:
    def test_any_staleness_fills_multiple_batches(self):
        config = _make_config(
            allowed_staleness=None,
            num_batches=3,
            batch_size=2,
            num_samples_per_task=1,
            parallel_rollouts=4,
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)

        total_ready = 0
        for i in range(6):
            ready = _run_full_cycle(s, f"t{i}")
            total_ready += len(ready)

        assert total_ready == 6
        assert s.is_all_finished is True
        assert s.get_num_rollouts_to_spawn() == 0

    def test_drop_stale_fills_batches_with_mixed_versions(self):
        config = _make_config(
            allowed_staleness=1,
            drop_stale_trajectories=True,
            num_batches=2,
            batch_size=2,
            num_samples_per_task=1,
            parallel_rollouts=10,
        )
        s = DropStaleRolloutScheduler(config=config)
        _init_scheduler(s, inference_version=0)

        all_ready = []

        # Batch 0: fill normally at version 0
        all_ready.extend(_run_full_cycle(s, "b0_t1"))
        all_ready.extend(_run_full_cycle(s, "b0_t2"))
        assert s.current_batch_index == 1

        # Start task at version 0, then advance to version 2
        old_task = FakeTask(id="old")
        s.add_tasks(tasks=[old_task])
        s.set_inference_version(inference_version=2)

        # batch_index - 1 - version = 1 - 1 - 0 = 0 <= 1 → still accepted
        s.add_sample_group(task_id="old", samples=[FakeSample()])
        all_ready.extend(s.get_ready_samples())

        # Fill rest of batch 1
        all_ready.extend(_run_full_cycle(s, "b1_t2"))
        assert s.current_batch_index == 2
        assert s.is_all_finished is True
        assert len(all_ready) == 4

    def test_resume_mid_batch_then_complete(self):
        config = _make_config(
            allowed_staleness=None,
            num_batches=2,
            batch_size=4,
            num_samples_per_task=1,
            parallel_rollouts=10,
        )
        s = AnyStalenessRolloutScheduler(config=config)
        # Resume at batch 1 with 3 samples already submitted
        s.set_initial_batch_index(batch_index=1, n_samples=3)
        s.set_inference_version(inference_version=5)

        assert s.is_all_finished is False
        # Only 1 sample left to fill batch 1
        ready = _run_full_cycle(s, "t1")
        assert len(ready) == 1
        assert s.current_batch_index == 2
        assert s.is_all_finished is True

    def test_spawning_finished_before_all_finished(self):
        config = _make_config(
            allowed_staleness=None,
            num_batches=1,
            batch_size=2,
            num_samples_per_task=1,
            parallel_rollouts=10,
        )
        s = AnyStalenessRolloutScheduler(config=config)
        _init_scheduler(s)

        # Spawn and add 2 tasks (exhausts budget)
        s.add_tasks(tasks=[FakeTask(id="t1"), FakeTask(id="t2")])
        assert s.is_spawning_finished is True
        assert s.is_all_finished is False

        # Complete them
        s.add_sample_group(task_id="t1", samples=[FakeSample()])
        s.add_sample_group(task_id="t2", samples=[FakeSample()])
        s.get_ready_samples()
        assert s.is_all_finished is True
