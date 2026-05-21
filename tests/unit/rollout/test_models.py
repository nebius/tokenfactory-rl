from dataclasses import asdict

import pytest

from tokenfactory.rl.rollout.models import (
    BaseID,
    RolloutException,
    RolloutResult,
    Sample,
    SampleGroup,
    SampleID,
    Task,
    TaskID,
)


# ===========================================================================
# BaseID / TaskID / SampleID
# ===========================================================================


class TestBaseID:
    def test_generate_raises_when_prefix_is_none(self):
        with pytest.raises(NotImplementedError, match="Prefix must be set"):
            BaseID.generate()

    def test_base_id_is_str_subclass(self):
        tid = TaskID.generate()
        assert isinstance(tid, str)


class TestTaskID:
    def test_generate_has_correct_prefix(self):
        tid = TaskID.generate()
        assert tid.startswith("task-")

    def test_generate_is_unique(self):
        ids = {TaskID.generate() for _ in range(100)}
        assert len(ids) == 100

    def test_can_construct_from_string(self):
        tid = TaskID("task-custom")
        assert tid == "task-custom"

    def test_is_instance_of_base_id(self):
        assert issubclass(TaskID, BaseID)


class TestSampleID:
    def test_generate_has_correct_prefix(self):
        sid = SampleID.generate()
        assert sid.startswith("sample-")

    def test_generate_is_unique(self):
        ids = {SampleID.generate() for _ in range(100)}
        assert len(ids) == 100


# ===========================================================================
# Task
# ===========================================================================


class TestTask:
    def test_auto_generates_id(self):
        t = Task(spec="some_spec", starting_inference_version=0)
        assert t.id.startswith("task-")

    def test_custom_id(self):
        tid = TaskID("task-custom")
        t = Task(id=tid, spec="spec", starting_inference_version=3)
        assert t.id == "task-custom"

    def test_stores_spec_and_version(self):
        t = Task(spec={"key": "val"}, starting_inference_version=7)
        assert t.spec == {"key": "val"}
        assert t.starting_inference_version == 7

    def test_unique_ids_across_instances(self):
        tasks = [Task(spec="s", starting_inference_version=0) for _ in range(50)]
        ids = {t.id for t in tasks}
        assert len(ids) == 50


# ===========================================================================
# Sample
# ===========================================================================


class TestSample:
    def test_auto_generates_id(self):
        s = Sample(token_ids=[1], logprobs=[0.1], mask=[1], normalized_reward=0.5)
        assert s.id.startswith("sample-")

    def test_defaults(self):
        s = Sample(token_ids=[1], logprobs=[0.1], mask=[1], normalized_reward=0.5)
        assert s.metadata == {}
        assert s.debug_info == {}

    def test_metadata_and_debug_info(self):
        s = Sample(
            token_ids=[1],
            logprobs=[0.1],
            mask=[1],
            normalized_reward=0.5,
            metadata={"k": "v"},
            debug_info={"step": 42},
        )
        assert s.metadata == {"k": "v"}
        assert s.debug_info == {"step": 42}

    def test_asdict_roundtrip(self):
        s = Sample(
            token_ids=[1, 2],
            logprobs=[0.1, 0.2],
            mask=[0, 1],
            normalized_reward=1.0,
        )
        d = asdict(s)
        assert d["token_ids"] == [1, 2]
        assert d["logprobs"] == [0.1, 0.2]
        assert d["mask"] == [0, 1]
        assert d["normalized_reward"] == 1.0
        assert "id" in d

    def test_default_dicts_are_independent(self):
        s1 = Sample(token_ids=[], logprobs=[], mask=[], normalized_reward=0.0)
        s2 = Sample(token_ids=[], logprobs=[], mask=[], normalized_reward=0.0)
        s1.metadata["x"] = "y"
        assert "x" not in s2.metadata


# ===========================================================================
# SampleGroup
# ===========================================================================


class TestSampleGroup:
    def test_defaults(self):
        sg = SampleGroup()
        assert sg.is_rejected is False
        assert sg.samples == []

    def test_rejected(self):
        sg = SampleGroup(is_rejected=True)
        assert sg.is_rejected is True

    def test_with_samples(self):
        s = Sample(token_ids=[1], logprobs=[0.1], mask=[1], normalized_reward=0.0)
        sg = SampleGroup(samples=[s])
        assert len(sg.samples) == 1


# ===========================================================================
# RolloutResult / RolloutException
# ===========================================================================


class TestRolloutResult:
    def test_stores_task_and_sample_group(self):
        task = Task(spec="s", starting_inference_version=0)
        sg = SampleGroup(samples=[])
        r = RolloutResult(task=task, sample_group=sg)
        assert r.task is task
        assert r.sample_group is sg


class TestRolloutException:
    def test_stores_exception_and_traceback(self):
        err = ValueError("boom")
        r = RolloutException(exception=err, traceback="tb text")
        assert r.exception is err
        assert r.traceback == "tb text"
