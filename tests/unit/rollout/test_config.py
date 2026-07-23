from typing import Any, cast

import pytest
from pydantic import ValidationError

from tokenfactory.rl.rollout.config import ExecutorType, RolloutConfig


class TestRolloutConfig:
    def test_required_fields(self):
        cfg = RolloutConfig(
            job_id="j1",
            model_name="m1",
            batch_size=64,
            num_samples_per_task=4,
        )
        assert cfg.job_id == "j1"
        assert cfg.model_name == "m1"
        assert cfg.batch_size == 64
        assert cfg.num_samples_per_task == 4

    def test_defaults(self):
        cfg = RolloutConfig(
            job_id="j1",
            model_name="m1",
            batch_size=64,
            num_samples_per_task=4,
        )
        assert cfg.max_concurrency == 32
        assert cfg.executor_type == ExecutorType.THREAD
        assert cfg.executor_type == "thread"
        assert cfg.num_batches is None
        assert cfg.allowed_staleness == 0
        assert cfg.drop_stale_trajectories is False
        assert cfg.num_rollout_retries == 0
        assert cfg.raise_on_rollout_failure is False

    def test_override_defaults(self):
        cfg = RolloutConfig(
            job_id="j1",
            model_name="m1",
            batch_size=64,
            num_samples_per_task=4,
            max_concurrency=16,
            executor_type=ExecutorType.PROCESS,
            num_batches=100,
            allowed_staleness=2,
            drop_stale_trajectories=True,
            num_rollout_retries=3,
            raise_on_rollout_failure=True,
        )
        assert cfg.max_concurrency == 16
        assert cfg.executor_type == ExecutorType.PROCESS
        assert cfg.executor_type == "process"
        assert cfg.num_batches == 100
        assert cfg.allowed_staleness == 2
        assert cfg.drop_stale_trajectories is True
        assert cfg.num_rollout_retries == 3
        assert cfg.raise_on_rollout_failure is True

    def test_executor_type_rejects_invalid(self):
        with pytest.raises(ValidationError):
            RolloutConfig.model_validate(
                {
                    "job_id": "j1",
                    "model_name": "m1",
                    "batch_size": 64,
                    "num_samples_per_task": 4,
                    "executor_type": "invalid",
                }
            )

    def test_executor_type_accepts_process_string(self):
        cfg = RolloutConfig.model_validate(
            {
                "job_id": "j1",
                "model_name": "m1",
                "batch_size": 64,
                "num_samples_per_task": 4,
                "executor_type": "process",
            }
        )

        assert cfg.executor_type == ExecutorType.PROCESS
        assert cfg.executor_type == "process"

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_JOB_ID", "env-job")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_MODEL_NAME", "env-model")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_BATCH_SIZE", "128")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_NUM_SAMPLES_PER_TASK", "8")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_MAX_CONCURRENCY", "64")

        cfg = cast(RolloutConfig, cast(Any, RolloutConfig)())
        assert cfg.job_id == "env-job"
        assert cfg.model_name == "env-model"
        assert cfg.batch_size == 128
        assert cfg.num_samples_per_task == 8
        assert cfg.max_concurrency == 64

    def test_env_executor_type(self, monkeypatch):
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_JOB_ID", "env-job")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_MODEL_NAME", "env-model")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_BATCH_SIZE", "128")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_NUM_SAMPLES_PER_TASK", "8")
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_EXECUTOR_TYPE", "process")

        cfg = cast(RolloutConfig, cast(Any, RolloutConfig)())

        assert cfg.executor_type == ExecutorType.PROCESS
        assert cfg.executor_type == "process"

    def test_explicit_values_override_env(self, monkeypatch):
        monkeypatch.setenv("TOKENFACTORY_ROLLOUT_MAX_CONCURRENCY", "999")

        cfg = RolloutConfig(
            job_id="j1",
            model_name="m1",
            batch_size=64,
            num_samples_per_task=4,
            max_concurrency=16,
        )
        assert cfg.max_concurrency == 16
