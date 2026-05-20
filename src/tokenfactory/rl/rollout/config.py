from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class RolloutConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="tokenfactory_rollout_")

    job_id: str
    model_name: str
    max_concurrency: int = 32
    executor_type: Literal["thread", "process"] = "thread"

    num_batches: int | None = None
    batch_size: int
    num_samples_per_task: int
    allowed_staleness: int = 0
    drop_stale_trajectories: bool = False
    num_rollout_retries: int = 0
    raise_on_rollout_failure: bool = False

    status_polling_interval: float = 1.0
    job_init_timeout: int | None = 600
