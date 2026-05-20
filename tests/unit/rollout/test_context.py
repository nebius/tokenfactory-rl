from typing import Literal
from unittest.mock import MagicMock

import pytest
from tokenfactory.rl.rollout.config import RolloutConfig
from tokenfactory.rl.rollout.context import RolloutContext


@pytest.fixture(autouse=True)
def _fake_openai_key(monkeypatch):
    """OpenAI client requires an API key even when pointing at a custom endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _make_config(
    *,
    job_id: str = "ctx-job",
    model_name: str = "test-model",
    max_concurrency: int = 32,
    executor_type: Literal["thread", "process"] = "thread",
    num_batches: int | None = None,
    batch_size: int = 4,
    num_samples_per_task: int = 1,
    allowed_staleness: int = 0,
    drop_stale_trajectories: bool = False,
    num_rollout_retries: int = 0,
    raise_on_rollout_failure: bool = False,
) -> RolloutConfig:
    return RolloutConfig(
        job_id=job_id,
        model_name=model_name,
        max_concurrency=max_concurrency,
        executor_type=executor_type,
        num_batches=num_batches,
        batch_size=batch_size,
        num_samples_per_task=num_samples_per_task,
        allowed_staleness=allowed_staleness,
        drop_stale_trajectories=drop_stale_trajectories,
        num_rollout_retries=num_rollout_retries,
        raise_on_rollout_failure=raise_on_rollout_failure,
    )


def _make_context(config=None, api_client=None) -> tuple[RolloutContext, MagicMock]:
    client = api_client or MagicMock()
    cfg = config or _make_config()
    return RolloutContext(config=cfg, api_client=client), client


class TestRolloutContext:
    def test_config_property(self):
        cfg = _make_config()
        ctx, _ = _make_context(config=cfg)
        assert ctx.config is cfg

    def test_openai_endpoint_calls_api(self):
        ctx, client = _make_context()
        client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.return_value = "http://inference:8000/v1"

        endpoint = ctx.openai_endpoint

        assert endpoint == "http://inference:8000/v1"
        client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.assert_called_once_with(job_id="ctx-job")

    def test_openai_endpoint_is_cached(self):
        ctx, client = _make_context()
        client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.return_value = "http://inference:8000/v1"

        _ = ctx.openai_endpoint
        _ = ctx.openai_endpoint

        assert client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.call_count == 1

    def test_openai_client_uses_endpoint(self):
        ctx, client = _make_context()
        client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.return_value = "http://inference:8000/v1"

        oai = ctx.openai_client

        from openai import OpenAI

        assert isinstance(oai, OpenAI)
        assert str(oai.base_url) == "http://inference:8000/v1/"

    def test_openai_client_is_cached(self):
        ctx, client = _make_context()
        client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.return_value = "http://inference:8000/v1"

        c1 = ctx.openai_client
        c2 = ctx.openai_client
        assert c1 is c2

    def test_async_openai_client_uses_endpoint(self):
        ctx, client = _make_context()
        client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.return_value = "http://inference:8000/v1"

        oai = ctx.async_openai_client()

        from openai import AsyncOpenAI

        assert isinstance(oai, AsyncOpenAI)
        assert str(oai.base_url) == "http://inference:8000/v1/"

    def test_async_openai_client_is_not_cached(self):
        ctx, client = _make_context()
        client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint.return_value = "http://inference:8000/v1"

        c1 = ctx.async_openai_client()
        c2 = ctx.async_openai_client()
        assert c1 is not c2
