import json

import pytest
from pytest_httpx import HTTPXMock
from tokenfactory.rl.api_client import MissingAPIKeyError, NotFoundError, TokenFactory
from tokenfactory.rl.api_client.models import Batch, BatchStatus, ChatCompletion, Completion, JobStatus, Sample


BASE_URL = "http://test-server"


@pytest.fixture
def client() -> TokenFactory:
    return TokenFactory(BASE_URL, "123", max_retries=1)


def test_constructor_api_key_is_used_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TOKENFACTORY_API_KEY", raising=False)

    client = TokenFactory(BASE_URL, api_key="constructor-key")

    assert client.api_key == "constructor-key"


def test_constructor_api_key_overrides_env(monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock):
    monkeypatch.setenv("TOKENFACTORY_API_KEY", "env-key")
    client = TokenFactory(BASE_URL, api_key="constructor-key")
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/status",
        json={
            "object": "fine_tuning.job.status",
            "inference_version": 5,
            "total_batches": 20,
            "filled_batches": 10,
        },
    )

    client.v1alpha1.fine_tuning.jobs.get_runtime_status(job_id="job-123")

    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers["Authorization"] == "Bearer constructor-key"


def test_missing_api_key_raises_when_constructor_and_env_are_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TOKENFACTORY_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        TokenFactory(BASE_URL)


def test_resource_hierarchy(client: TokenFactory):
    jobs = client.v1alpha1.fine_tuning.jobs
    assert jobs.endpoint(job_id="job-123") == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123"
    assert jobs.batches.endpoint(job_id="job-123") == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/batches"


def test_get_runtime_status(client: TokenFactory, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/status",
        json={
            "object": "fine_tuning.job.status",
            "inference_version": 5,
            "total_batches": 20,
            "filled_batches": 10,
        },
    )

    status = client.v1alpha1.fine_tuning.jobs.get_runtime_status(job_id="job-123")

    assert isinstance(status, JobStatus)
    assert status.inference_version == 5
    assert status.total_batches == 20
    assert status.filled_batches == 10


def test_batches_create(client: TokenFactory, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/batches",
        json={
            "object": "fine_tuning.batch",
            "index": 0,
            "status": "filling",
            "sample_count": 0,
        },
    )

    batch = client.v1alpha1.fine_tuning.jobs.batches.create(job_id="job-123", index=0)

    assert isinstance(batch, Batch)
    assert batch.index == 0
    assert batch.status == BatchStatus.filling

    request = httpx_mock.get_request()
    assert request is not None
    assert request.content == b'{"index":0}'


def test_batches_get(client: TokenFactory, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/batches/2",
        json={
            "object": "fine_tuning.batch",
            "index": 2,
            "status": "filled",
            "sample_count": 64,
        },
    )

    batch = client.v1alpha1.fine_tuning.jobs.batches.get(job_id="job-123", batch_index=2)

    assert isinstance(batch, Batch)
    assert batch.index == 2
    assert batch.status == BatchStatus.filled
    assert batch.sample_count == 64


def test_batches_submit_samples(client: TokenFactory, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/batches/0/samples",
        json={
            "object": "fine_tuning.batch",
            "index": 0,
            "status": "filling",
            "sample_count": 1,
        },
    )

    sample = Sample(
        object="fine_tuning.rl_sample",
        id="ftsample-1",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        mask=[1, 1, 1],
        normalized_reward=1.0,
        metadata={"key": "value"},
        debug_info={"info": "data"},
    )

    batch = client.v1alpha1.fine_tuning.jobs.batches.submit_samples(job_id="job-123", batch_index=0, samples=[sample])

    assert isinstance(batch, Batch)
    assert batch.sample_count == 1

    request = httpx_mock.get_request()
    assert request is not None
    payload = json.loads(request.content)
    assert len(payload["samples"]) == 1
    assert payload["samples"][0]["id"] == "ftsample-1"


def test_retry_on_server_error(httpx_mock: HTTPXMock):
    client = TokenFactory(BASE_URL, "123", max_retries=3)

    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/status",
        status_code=500,
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/status",
        status_code=500,
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/status",
        json={
            "object": "fine_tuning.job.status",
            "inference_version": 1,
            "total_batches": 10,
            "filled_batches": 5,
        },
    )

    status = client.v1alpha1.fine_tuning.jobs.get_runtime_status(job_id="job-123")
    assert status.filled_batches == 5
    assert len(httpx_mock.get_requests()) == 3


def test_no_retry_on_client_error(httpx_mock: HTTPXMock):
    client = TokenFactory(BASE_URL, "123", max_retries=3)

    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/bad-id/status",
        status_code=404,
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.v1alpha1.fine_tuning.jobs.get_runtime_status(job_id="bad-id")

    assert exc_info.value.response is not None
    assert exc_info.value.response.status_code == 404
    assert len(httpx_mock.get_requests()) == 1


def test_context_manager():
    client = TokenFactory(BASE_URL, "123")
    with client as c:
        assert c is client


def test_inference_resource_hierarchy(client: TokenFactory):
    inference = client.v1alpha1.fine_tuning.jobs.inference
    assert inference.endpoint(job_id="job-123") == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference"
    assert (
        inference.openai.endpoint(job_id="job-123") == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai"
    )
    assert (
        inference.openai.v1.endpoint("job-123") == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai/v1"
    )
    assert (
        inference.openai.v1.chat.endpoint(job_id="job-123")
        == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai/v1/chat"
    )
    assert (
        inference.openai.v1.completions.endpoint(job_id="job-123")
        == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai/v1/completions"
    )
    assert (
        inference.openai.v1.chat.completions.endpoint(job_id="job-123")
        == f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai/v1/chat/completions"
    )


def test_chat_completions_create(client: TokenFactory, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai/v1/chat/completions",
        json={
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    )

    result = client.v1alpha1.fine_tuning.jobs.inference.openai.v1.chat.completions.create(
        job_id="job-123",
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert isinstance(result, ChatCompletion)
    assert result.id == "chatcmpl-abc"
    assert result.choices[0].message.content == "Hello!"

    request = httpx_mock.get_request()
    assert request is not None
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "name": None, "content": "Hi"}]


def test_completions_create(client: TokenFactory, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai/v1/completions",
        json={
            "id": "cmpl-xyz",
            "object": "text_completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "text": "world",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        },
    )

    result = client.v1alpha1.fine_tuning.jobs.inference.openai.v1.completions.create(
        job_id="job-123",
        model="test-model",
        prompt="Hello ",
    )

    assert isinstance(result, Completion)
    assert result.id == "cmpl-xyz"
    assert result.choices[0].text == "world"

    request = httpx_mock.get_request()
    assert request is not None
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["prompt"] == "Hello "


def test_models_get(client: TokenFactory, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1alpha1/fine_tuning/jobs/job-123/inference/openai/v1/models",
        json={
            "object": "list",
            "data": [{"id": "model-1", "object": "model"}],
        },
    )

    result = client.v1alpha1.fine_tuning.jobs.inference.openai.v1.models(job_id="job-123")

    assert isinstance(result, dict)
    assert result["object"] == "list"
    assert result["data"][0]["id"] == "model-1"
