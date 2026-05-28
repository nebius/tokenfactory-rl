# tokenfactory-rl

Python utilities for collecting reinforcement fine-tuning rollout data for TokenFactory jobs.

This package does not create or configure fine-tuning jobs. It assumes a TokenFactory RFT job already exists, then helps you:

- poll the job runtime status and current inference model version;
- run your own rollout function concurrently against the job's OpenAI-compatible inference endpoint;
- convert completions into token-level training samples;
- schedule samples into batches while respecting model-version staleness rules;
- upload completed batches back to the TokenFactory API.

For the internal end-to-end job workflow, see [RFT Guide v2](https://nebius.atlassian.net/wiki/spaces/NBAI/pages/1831534780/RFT+Guide+v2).

## Installation

The package requires Python `>=3.10,<3.15`.

```bash
pip install tokenfactory-rl
```

For local development from this repository:

```bash
uv sync --group dev
```

## Authentication

`TokenFactory()` reads credentials from the environment by default:

```bash
export TOKENFACTORY_API_KEY="..."
```

Optional API endpoint override:

```bash
export TOKENFACTORY_BASE_URL="https://api.tokenfactory.nebius.com"
```

If `TOKENFACTORY_BASE_URL` is not set, the client uses `https://api.tokenfactory.nebius.com`.

## Quick Start

A rollout has two user-owned pieces:

- a dataset with `sample(n_samples)`;
- a rollout function that accepts one task and returns a `SampleGroup`.

```python
from dataclasses import dataclass
from functools import partial
from random import choices

from pydantic_settings import BaseSettings

from tokenfactory.rl.api_client import TokenFactory
from tokenfactory.rl.examples.utils import parse_completion
from tokenfactory.rl.rollout import RolloutConfig, RolloutContext, RolloutRunner
from tokenfactory.rl.rollout.models import Sample, SampleGroup


@dataclass
class Task:
    question: str
    answer: str


class Dataset:
    def __init__(self):
        self._tasks = [
            Task(question="Tell me a random number.", answer="42"),
        ]

    def sample(self, n_samples: int) -> list[Task]:
        return choices(self._tasks, k=n_samples)


class MyConfig(BaseSettings):
    system_prompt: str = "You are a helpful assistant. Reply with the answer only."


def reward_fn(response: str, answer: str) -> float:
    text = response.strip()
    if text == answer:
        return 1.0
    if answer in text:
        return 0.5
    return 0.0


def roll_out_task(task: Task, context: RolloutContext, my_config: MyConfig) -> SampleGroup:
    completion = context.openai_client.chat.completions.create(
        model=context.config.model_name,
        messages=[
            {"role": "system", "content": my_config.system_prompt},
            {"role": "user", "content": task.question},
        ],
        extra_body={"return_token_ids": True, "logprobs": True},
    )

    token_ids, logprobs, mask = parse_completion(completion)
    response = completion.choices[0].message.content
    assert response is not None

    return SampleGroup(samples=[
        Sample(
            token_ids=token_ids,
            logprobs=logprobs,
            mask=mask,
            normalized_reward=reward_fn(response=response, answer=task.answer),
        )
    ])


runner = RolloutRunner(
    api_client=TokenFactory(),
    config=RolloutConfig(
        job_id="your-tokenfactory-job-id",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        batch_size=128,
        num_batches=1000,
        num_samples_per_task=1,
        max_concurrency=32,
        allowed_staleness=0,
    ),
    dataset=Dataset(),
    rollout_fn=partial(roll_out_task, my_config=MyConfig()),
)

runner.run()
```

Runnable templates are also available in:

- `src/tokenfactory/rl/examples/basic.py`: one completion per task;
- `src/tokenfactory/rl/examples/grpo.py`: multiple completions per task with group-normalized rewards.

Update the example `job_id` before running an example module:

```bash
uv run python -m tokenfactory.rl.examples.basic
uv run python -m tokenfactory.rl.examples.grpo
```

## Rollout Samples

Each accepted rollout returns a `SampleGroup` containing exactly `num_samples_per_task` samples.

`Sample` fields:

- `token_ids`: prompt and completion token IDs;
- `logprobs`: token log probabilities, usually `0.0` for prompt tokens and model logprobs for completion tokens;
- `mask`: `0` for context tokens and `1` for tokens that should receive loss;
- `normalized_reward`: reward value consumed by training;
- `metadata`: optional string metadata sent with the sample;
- `debug_info`: optional diagnostic payload, also used by the runner to record `starting_inference_version`.

Return `SampleGroup(is_rejected=True)` to discard a task without submitting samples.

## Configuration

`RolloutConfig` can be created directly or loaded from environment variables with the `TOKENFACTORY_ROLLOUT_` prefix.

```bash
export TOKENFACTORY_ROLLOUT_JOB_ID="your-tokenfactory-job-id"
export TOKENFACTORY_ROLLOUT_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export TOKENFACTORY_ROLLOUT_BATCH_SIZE="128"
export TOKENFACTORY_ROLLOUT_NUM_SAMPLES_PER_TASK="4"
export TOKENFACTORY_ROLLOUT_MAX_CONCURRENCY="32"
```

Important settings:

| Setting | Default | Description |
| --- | --- | --- |
| `job_id` | required | TokenFactory fine-tuning job ID. |
| `model_name` | required | Model name passed to the job inference endpoint. |
| `batch_size` | required | Number of samples per training batch. Must be divisible by `num_samples_per_task`. |
| `num_samples_per_task` | required | Number of samples produced for each dataset task. |
| `num_batches` | `None` | Number of batches to fill. `None` runs continuously. |
| `max_concurrency` | `32` | Maximum concurrent rollout samples. |
| `allowed_staleness` | `0` | Allowed lag between a rollout's starting inference version and the batch it enters. Set to `None` to disable staleness checks. |
| `drop_stale_trajectories` | `False` | If true, stale sample groups are dropped. If false, the scheduler throttles and buffers instead. |
| `num_rollout_retries` | `0` | Number of retries after rollout function failures. |
| `raise_on_rollout_failure` | `False` | Raise exhausted rollout failures instead of rejecting that task. |
| `status_polling_interval` | `1.0` | Seconds between job status polls. |
| `job_init_timeout` | `600` | Seconds to wait for the job status endpoint during startup. Set to `None` to wait forever. |

`executor_type` is accepted by the config for compatibility, but the current runner dispatches rollouts with a thread pool.

## Scheduling Behavior

The runner uses the job's `inference_version` to avoid silently mixing very old trajectories into newer training batches.

- `allowed_staleness=None`: accept samples regardless of version gap.
- `allowed_staleness=N` and `drop_stale_trajectories=True`: upload fresh sample groups and drop groups that exceed the staleness window.
- `allowed_staleness=N` and `drop_stale_trajectories=False`: preserve every accepted sample group by buffering or slowing new rollouts when needed.

The runner resumes from the job's current `filled_batches` value and creates the current batch if it does not already exist.

## API Client

You can also use the low-level client directly:

```python
from tokenfactory.rl.api_client import TokenFactory

client = TokenFactory()

status = client.v1alpha1.fine_tuning.jobs.get_runtime_status(job_id="job-123")
batch = client.v1alpha1.fine_tuning.jobs.batches.get(job_id="job-123", batch_index=0)

models = client.v1alpha1.fine_tuning.jobs.inference.openai.v1.models(job_id="job-123")
```

The rollout context exposes the job's OpenAI-compatible inference endpoint:

```python
completion = context.openai_client.chat.completions.create(
    model=context.config.model_name,
    messages=[{"role": "user", "content": "Hello"}],
)
```

`context.async_openai_client()` creates an async OpenAI client for the same endpoint.

## Development

Run tests:

```bash
uv run --group tests pytest
```

Run code checks:

```bash
uv run --group dev ruff check
uv run --group dev ruff format --check
uv run --group dev ty check
uv lock --check
```

Build the package:

```bash
uv build
```

The generated API models live in `src/tokenfactory/rl/api_client/models.py`; see `src/tokenfactory/rl/api_client/README.md` for regeneration notes.

## Project Layout

```text
src/tokenfactory/rl/
  api_client/      TokenFactory HTTP client and generated API models
  rollout/         public rollout config, context, runner, dataset protocol, and sample models
  examples/        basic and GRPO-style rollout examples
  schedulers.py    pure scheduling logic for batching and staleness handling
tests/
  unit/api_client/ API client tests
  unit/rollout/    runner, scheduler, config, model, context, and example tests
```

## License

Copyright 2026 Nebius B.V.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

_Apache and the Apache logo are either registered trademarks or trademarks of The Apache Software Foundation in the United States and/or other countries._