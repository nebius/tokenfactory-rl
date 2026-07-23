from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tokenfactory.rl.examples.grpo import MyConfig, Task, roll_out_task


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_roll_out_task_returns_normalized_rewards_for_all_samples(monkeypatch):
    completion_create = MagicMock(
        side_effect=[
            _make_completion("42"),
            _make_completion("The answer is 42."),
            _make_completion("13"),
        ]
    )
    parse_completion = MagicMock(
        side_effect=[
            ([1, 2], [0.0, -0.1], [0, 1]),
            ([3, 4], [0.0, -0.2], [0, 1]),
            ([5, 6], [0.0, -0.3], [0, 1]),
        ]
    )
    monkeypatch.setattr("tokenfactory.rl.examples.grpo.parse_completion", parse_completion)

    context = SimpleNamespace(
        config=SimpleNamespace(model_name="test-model", num_samples_per_task=3),
        openai_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion_create))),
    )

    result = roll_out_task(
        task=Task(question="Tell me a random number.", answer="42"),
        context=context,  # ty: ignore[invalid-argument-type]
        my_config=MyConfig(system_prompt="Reply with the answer only."),
    )

    assert len(result.samples) == 3
    assert [sample.token_ids for sample in result.samples] == [[1, 2], [3, 4], [5, 6]]
    assert [sample.logprobs for sample in result.samples] == [
        [0.0, -0.1],
        [0.0, -0.2],
        [0.0, -0.3],
    ]
    assert [sample.mask for sample in result.samples] == [[0, 1], [0, 1], [0, 1]]
    assert [sample.debug_info for sample in result.samples] == [
        {"reward": 1.0},
        {"reward": 0.5},
        {"reward": 0.0},
    ]
    assert [sample.normalized_reward for sample in result.samples] == pytest.approx(
        [
            1.224741871398925,
            0.0,
            -1.224741871398925,
        ]
    )
    assert completion_create.call_count == 3
    completion_create.assert_has_calls(
        [
            call(
                model="test-model",
                messages=[
                    {
                        "role": "system",
                        "content": "Reply with the answer only.",
                    },
                    {"role": "user", "content": "Tell me a random number."},
                ],
                extra_body={"return_token_ids": True, "logprobs": True},
            )
        ]
        * 3
    )
