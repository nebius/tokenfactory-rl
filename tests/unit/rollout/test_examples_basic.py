from types import SimpleNamespace
from unittest.mock import MagicMock

from tokenfactory.rl.examples.basic import MyConfig, Task, roll_out_task


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_roll_out_task_returns_single_sample(monkeypatch):
    completion_create = MagicMock(return_value=_make_completion("The answer is 42."))
    parse_completion = MagicMock(return_value=([1, 2], [0.0, -0.1], [0, 1]))
    monkeypatch.setattr("tokenfactory.rl.examples.basic.parse_completion", parse_completion)

    context = SimpleNamespace(
        config=SimpleNamespace(model_name="test-model", num_samples_per_task=1),
        openai_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion_create))),
    )

    result = roll_out_task(
        task=Task(question="Tell me a random number.", answer="42"),
        context=context,  # ty: ignore[invalid-argument-type]
        my_config=MyConfig(system_prompt="Reply with the answer only."),
    )

    assert len(result.samples) == 1
    assert result.samples[0].token_ids == [1, 2]
    assert result.samples[0].logprobs == [0.0, -0.1]
    assert result.samples[0].mask == [0, 1]
    assert result.samples[0].normalized_reward == 0.5
    completion_create.assert_called_once_with(
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
