from operator import eq
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tokenfactory.rl.examples.math import (
    DEFAULT_DATASET_PATH,
    MathDAPODataset,
    MathExampleConfig,
    MathTask,
    build_messages,
    check_answer,
    check_format,
    reward_fn,
    roll_out_task,
)


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _parse(text: str) -> list[str]:
    return [text]


_verify = eq
NUM_DATASET_ROWS = 2
NUM_ROLLOUT_SAMPLES = 3


def test_dataset_loads_math_dapo_rows():
    load_dataset = MagicMock(
        return_value=[
            {"id": "1", "source": "aime", "problem": "  What is 1 + 1? ", "answer": " 2 "},
            {"id": "2", "source": "math", "problem": "What is 2 + 2?", "answer": "4"},
        ]
    )

    dataset = MathDAPODataset(seed=0, load_dataset_fn=load_dataset)
    sample = dataset.sample(2)

    load_dataset.assert_called_once_with(path=DEFAULT_DATASET_PATH, split="train")
    assert {task.question for task in sample} == {"What is 1 + 1?", "What is 2 + 2?"}
    assert {task.answer for task in sample} == {"2", "4"}
    assert {task.id for task in sample} == {"1", "2"}
    assert {task.source for task in sample} == {"aime", "math"}


def test_dataset_without_replacement_wraps_after_one_pass():
    dataset = MathDAPODataset(
        seed=0,
        with_replacement=False,
        load_dataset_fn=MagicMock(
            return_value=[
                {"id": "1", "source": "s", "problem": "p1", "answer": "a1"},
                {"id": "2", "source": "s", "problem": "p2", "answer": "a2"},
            ]
        ),
    )

    first = dataset.sample(2)
    second = dataset.sample(1)

    assert len(first) == NUM_DATASET_ROWS
    assert second == [first[0]]


def test_reward_matches_math_env_contract():
    assert check_format("reasoning \\boxed{42}")
    assert check_answer("reasoning \\boxed{42}", "42", parse_fn=_parse, verify_fn=_verify)
    assert reward_fn("reasoning \\boxed{42}", "42", parse_fn=_parse, verify_fn=_verify) == pytest.approx(1.0)
    assert reward_fn("reasoning \\boxed{13}", "42", parse_fn=_parse, verify_fn=_verify) == pytest.approx(0.0)
    assert reward_fn("reasoning without final box", "42", parse_fn=_parse, verify_fn=_verify) == pytest.approx(-1.0)


def test_build_messages_adds_fewshot_and_boxed_suffix():
    messages = build_messages(
        MathTask(question="What is 3 + 4?", answer="7"),
        MathExampleConfig(system_prompt="Solve carefully.", fewshot=True),
    )

    assert messages[0] == {"role": "system", "content": "Solve carefully."}
    assert messages[-1] == {
        "role": "user",
        "content": "What is 3 + 4? Write your answer in \\boxed{} format.",
    }
    assert any(message["role"] == "assistant" and "\\boxed{3}" in message["content"] for message in messages)


def test_roll_out_task_normalizes_rewards(monkeypatch: pytest.MonkeyPatch):
    completion_create = MagicMock(
        side_effect=[
            _make_completion("reasoning \\boxed{42}"),
            _make_completion("reasoning \\boxed{13}"),
            _make_completion("reasoning without final box"),
        ]
    )
    parse_completion = MagicMock(
        side_effect=[
            ([1, 2], [0.0, -0.1], [0, 1]),
            ([3, 4], [0.0, -0.2], [0, 1]),
            ([5, 6], [0.0, -0.3], [0, 1]),
        ]
    )
    monkeypatch.setattr("tokenfactory.rl.examples.math.example.parse_completion", parse_completion)

    context = SimpleNamespace(
        config=SimpleNamespace(model_name="test-model", num_samples_per_task=NUM_ROLLOUT_SAMPLES),
        openai_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion_create))),
    )

    result = roll_out_task(
        task=MathTask(question="What is the answer?", answer="42", id="task-1", source="unit"),
        context=context,  # ty: ignore[invalid-argument-type]
        my_config=MathExampleConfig(
            system_prompt="Reply with a boxed answer.",
            fewshot=False,
            max_tokens=128,
        ),
        parse_fn=_parse,
        verify_fn=_verify,
    )

    assert not result.is_rejected
    assert len(result.samples) == NUM_ROLLOUT_SAMPLES
    assert [sample.debug_info["reward"] for sample in result.samples] == [1.0, 0.0, -1.0]
    assert [sample.metadata for sample in result.samples] == [
        {"task_id": "task-1", "source": "unit"},
        {"task_id": "task-1", "source": "unit"},
        {"task_id": "task-1", "source": "unit"},
    ]
    assert [sample.normalized_reward for sample in result.samples] == pytest.approx([
        1.2247433713938008,
        0.0,
        -1.2247433713938008,
    ])
    completion_create.assert_has_calls(
        [
            call(
                model="test-model",
                messages=[
                    {"role": "system", "content": "Reply with a boxed answer."},
                    {"role": "user", "content": "What is the answer? Write your answer in \\boxed{} format."},
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=128,
                extra_body={"return_token_ids": True, "logprobs": True},
            )
        ]
        * 3
    )


def test_roll_out_task_rejects_zero_variance_groups(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "tokenfactory.rl.examples.math.example.parse_completion",
        MagicMock(return_value=([1, 2], [0.0, -0.1], [0, 1])),
    )
    context = SimpleNamespace(
        config=SimpleNamespace(model_name="test-model", num_samples_per_task=2),
        openai_client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=MagicMock(return_value=_make_completion("\\boxed{42}")))
            )
        ),
    )

    result = roll_out_task(
        task=MathTask(question="What is the answer?", answer="42"),
        context=context,  # ty: ignore[invalid-argument-type]
        my_config=MathExampleConfig(fewshot=False),
        parse_fn=_parse,
        verify_fn=_verify,
    )

    assert result.is_rejected
    assert result.samples == []
