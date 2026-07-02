from operator import eq
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from click.testing import CliRunner

from tokenfactory.rl.examples.math import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DATASET_SPLIT,
    MathDAPODataset,
    MathExampleConfig,
    MathTask,
    build_messages,
    check_answer,
    check_format,
    main,
    reward_fn,
    roll_out_task,
)
from tokenfactory.rl.rollout import RolloutConfig


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _parse(text: str) -> list[str]:
    return [text]


_verify = eq
NUM_DATASET_ROWS = 2
NUM_ROLLOUT_SAMPLES = 3
CLI_JOB_ID = "job-test"
CLI_MODEL_NAME = "test-model"
CLI_NUM_BATCHES = 7
CLI_ALLOWED_STALENESS = 2
CLI_JOB_INIT_TIMEOUT = 45
CLI_BATCH_SIZE = 12
CLI_NUM_SAMPLES_PER_TASK = 4
DEFAULT_MAX_CONCURRENCY = 32


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


def test_main_builds_runner_from_cli_options(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    fake_api_client = object()
    fake_dataset = object()

    class FakeRunner:
        def __init__(
            self,
            api_client: object,
            config: RolloutConfig,
            dataset: object,
            rollout_fn: object,
        ) -> None:
            captured["api_client"] = api_client
            captured["config"] = config
            captured["dataset"] = dataset
            captured["rollout_fn"] = rollout_fn

        @staticmethod
        def run() -> None:
            captured["ran"] = True

    def fake_token_factory() -> object:
        return fake_api_client

    def fake_dataset_factory(
        *,
        dataset_path: str,
        split: str,
        seed: int | None,
        with_replacement: bool,
    ) -> object:
        captured["dataset_args"] = {
            "dataset_path": dataset_path,
            "split": split,
            "seed": seed,
            "with_replacement": with_replacement,
        }
        return fake_dataset

    monkeypatch.setattr("tokenfactory.rl.examples.math.example.TokenFactory", fake_token_factory)
    monkeypatch.setattr("tokenfactory.rl.examples.math.example.RolloutRunner", FakeRunner)
    monkeypatch.setattr("tokenfactory.rl.examples.math.example.MathDAPODataset", fake_dataset_factory)

    result = CliRunner().invoke(
        main,
        [
            "--job-id",
            CLI_JOB_ID,
            "--model-name",
            CLI_MODEL_NAME,
            "--num-batches",
            str(CLI_NUM_BATCHES),
            "--allowed-staleness",
            str(CLI_ALLOWED_STALENESS),
            "--job-init-timeout",
            str(CLI_JOB_INIT_TIMEOUT),
            "--batch-size",
            str(CLI_BATCH_SIZE),
            "--num-samples-per-task",
            str(CLI_NUM_SAMPLES_PER_TASK),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["api_client"] is fake_api_client
    assert captured["dataset"] is fake_dataset
    assert captured["ran"] is True
    assert captured["dataset_args"] == {
        "dataset_path": DEFAULT_DATASET_PATH,
        "split": DEFAULT_DATASET_SPLIT,
        "seed": 0,
        "with_replacement": True,
    }

    config = captured["config"]
    assert isinstance(config, RolloutConfig)
    assert config.job_id == CLI_JOB_ID
    assert config.model_name == CLI_MODEL_NAME
    assert config.max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert config.num_batches == CLI_NUM_BATCHES
    assert config.allowed_staleness == CLI_ALLOWED_STALENESS
    assert config.job_init_timeout == CLI_JOB_INIT_TIMEOUT
    assert config.batch_size == CLI_BATCH_SIZE
    assert config.num_samples_per_task == CLI_NUM_SAMPLES_PER_TASK
