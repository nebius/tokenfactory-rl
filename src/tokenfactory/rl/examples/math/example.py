import logging
import random
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from math import isclose
from operator import itemgetter
from statistics import mean, pstdev
from typing import Any, cast

import click
from pydantic_settings import BaseSettings

from tokenfactory.rl.api_client import TokenFactory
from tokenfactory.rl.examples.utils import parse_completion
from tokenfactory.rl.rollout import ExecutorType, RolloutConfig, RolloutContext, RolloutRunner
from tokenfactory.rl.rollout.models import Sample, SampleGroup


logger = logging.getLogger(__name__)

QUESTION_SUFFIX = " Write your answer in \\boxed{} format."
DEFAULT_DATASET_PATH = "antonpl/math-dapo-25k"
DEFAULT_DATASET_SPLIT = "train"
ZERO_VARIANCE_TOLERANCE = 1e-6

MathParseFn = Callable[[str], list[Any]]
MathVerifyFn = Callable[[list[Any], list[Any]], bool]
LoadDatasetFn = Callable[..., Iterable[Mapping[str, Any]]]
LoadHFDatasetFn = Callable[..., Iterable[Mapping[str, Any]]]
MathParseWithKwargsFn = Callable[..., list[Any]]
MathVerifyWithKwargsFn = Callable[..., bool]


@dataclass(frozen=True)
class MathTask:
    question: str
    answer: str
    id: str | None = None
    source: str | None = None


class MathDAPODataset:
    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        split: str = DEFAULT_DATASET_SPLIT,
        seed: int | None = 0,
        with_replacement: bool = True,
        load_dataset_fn: LoadDatasetFn | None = None,
    ):
        self._random = random.Random(seed)
        self._row_index = 0
        self._with_replacement = with_replacement
        self._tasks = self._load_tasks(
            dataset_path=dataset_path,
            split=split,
            load_dataset_fn=load_dataset_fn or _load_hf_dataset,
        )
        if not self._tasks:
            raise ValueError(f"No valid math tasks found in {dataset_path}:{split}")
        if not self._with_replacement:
            self._random.shuffle(self._tasks)

    @staticmethod
    def _load_tasks(
        dataset_path: str,
        split: str,
        load_dataset_fn: LoadDatasetFn,
    ) -> list[MathTask]:
        dataset = load_dataset_fn(path=dataset_path, split=split)
        tasks: list[MathTask] = []
        for row in dataset:
            try:
                task = _task_from_row(row)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed math task row: %s", exc)
                continue
            tasks.append(task)
        return tasks

    def sample(self, n_samples: int) -> list[MathTask]:
        if n_samples <= 0:
            return []
        if len(self._tasks) < n_samples:
            raise ValueError(
                f"Not enough prompts in the dataset to sample {n_samples}. Available prompts: {len(self._tasks)}"
            )
        if self._with_replacement:
            return self._random.sample(self._tasks, n_samples)

        result = self._tasks[self._row_index : self._row_index + n_samples]
        if len(result) < n_samples:
            self._row_index = n_samples - len(result)
            result.extend(self._tasks[: self._row_index])
        else:
            self._row_index += n_samples
        return list(result)


class MathExampleConfig(BaseSettings):
    system_prompt: str = (
        "You are a helpful math assistant. Solve the problem step by step and put the final answer in \\boxed{}."
    )
    fewshot: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    drop_zero_variance_groups: bool = True
    dataset_path: str = DEFAULT_DATASET_PATH
    dataset_split: str = DEFAULT_DATASET_SPLIT
    dataset_seed: int | None = 0
    dataset_with_replacement: bool = True


def _load_hf_dataset(*, path: str, split: str) -> Iterable[Mapping[str, Any]]:
    load_dataset = cast(LoadHFDatasetFn, import_module("datasets").load_dataset)
    return load_dataset(path=path, split=split)


def _task_from_row(row: Mapping[str, Any]) -> MathTask:
    question = str(row["problem"]).strip()
    answer = str(row["answer"]).strip()
    if not question:
        raise ValueError("Empty problem")
    if not answer:
        raise ValueError("Empty answer")
    return MathTask(
        question=question,
        answer=answer,
        id=str(row["id"]) if row.get("id") is not None else None,
        source=str(row["source"]) if row.get("source") is not None else None,
    )


def _extract_boxed(text: str) -> str:
    boxed_strs: list[tuple[int, str]] = []
    stack: list[int] = []
    for ichar, char in enumerate(text):
        if char == "{":
            stack.append(ichar)
        elif char == "}":
            if not stack:
                raise ValueError("Unmatched }")
            last_open_start = stack.pop()
            if text[:last_open_start].endswith("\\boxed"):
                boxed_strs.append((last_open_start, text[last_open_start + 1 : ichar]))

    boxed_strs.extend((match.start(), match.group(1)) for match in re.finditer(r"\\boxed\s+([a-zA-Z0-9]+)", text))

    if boxed_strs:
        return max(boxed_strs, key=itemgetter(0))[1]
    raise ValueError("No boxed strings found")


def _default_parse(text: str) -> list[Any]:
    math_parse = cast(MathParseWithKwargsFn, import_module("math_verify").parse)
    return math_parse(text, parsing_timeout=None)


def _default_verify(expected: list[Any], actual: list[Any]) -> bool:
    math_verify = cast(MathVerifyWithKwargsFn, import_module("math_verify").verify)
    return bool(math_verify(expected, actual, timeout_seconds=None))


def _parse_answer(answer: str, parse_fn: MathParseFn) -> list[Any]:
    return parse_fn(rf"\boxed{{{answer}}}") or parse_fn(answer)


def check_format(response: str) -> bool:
    try:
        _extract_boxed(response)
    except ValueError:
        return False
    return True


def check_answer(
    response: str,
    answer: str,
    parse_fn: MathParseFn | None = None,
    verify_fn: MathVerifyFn | None = None,
) -> bool:
    try:
        response_answer = _extract_boxed(response)
    except ValueError:
        return False

    parse = parse_fn or _default_parse
    verify = verify_fn or _default_verify
    return verify(
        _parse_answer(answer, parse),
        _parse_answer(response_answer, parse),
    )


def reward_fn(
    response: str,
    answer: str,
    parse_fn: MathParseFn | None = None,
    verify_fn: MathVerifyFn | None = None,
) -> float:
    if not check_format(response):
        return -1.0
    return float(check_answer(response, answer, parse_fn=parse_fn, verify_fn=verify_fn))


def build_messages(task: MathTask, my_config: MathExampleConfig) -> list[dict[str, str]]:
    messages = []
    if my_config.system_prompt:
        messages.append({"role": "system", "content": my_config.system_prompt})
    if my_config.fewshot:
        messages.extend(standard_fewshot_prefix())
    messages.append({"role": "user", "content": task.question + QUESTION_SUFFIX})
    return messages


def standard_fewshot_prefix() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "How many r's are in strawberry?" + QUESTION_SUFFIX,
        },
        {
            "role": "assistant",
            "content": (
                "Let's spell the word out and number all the letters: 1) s 2) t "
                "3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at "
                "positions 3, 8, and 9. \\boxed{3}"
            ),
        },
    ]


def roll_out_task(
    task: MathTask,
    context: RolloutContext,
    my_config: MathExampleConfig,
    parse_fn: MathParseFn | None = None,
    verify_fn: MathVerifyFn | None = None,
) -> SampleGroup:
    messages = build_messages(task, my_config)
    samples = []

    for _ in range(context.config.num_samples_per_task):
        completion = context.openai_client.chat.completions.create(
            model=context.config.model_name,
            messages=messages,  # ty: ignore[invalid-argument-type]
            temperature=my_config.temperature,
            top_p=my_config.top_p,
            max_tokens=my_config.max_tokens,
            extra_body={"return_token_ids": True, "logprobs": True},
        )
        token_ids, logprobs, mask = parse_completion(completion)

        response = completion.choices[0].message.content
        assert response is not None
        reward = reward_fn(
            response=response,
            answer=task.answer,
            parse_fn=parse_fn,
            verify_fn=verify_fn,
        )
        is_correct = isclose(reward, 1.0)
        is_format_correct = reward >= 0.0

        samples.append(
            Sample(
                token_ids=token_ids,
                logprobs=logprobs,
                mask=mask,
                normalized_reward=reward,
                metadata=_task_metadata(task),
                debug_info={
                    "reward": reward,
                    "format": float(is_format_correct),
                    "correct": is_correct,
                    "resolved_rate": float(is_correct),
                    "prompt_text": task.question + QUESTION_SUFFIX,
                    "generated_text": response,
                    "reference_answer": task.answer,
                    "ground_truth_answer": task.answer,
                },
            )
        )

    rewards = [sample.normalized_reward for sample in samples]
    reward_stddev = pstdev(rewards)
    if my_config.drop_zero_variance_groups and reward_stddev < ZERO_VARIANCE_TOLERANCE:
        return SampleGroup(is_rejected=True)

    reward_mean = mean(rewards)
    for sample in samples:
        sample.normalized_reward = (sample.normalized_reward - reward_mean) / (reward_stddev + ZERO_VARIANCE_TOLERANCE)

    return SampleGroup(samples=samples)


def _task_metadata(task: MathTask) -> dict[str, str]:
    metadata = {}
    if task.id is not None:
        metadata["task_id"] = task.id
    if task.source is not None:
        metadata["source"] = task.source
    return metadata


@click.command()
@click.option("--job-id", required=True, help="TokenFactory fine-tuning job ID.")
@click.option("--model-name", required=True, help="Model name passed to the job inference endpoint.")
@click.option("--num-batches", type=int, default=1000, show_default=True, help="Number of batches to fill.")
@click.option("--allowed-staleness", type=int, default=0, show_default=True, help="Allowed inference-version lag.")
@click.option(
    "--job-init-timeout",
    type=int,
    default=600,
    show_default=True,
    help="Seconds to wait for the job runtime status endpoint during startup.",
)
@click.option("--batch-size", type=int, default=128, show_default=True, help="Number of samples per training batch.")
@click.option(
    "--num-samples-per-task",
    type=int,
    default=4,
    show_default=True,
    help="Number of completions sampled for each math problem.",
)
def main(
    job_id: str,
    model_name: str,
    num_batches: int,
    allowed_staleness: int,
    job_init_timeout: int,
    batch_size: int,
    num_samples_per_task: int,
) -> None:
    logging.basicConfig(level=logging.DEBUG)
    my_config = MathExampleConfig()

    runner = RolloutRunner(
        api_client=TokenFactory(),
        config=RolloutConfig(
            job_id=job_id,
            model_name=model_name,
            max_concurrency=32,
            executor_type=ExecutorType.THREAD,
            allowed_staleness=allowed_staleness,
            job_init_timeout=job_init_timeout,
            num_batches=num_batches,
            batch_size=batch_size,
            num_samples_per_task=num_samples_per_task,
        ),
        dataset=MathDAPODataset(
            dataset_path=my_config.dataset_path,
            split=my_config.dataset_split,
            seed=my_config.dataset_seed,
            with_replacement=my_config.dataset_with_replacement,
        ),
        rollout_fn=partial(roll_out_task, my_config=my_config),
    )

    runner.run()


if __name__ == "__main__":
    main()
