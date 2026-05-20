import logging
from dataclasses import dataclass
from functools import partial
from random import choices
from statistics import mean, pstdev

from pydantic_settings import BaseSettings

from tokenfactory.rl.api_client import TokenFactory
from tokenfactory.rl.rollout import RolloutConfig, RolloutContext, RolloutRunner
from tokenfactory.rl.rollout.models import Sample, SampleGroup

from .utils import parse_completion


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


def roll_out_task(
    task: Task, context: RolloutContext, my_config: MyConfig
) -> SampleGroup:
    messages = [
        {"role": "system", "content": my_config.system_prompt},
        {"role": "user", "content": task.question},
    ]

    samples = []
    for _ in range(context.config.num_samples_per_task):
        completion = context.openai_client.chat.completions.create(
            model=context.config.model_name,
            messages=messages,  # ty: ignore[invalid-argument-type]
            extra_body={"return_token_ids": True, "logprobs": True},
        )
        token_ids, logprobs, mask = parse_completion(completion)

        assert completion.choices[0].message.content is not None
        reward = reward_fn(
            response=completion.choices[0].message.content, answer=task.answer
        )

        samples.append(
            Sample(
                token_ids=token_ids,
                logprobs=logprobs,
                mask=mask,
                normalized_reward=reward,
                debug_info={"reward": reward},
            )
        )

    rewards = [s.normalized_reward for s in samples]
    reward_mean = mean(rewards)
    reward_stddev = pstdev(rewards)
    for s in samples:
        s.normalized_reward = (s.normalized_reward - reward_mean) / (
            reward_stddev + 1e-6
        )

    return SampleGroup(samples=samples)


def main():
    logging.basicConfig(level=logging.DEBUG)

    runner = RolloutRunner(
        api_client=TokenFactory(),
        config=RolloutConfig(
            job_id="example-job-id",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_concurrency=32,
            executor_type="thread",
            allowed_staleness=0,
            num_batches=1000,
            batch_size=128,
            num_samples_per_task=4,
        ),
        dataset=Dataset(),
        rollout_fn=partial(roll_out_task, my_config=MyConfig()),
    )

    runner.run()


if __name__ == "__main__":
    main()
