from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

    from tokenfactory.rl.api_client import TokenFactory
    from tokenfactory.rl.rollout.config import RolloutConfig


@dataclass
class RolloutContext:
    def __init__(self, config: RolloutConfig, api_client: TokenFactory):
        self._config: RolloutConfig = config
        self._api_client: TokenFactory = api_client

    @property
    def config(self) -> RolloutConfig:
        return self._config

    @cached_property
    def openai_endpoint(self) -> str:
        return self._api_client.v1alpha1.fine_tuning.jobs.inference.openai.v1.endpoint(job_id=self._config.job_id)

    @cached_property
    def openai_client(self) -> OpenAI:
        from openai import OpenAI

        return OpenAI(base_url=self.openai_endpoint, api_key=self._api_client.api_key)

    def async_openai_client(self, **kwargs) -> AsyncOpenAI:
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=self.openai_endpoint, api_key=self._api_client.api_key, **kwargs)
