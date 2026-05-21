from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import BaseJobIDResource
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs.inference.openai import OpenAI


class Inference(BaseJobIDResource):
    PATH_SEGMENT = "inference"

    @cached_property
    def openai(self) -> OpenAI:
        return OpenAI(self._client, parent=self)
