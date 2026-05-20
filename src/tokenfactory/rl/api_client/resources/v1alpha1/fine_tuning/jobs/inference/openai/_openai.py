from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import (
    BaseJobIDResource,
)
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs.inference.openai.v1 import (
    V1,
)


class OpenAI(BaseJobIDResource):
    PATH_SEGMENT = "openai"

    @cached_property
    def v1(self) -> V1:
        return V1(self._client, parent=self)
