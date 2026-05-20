from __future__ import annotations

from functools import cached_property
from typing import Any

from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import (
    BaseJobIDResource,
)
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs.inference.openai.v1.chat import (  # noqa: E501
    Chat,
)
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs.inference.openai.v1.completions import (  # noqa: E501
    Completions,
)


class V1(BaseJobIDResource):
    PATH_SEGMENT = "v1"

    @cached_property
    def chat(self) -> Chat:
        return Chat(self._client, parent=self)

    @cached_property
    def completions(self) -> Completions:
        return Completions(self._client, parent=self)

    def models(self, *, job_id: str) -> dict[str, Any]:
        response = self._client.get(
            f"{self.endpoint(job_id)}/models",
        )
        return response.json()
