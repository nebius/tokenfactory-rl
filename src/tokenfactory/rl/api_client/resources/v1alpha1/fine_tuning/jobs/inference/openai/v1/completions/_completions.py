from __future__ import annotations

from tokenfactory.rl.api_client.models import Completion, OpenAICompletionRequest
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import (
    BaseJobIDResource,
)


class Completions(BaseJobIDResource):
    PATH_SEGMENT = "completions"

    def create(self, *, job_id: str, **kwargs) -> Completion:
        request = OpenAICompletionRequest(**kwargs)
        response = self._client.post(
            self.endpoint(job_id),
            json=request.model_dump(),
        )
        return Completion.model_validate(response.json())
