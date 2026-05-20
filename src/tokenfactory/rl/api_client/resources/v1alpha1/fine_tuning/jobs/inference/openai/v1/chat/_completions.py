from __future__ import annotations

from tokenfactory.rl.api_client.models import ChatCompletion, OpenAIChatCompletionRequest
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import BaseJobIDResource


class ChatCompletions(BaseJobIDResource):
    PATH_SEGMENT = "completions"

    def create(self, *, job_id: str, **kwargs) -> ChatCompletion:
        request = OpenAIChatCompletionRequest(**kwargs)
        response = self._client.post(
            self.endpoint(job_id),
            json=request.model_dump(),
        )
        return ChatCompletion.model_validate(response.json())
