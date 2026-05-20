from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import (
    BaseJobIDResource,
)
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs.inference.openai.v1.chat._completions import (  # noqa: E501
    ChatCompletions,
)


class Chat(BaseJobIDResource):
    PATH_SEGMENT = "chat"

    @cached_property
    def completions(self) -> ChatCompletions:
        return ChatCompletions(self._client, parent=self)
