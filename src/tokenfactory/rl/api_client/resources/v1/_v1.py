from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.resources._base import BaseBareResource
from tokenfactory.rl.api_client.resources.v1.fine_tuning import FineTuning


class V1(BaseBareResource):
    PATH_SEGMENT = "v1"

    @cached_property
    def fine_tuning(self) -> FineTuning:
        return FineTuning(self._client, parent=self)
