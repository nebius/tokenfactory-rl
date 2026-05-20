from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.resources._base import BaseResource
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning import FineTuning


class V1Alpha1(BaseResource):
    PATH_SEGMENT = "v1alpha1"

    @cached_property
    def fine_tuning(self) -> FineTuning:
        return FineTuning(self._client, parent=self)
