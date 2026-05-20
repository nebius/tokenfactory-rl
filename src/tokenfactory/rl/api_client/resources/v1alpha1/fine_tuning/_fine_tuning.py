from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.resources._base import BaseResource
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs import Jobs


class FineTuning(BaseResource):
    PATH_SEGMENT = "fine_tuning"

    @cached_property
    def jobs(self) -> Jobs:
        return Jobs(self._client, parent=self)
