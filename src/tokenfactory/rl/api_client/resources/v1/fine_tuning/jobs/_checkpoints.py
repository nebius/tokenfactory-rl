from __future__ import annotations

from tokenfactory.rl.api_client.models.fine_tuning import ListCheckpointsResponse
from tokenfactory.rl.api_client.models.fine_tuning import (
    OpenaiTypesFineTuningJobsFineTuningJobCheckpointFineTuningJobCheckpoint as FineTuningJobCheckpoint,
)
from tokenfactory.rl.api_client.resources.v1.fine_tuning.jobs._base import BaseJobIDResource, optional_params


class Checkpoints(BaseJobIDResource):
    PATH_SEGMENT = "checkpoints"

    def list(self, *, job_id: str, limit: int | None = None, after: str | None = None) -> ListCheckpointsResponse:
        response = self._client.get(
            self.endpoint(job_id=job_id),
            params=optional_params(limit=limit, after=after),
        )
        return ListCheckpointsResponse.model_validate(response.json())

    def get(self, *, job_id: str, checkpoint_id: str) -> FineTuningJobCheckpoint:
        response = self._client.get(f"{self.endpoint(job_id=job_id)}/{checkpoint_id}")
        return FineTuningJobCheckpoint.model_validate(response.json())
