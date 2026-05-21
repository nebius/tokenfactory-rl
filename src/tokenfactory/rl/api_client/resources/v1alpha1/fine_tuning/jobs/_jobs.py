from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.models import JobStatus
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import BaseJobIDResource
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._batches import Batches
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs.inference import Inference


class Jobs(BaseJobIDResource):
    PATH_SEGMENT = "jobs/{job_id}"

    def get_runtime_status(self, *, job_id: str) -> JobStatus:
        response = self._client.get(f"{self.endpoint(job_id=job_id)}/status")
        return JobStatus.model_validate(response.json())

    @cached_property
    def batches(self) -> Batches:
        return Batches(self._client, parent=self)

    @cached_property
    def inference(self) -> Inference:
        return Inference(self._client, parent=self)
