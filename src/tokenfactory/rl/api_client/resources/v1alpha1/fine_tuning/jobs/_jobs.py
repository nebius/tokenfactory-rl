from __future__ import annotations

from functools import cached_property

from tokenfactory.rl.api_client.models.rl_job import JobStatus
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import BaseJobIDResource
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._batches import Batches


class Jobs(BaseJobIDResource):
    BARE_PATH_SEGMENT = "jobs"
    PATH_SEGMENT = BARE_PATH_SEGMENT + "/{job_id}"

    def bare_endpoint(self) -> str:
        return self._endpoint_template(last_segment=self.BARE_PATH_SEGMENT)

    def get_runtime_status(self, *, job_id: str) -> JobStatus:
        response = self._client.get(f"{self.endpoint(job_id=job_id)}/status")
        return JobStatus.model_validate(response.json())

    @cached_property
    def batches(self) -> Batches:
        return Batches(self._client, parent=self)
