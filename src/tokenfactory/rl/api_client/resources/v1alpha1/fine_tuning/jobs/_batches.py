from __future__ import annotations

from tokenfactory.rl.api_client.models.rl_job import Batch, CreateBatchRequest, Sample, UploadSamplesRequest
from tokenfactory.rl.api_client.resources.v1alpha1.fine_tuning.jobs._base import BaseJobIDResource


class Batches(BaseJobIDResource):
    PATH_SEGMENT = "batches"

    def create(self, *, job_id: str, index: int) -> Batch:
        request = CreateBatchRequest(index=index)
        response = self._client.post(
            self.endpoint(job_id=job_id),
            json=request.model_dump(),
        )
        return Batch.model_validate(response.json())

    def get(self, *, job_id: str, batch_index: int) -> Batch:
        response = self._client.get(f"{self.endpoint(job_id=job_id)}/{batch_index}")
        return Batch.model_validate(response.json())

    def submit_samples(self, *, job_id: str, batch_index: int, samples: list[Sample]) -> Batch:
        request = UploadSamplesRequest(samples=samples)
        response = self._client.post(
            f"{self.endpoint(job_id=job_id)}/{batch_index}/samples",
            json=request.model_dump(),
        )
        return Batch.model_validate(response.json())
