from __future__ import annotations

from functools import cached_property
from typing import TypeAlias

from tokenfactory.rl.api_client.models.fine_tuning import (
    FineTuningJob,
    FineTuningJobListResponse,
    FineTuningRequest,
    FTCheckpointParameters,
    HFCheckpointParametersRequestInput,
    HfExportIntegrationRequest,
    ListEventsResponse,
    MlflowIntegrationRequest,
    SpecDraftMethodConfig,
    Suffix1,
    SupervisedHParameters,
    SupervisedMethodConfig,
    WandbIntegrationRequest,
)
from tokenfactory.rl.api_client.resources.v1.fine_tuning.jobs._base import (
    BaseJobIDResource,
    optional_params,
    request_json,
)
from tokenfactory.rl.api_client.resources.v1.fine_tuning.jobs._checkpoints import Checkpoints


CheckpointParameters = HFCheckpointParametersRequestInput | FTCheckpointParameters
FineTuningMethod = SupervisedMethodConfig | SpecDraftMethodConfig
IntegrationRequest = WandbIntegrationRequest | MlflowIntegrationRequest | HfExportIntegrationRequest
IntegrationRequests: TypeAlias = list[IntegrationRequest]
Tags: TypeAlias = list[str]


class Jobs(BaseJobIDResource):
    BARE_PATH_SEGMENT = "jobs"
    PATH_SEGMENT = BARE_PATH_SEGMENT + "/{job_id}"

    def bare_endpoint(self) -> str:
        return self._endpoint_template(last_segment=self.BARE_PATH_SEGMENT)

    def create(
        self,
        *,
        model: str,
        training_file: str,
        ai_project_id: str | None = None,
        from_checkpoint: CheckpointParameters | None = None,
        validation_file: str | None = None,
        hyperparameters: SupervisedHParameters | None = None,
        method: FineTuningMethod | None = None,
        integrations: IntegrationRequests | None = None,
        seed: int | None = None,
        suffix: str | Suffix1 | None = None,
        tags: Tags | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> FineTuningJob:
        request_suffix = Suffix1(root=suffix) if isinstance(suffix, str) else suffix
        request = FineTuningRequest(
            model=model,
            from_checkpoint=from_checkpoint,
            training_file=training_file,
            validation_file=validation_file,
            hyperparameters=hyperparameters,
            method=method,
            integrations=integrations,
            seed=seed,
            suffix=request_suffix,
            tags=tags,
            extra_body=extra_body,
        )
        response = self._client.post(
            self.bare_endpoint(),
            params=optional_params(ai_project_id=ai_project_id),
            json=request_json(request),
        )
        return FineTuningJob.model_validate(response.json())

    def list(
        self,
        *,
        limit: int | None = None,
        after: str | None = None,
        ai_project_id: str | None = None,
    ) -> FineTuningJobListResponse:
        response = self._client.get(
            self.bare_endpoint(),
            params=optional_params(limit=limit, after=after, ai_project_id=ai_project_id),
        )
        return FineTuningJobListResponse.model_validate(response.json())

    def get(self, *, job_id: str) -> FineTuningJob:
        response = self._client.get(self.endpoint(job_id=job_id))
        return FineTuningJob.model_validate(response.json())

    def cancel(self, *, job_id: str) -> FineTuningJob:
        response = self._client.post(f"{self.endpoint(job_id=job_id)}/cancel")
        return FineTuningJob.model_validate(response.json())

    def get_events(
        self,
        *,
        job_id: str,
        limit: int | None = None,
        after: str | None = None,
    ) -> ListEventsResponse:
        response = self._client.get(
            f"{self.endpoint(job_id=job_id)}/events",
            params=optional_params(limit=limit, after=after),
        )
        return ListEventsResponse.model_validate(response.json())

    @cached_property
    def checkpoints(self) -> Checkpoints:
        return Checkpoints(self._client, parent=self)
