from tokenfactory.rl.api_client.resources._base import BaseResource


class BaseJobIDResource(BaseResource):
    def endpoint(self, job_id: str) -> str:  # type: ignore[reportIncompatibleMethodOverride]
        return super().endpoint(job_id=job_id)
