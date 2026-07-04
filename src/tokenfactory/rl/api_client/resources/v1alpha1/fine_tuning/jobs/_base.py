from tokenfactory.rl.api_client.resources._base import BaseResource


class BaseJobIDResource(BaseResource):
    def endpoint(self, *, job_id: str) -> str:
        return super()._endpoint_template().format(job_id=job_id)
