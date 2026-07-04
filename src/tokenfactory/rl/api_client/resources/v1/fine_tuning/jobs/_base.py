from __future__ import annotations

from enum import Enum
from typing import cast

from pydantic import BaseModel, SecretBytes, SecretStr

from tokenfactory.rl.api_client.resources._base import BaseResource


class BaseJobIDResource(BaseResource):
    def endpoint(self, *, job_id: str) -> str:
        return super()._endpoint_template().format(job_id=job_id)


def request_json(model: BaseModel) -> dict[str, object]:
    data = _jsonable(model.model_dump(mode="python", exclude_none=True))
    if not isinstance(data, dict):
        raise TypeError("Request body must be a JSON object")
    return cast("dict[str, object]", data)


def optional_params(**params: object) -> dict[str, object]:
    return {key: value for key, value in params.items() if value is not None}


def _jsonable(value: object) -> object:
    if isinstance(value, SecretStr):
        result: object = value.get_secret_value()
    elif isinstance(value, SecretBytes):
        result = value.get_secret_value().decode()
    elif isinstance(value, BaseModel):
        result = _jsonable(value.model_dump(mode="python", exclude_none=True))
    elif isinstance(value, Enum):
        result = value.value
    elif isinstance(value, list | tuple):
        result = [_jsonable(item) for item in value]
    elif isinstance(value, dict):
        result = {key: _jsonable(item) for key, item in value.items()}
    else:
        result = value
    return result
