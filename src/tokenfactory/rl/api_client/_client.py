from __future__ import annotations

import os
from functools import cached_property
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from tokenfactory.rl.api_client._exceptions import (
    BadRequestError,
    InformationalResponseError,
    InternalServerError,
    JsonParseError,
    MissingAPIKeyError,
    NotFoundError,
    RedirectResponseError,
)
from tokenfactory.rl.api_client.resources.v1alpha1 import V1Alpha1


if TYPE_CHECKING:
    from collections.abc import Callable

    from tenacity.wait import wait_base
    from typing_extensions import Self

ENV_PREFIX = "TOKENFACTORY_"
BASE_URL_ENV_VAR = f"{ENV_PREFIX}BASE_URL"
API_KEY_ENV_VAR = f"{ENV_PREFIX}API_KEY"

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com"


def default_httpx_client_factory() -> httpx.Client:
    return httpx.Client()


class TokenFactory:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        httpx_client: httpx.Client | None = None,
        httpx_client_factory: Callable[[], httpx.Client] | None = None,
        max_retries: int = 3,
        tenacity_retry_wait: wait_base | None = None,
    ) -> None:
        if base_url is None:
            base_url = os.environ.get(BASE_URL_ENV_VAR)
        if base_url is None:
            base_url = DEFAULT_BASE_URL
        self._base_url = base_url

        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV_VAR)
        if api_key is None:
            raise MissingAPIKeyError(f"Missing API key, set env var {API_KEY_ENV_VAR} or pass to constructor")
        self._api_key = api_key
        self._max_retries = max_retries
        self._tenacity_retry_wait = tenacity_retry_wait

        self._httpx_client_factory = httpx_client_factory or default_httpx_client_factory
        self._httpx_client = self._httpx_client_factory()

    @property
    def httpx_client(self) -> httpx.Client:
        if self._httpx_client is None:
            self._httpx_client = self._httpx_client_factory()
        return self._httpx_client

    @cached_property
    def v1alpha1(self) -> V1Alpha1:
        return V1Alpha1(self)

    @property
    def base_url(self) -> str:
        return str(self._base_url).rstrip("/")

    @property
    def api_key(self) -> str:
        return self._api_key

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        @retry(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10)
            if self._tenacity_retry_wait is None
            else self._tenacity_retry_wait,
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        def _do_request() -> httpx.Response:
            headers = self._build_headers() | kwargs.pop("headers", {})
            response = self.httpx_client.request(method, url, headers=headers, **kwargs)
            _raise_errors(response)
            return response

        return _do_request()

    def close(self) -> None:
        self._httpx_client.close()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_httpx_client"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        self.close()

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(
        exc,
        httpx.TransportError | httpx.DecodingError | InternalServerError | JsonParseError,
    )


def _get_error_message(body: dict[str, Any] | None) -> str:
    if body is not None and "detail" in body:
        if isinstance(body["detail"], list):
            if all("msg" in x for x in body["detail"]):
                return "; ".join([str(x["msg"]) for x in body["detail"]])
        elif isinstance(body["detail"], str):
            return body["detail"]
    return "Message is not specified"


def _raise_errors(response: httpx.Response) -> None:
    status_type = response.status_code // 100
    if status_type == 1:
        raise InformationalResponseError(f"Status code: {response.status_code}", response=response)
    if status_type == 3:  # noqa: PLR2004
        raise RedirectResponseError(f"Status code: {response.status_code}", response=response)
    if status_type == 5:  # noqa: PLR2004
        raise InternalServerError(f"Status code: {response.status_code}", response=response)
    if response.status_code in {400, 422}:
        try:
            response_json = response.json()
        except ValueError:
            response_json = None
        raise BadRequestError(_get_error_message(response_json), response=response)
    if response.status_code == 404:  # noqa: PLR2004
        try:
            response_json = response.json()
            raise NotFoundError(_get_error_message(response_json), response=response)
        except ValueError:
            pass
        raise NotFoundError("Resource not found", response=response)
    if status_type == 4:  # noqa: PLR2004
        raise BadRequestError(response.reason_phrase, response=response)
