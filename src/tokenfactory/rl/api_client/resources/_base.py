from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from tokenfactory.rl.api_client._client import TokenFactory


class BaseResource(ABC):  # noqa: B024
    PATH_SEGMENT = ""

    def __init__(
        self,
        client: TokenFactory,
        parent: BaseResource | None = None,
    ) -> None:
        self._client = client
        self._parent = parent

    def _endpoint_segments(self) -> list[str]:
        if self._parent is None:
            return [self._base_url.rstrip("/"), self.PATH_SEGMENT]
        segments = self._parent._endpoint_segments()
        segments.append(self.PATH_SEGMENT)
        return segments

    def endpoint(self, **kwargs) -> str:
        return "/".join(self._endpoint_segments()).format(**kwargs)

    @property
    def _base_url(self) -> str:
        return self._client.base_url
