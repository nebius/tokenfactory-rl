from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar


if TYPE_CHECKING:
    from tokenfactory.rl.api_client._client import TokenFactory


class BaseResource(ABC):  # noqa: B024
    PATH_SEGMENT: ClassVar[str] = "/"

    def __init__(
        self,
        client: TokenFactory,
        parent: BaseResource | None = None,
    ) -> None:
        self._client = client
        self._parent = parent

    def _endpoint_segments(self, last_segment: str | None = None) -> list[str]:
        last_segment = self.PATH_SEGMENT if last_segment is None else last_segment
        if self._parent is None:
            return [last_segment]
        segments = self._parent._endpoint_segments()
        segments.append(last_segment)
        return segments

    def _endpoint_template(self, last_segment: str | None = None, base_url: str | None = None) -> str:
        return "/".join([
            (self._base_url if base_url is None else base_url).rstrip("/"),
            *self._endpoint_segments(last_segment=last_segment),
        ])

    @property
    def _base_url(self) -> str:
        return self._client.base_url


class BaseBareResource(BaseResource, ABC):
    def endpoint(self) -> str:
        return self._endpoint_template()
