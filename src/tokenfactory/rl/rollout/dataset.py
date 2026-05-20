from collections.abc import Iterable
from typing import Protocol, TypeVar


TaskSpec_co = TypeVar("TaskSpec_co", covariant=True)


class DatasetProtocol(Protocol[TaskSpec_co]):
    def sample(self, n_samples: int) -> Iterable[TaskSpec_co]: ...
