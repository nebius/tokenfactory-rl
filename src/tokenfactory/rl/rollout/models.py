from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar
from uuid import uuid4


TaskSpec = TypeVar("TaskSpec")


class BaseID(ABC, str):
    __slots__ = ()
    prefix: ClassVar[str | None] = None

    @classmethod
    def generate(cls) -> "BaseID":
        if cls.prefix is None:
            raise NotImplementedError("Prefix must be set for ID generation.")
        return cls(f"{cls.prefix}-{uuid4().hex}")


class TaskID(BaseID):
    prefix: ClassVar[str | None] = "task"


class SampleID(BaseID):
    prefix: ClassVar[str | None] = "sample"


@dataclass(kw_only=True)
class Task(Generic[TaskSpec]):
    id: TaskID = field(default_factory=TaskID.generate)
    spec: TaskSpec
    starting_inference_version: int


@dataclass(kw_only=True)
class Sample:
    id: SampleID = field(default_factory=SampleID.generate)
    token_ids: list[int] = field(repr=False)
    logprobs: list[float] = field(repr=False)
    mask: list[int] = field(repr=False)
    normalized_reward: float
    metadata: dict[str, str] = field(default_factory=dict)
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleGroup:
    is_rejected: bool = field(default=False)
    samples: list[Sample] = field(repr=False, default_factory=list)


@dataclass
class JobStatusUpdate:
    inference_version: int


@dataclass
class RolloutException:
    exception: Exception
    traceback: str


@dataclass
class RolloutResult(Generic[TaskSpec]):
    task: Task[TaskSpec]
    sample_group: SampleGroup
