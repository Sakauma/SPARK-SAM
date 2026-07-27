from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Protocol

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..data.sample import Sample


class InferenceMode(str, Enum):
    PROMPTED = "prompted"
    BOX = "box"
    POINT = "point"
    BOX_POINT = "box+point"
    SINGLE_MASK = "single_mask"
    MULTI_MASK = "multi_mask"
    NO_PROMPT_AUTO_MASK = "no_prompt_auto_mask"


class Track(str, Enum):
    IMAGE_PROMPTED = "image_prompted_segmentation"
    AUTO_MASK = "automatic_mask_generation"


class PromptType(str, Enum):
    BOX = "box"
    POINT = "point"
    BOX_POINT = "box+point"
    NONE = "none"


class PromptSource(str, Enum):
    GT = "gt"
    SYNTHESIZED = "synthesized"
    NONE = "none"


@dataclass(frozen=True)
class PromptPolicy:
    name: str
    prompt_type: PromptType
    prompt_source: PromptSource
    prompt_budget: int
    refresh_interval: Optional[int] = None
    multi_mask: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCapabilities:
    supports_box_prompt: bool = True
    supports_point_prompt: bool = True
    supports_box_point_prompt: bool = True
    supports_single_mask: bool = True
    supports_multi_mask: bool = True
    supports_auto_mask: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRecord:
    stage: str
    artifact_dir: str
    artifact_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "artifact_dir": self.artifact_dir,
            "artifact_name": self.artifact_name,
            "metadata": self.metadata,
        }


class BenchmarkMethodProtocol(Protocol):
    inference_mode: InferenceMode

    def predict_sample(self, sample: "Sample") -> Dict[str, Any]:
        ...

    def predict_samples(self, samples: List["Sample"]) -> Dict[str, Dict[str, Any]]:
        ...


class DatasetAdapterProtocol(Protocol):
    adapter_name: str
    notes: str

    def can_handle(self, config: "AppConfig") -> bool:
        ...

    def load(self, config: "AppConfig") -> Any:
        ...

    def iter_samples(self, config: "AppConfig", *, shard_id: int = 0, num_shards: int = 1) -> Iterable["Sample"]:
        ...
