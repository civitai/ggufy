from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class OutputType(StrEnum):
    F8_E4M3 = "F8_E4M3"
    F8_E5M2 = "F8_E5M2"
    SCALED_F8_E4M3 = "SCALED_F8_E4M3"
    MXFP4 = "MXFP4"
    MXFP8_E4M3 = "MXFP8_E4M3"
    NVFP4 = "NVFP4"
    INT8 = "INT8"
    INT8_CONVROT = "INT8_CONVROT"
    INT4_CONVROT = "INT4_CONVROT"
    INT4_CONVROT_SR = "INT4_CONVROT_SR"
    BF16 = "BF16"
    F16 = "F16"
    F32 = "F32"
    Q8_0 = "Q8_0"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q6_K = "Q6_K"
    Q5_K = "Q5_K"
    Q4_K = "Q4_K"
    Q3_K = "Q3_K"
    Q2_K = "Q2_K"


class ModelFormat(StrEnum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"


class RuleMatch(StrEnum):
    EXACT = "exact"
    GLOB = "glob"
    REGEX = "regex"


class SchemaMatch(StrEnum):
    EXACT = "exact"
    SUFFIX = "suffix"


class UnmatchedPolicy(StrEnum):
    PRESERVE = "preserve"
    ERROR = "error"


class TensorDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmpty
    shape: list[Annotated[int, Field(ge=0)]]
    dtype: NonEmpty


class TensorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    format: ModelFormat = ModelFormat.SAFETENSORS
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    tensors: list[TensorDefinition]

    @model_validator(mode="after")
    def unique_tensor_names(self) -> TensorSchema:
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("tensor schema contains duplicate names")
        return self


class TensorRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: NonEmpty
    target_type: OutputType | Literal["PRESERVE"]
    match: RuleMatch = RuleMatch.GLOB


class QuantizationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    output_format: ModelFormat = ModelFormat.SAFETENSORS
    default_type: OutputType | Literal["PRESERVE"] | None = None
    reference_schema: TensorSchema | None = Field(
        default=None,
        validation_alias="schema",
        serialization_alias="schema",
    )
    rules: list[TensorRule] = Field(default_factory=list)
    unmatched: UnmatchedPolicy = UnmatchedPolicy.PRESERVE
    schema_match: SchemaMatch = SchemaMatch.EXACT

    @model_validator(mode="after")
    def has_policy(self) -> QuantizationPolicy:
        if self.default_type is None and self.reference_schema is None and not self.rules:
            raise ValueError("provide at least one of default_type, schema, or rules")
        return self


class PlanRequest(QuantizationPolicy):
    input_path: NonEmpty


class ConversionRequest(QuantizationPolicy):
    input_path: NonEmpty
    output_path: str | None = None
    threads: Annotated[int, Field(ge=1)] | None = None
    overwrite: bool = False
    allow_upscale: bool = False


class LocalSchemaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: NonEmpty
    read_markers: bool = True


class HuggingFaceSchemaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[^/\s]+/[^/\s]+$")]
    filename: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=6,
            pattern=r"^[^\\]+\.(?:safetensors|gguf)$",
        ),
    ]
    revision: NonEmpty = "main"
    read_markers: bool = True

    @model_validator(mode="after")
    def safe_components(self) -> HuggingFaceSchemaRequest:
        if ".." in self.filename.split("/") or self.filename.startswith("/"):
            raise ValueError("filename must be a repository-relative path")
        return self


class PlanResponse(BaseModel):
    source: str
    format: ModelFormat
    tensors: list[TensorDefinition]
    type_counts: dict[str, int]
    matched_by_schema: int
    matched_by_rule: int
    preserved: int


class ConversionResponse(BaseModel):
    output_path: str
    format: ModelFormat
    output_size: int
    tensor_count: int
    type_counts: dict[str, int]
    duration_seconds: float
    log_tail: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ggufy_binary: str
    max_concurrent_jobs: int
    cpu_only: Literal[True] = True
