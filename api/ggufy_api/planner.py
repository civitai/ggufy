from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import (
    ModelFormat,
    PlanResponse,
    QuantizationPolicy,
    RuleMatch,
    SchemaMatch,
    TensorDefinition,
    TensorRule,
    TensorSchema,
    UnmatchedPolicy,
)


class PlanError(ValueError):
    """Raised when a tensor policy cannot be resolved safely."""


SAFETENSORS_OUTPUT_TYPES = {
    "F8_E4M3",
    "F8_E5M2",
    "SCALED_F8_E4M3",
    "MXFP4",
    "MXFP8_E4M3",
    "NVFP4",
    "INT8",
    "INT8_CONVROT",
    "INT4_CONVROT",
    "INT4_CONVROT_SR",
    "BF16",
    "F16",
    "F32",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
}
SAFETENSORS_CLUSTER_TYPES = {
    "SCALED_F8_E4M3",
    "MXFP4",
    "MXFP8_E4M3",
    "NVFP4",
    "INT8",
    "INT8_CONVROT",
    "INT4_CONVROT",
    "INT4_CONVROT_SR",
}

GGUF_REQUEST_TO_TEMPLATE = {
    "F32": "f32",
    "F16": "f16",
    "BF16": "bf16",
    "MXFP4": "mxfp4",
    "Q8_0": "q8_0",
    "Q5_0": "q5_0",
    "Q5_1": "q5_1",
    "Q4_0": "q4_0",
    "Q4_1": "q4_1",
    "Q6_K": "q6_k",
    "Q5_K": "q5_k",
    "Q4_K": "q4_k",
    "Q3_K": "q3_k",
    "Q2_K": "q2_k",
}
GGUF_QUANTIZED_TYPES = {
    "q8_0",
    "q5_0",
    "q5_1",
    "q4_0",
    "q4_1",
    "q6_k",
    "q5_k",
    "q4_k",
    "q3_k",
    "q2_k",
    "mxfp4",
}
GGUF_DIRECT_TYPES = {"f32", "f16", "bf16", "f64", "i8", "i16", "i32", "i64"}
GGUF_OUTPUT_TYPES = GGUF_QUANTIZED_TYPES | GGUF_DIRECT_TYPES
GGUF_BLOCK_SIZES = {
    "q8_0": 32,
    "q5_0": 32,
    "q5_1": 32,
    "q4_0": 32,
    "q4_1": 32,
    "mxfp4": 32,
    "q6_k": 256,
    "q5_k": 256,
    "q4_k": 256,
    "q3_k": 256,
    "q2_k": 256,
}
SAFETENSORS_TO_GGUF = {
    "F32": "f32",
    "F16": "f16",
    "BF16": "bf16",
    "F64": "f64",
    "I8": "i8",
    "I16": "i16",
    "I32": "i32",
    "I64": "i64",
}

_EMBEDDING_SUFFIXES = (
    ".embed.weight",
    ".embed_tokens.weight",
    ".token_embedding.weight",
    ".token_embed.weight",
    ".word_embeddings.weight",
    ".tok_embeddings.weight",
    ".wte.weight",
)


@dataclass(frozen=True, slots=True)
class ResolvedPlan:
    response: PlanResponse
    template: dict[str, Any]


def _suffix_match(left: str, right: str) -> bool:
    return left == right or left.endswith(f".{right}") or right.endswith(f".{left}")


def _reference_match(
    source: TensorDefinition,
    reference: TensorSchema,
    reference_by_name: dict[str, TensorDefinition],
    mode: SchemaMatch,
) -> TensorDefinition | None:
    if exact := reference_by_name.get(source.name):
        return exact
    if mode == SchemaMatch.EXACT:
        return None
    candidates = [tensor for tensor in reference.tensors if _suffix_match(source.name, tensor.name)]
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates[:5])
        raise PlanError(f"schema match for {source.name!r} is ambiguous: {names}")
    return candidates[0] if candidates else None


def _rule_matches(name: str, rule: TensorRule) -> bool:
    if rule.match == RuleMatch.EXACT:
        return name == rule.pattern
    if rule.match == RuleMatch.GLOB:
        return fnmatch.fnmatchcase(name, rule.pattern)
    try:
        return re.search(rule.pattern, name) is not None
    except re.error as exc:
        raise PlanError(f"invalid regex {rule.pattern!r}: {exc}") from exc


def _elements(tensor: TensorDefinition) -> int:
    result = 1
    for dimension in tensor.shape:
        result *= dimension
    return result


def _preserved_type(dtype: str, output_format: ModelFormat) -> str:
    if output_format == ModelFormat.SAFETENSORS:
        if dtype not in SAFETENSORS_OUTPUT_TYPES:
            raise PlanError(f"cannot preserve source dtype {dtype!r} in safetensors output")
        return dtype

    if target := SAFETENSORS_TO_GGUF.get(dtype):
        return target
    raise PlanError(
        f"cannot preserve source dtype {dtype!r} in GGUF output; choose an explicit "
        "GGUF-compatible target type"
    )


def _normalize_target(target: str, output_format: ModelFormat) -> str:
    if output_format == ModelFormat.SAFETENSORS:
        if target not in SAFETENSORS_OUTPUT_TYPES:
            raise PlanError(f"unsupported safetensors target type {target!r}")
        return target

    normalized = GGUF_REQUEST_TO_TEMPLATE.get(target, target.lower())
    if normalized not in GGUF_OUTPUT_TYPES:
        raise PlanError(
            f"unsupported CPU GGUF target type {target!r}; IQ, TQ, NVFP4, Q8_K, "
            "and Q8_1 output are not implemented by this GGUFy build"
        )
    return normalized


def _validate_target(
    tensor: TensorDefinition,
    target: str,
    output_format: ModelFormat,
) -> None:
    if output_format == ModelFormat.GGUF:
        if target not in GGUF_OUTPUT_TYPES:
            raise PlanError(f"unsupported CPU GGUF target type {target!r} for {tensor.name!r}")
        if target in {"f64", "i8", "i16", "i32", "i64"}:
            if SAFETENSORS_TO_GGUF.get(tensor.dtype) != target:
                raise PlanError(
                    f"GGUF target {target!r} can only directly preserve an equivalent "
                    f"source tensor; {tensor.name!r} has dtype {tensor.dtype!r}"
                )
        if block_size := GGUF_BLOCK_SIZES.get(target):
            elements = _elements(tensor)
            if not elements:
                raise PlanError(f"{target} cannot quantize empty tensor {tensor.name!r}")
            if elements % block_size:
                raise PlanError(
                    f"{target} requires an element count divisible by {block_size}; "
                    f"{tensor.name!r} has {elements} elements"
                )
        return

    if target not in SAFETENSORS_OUTPUT_TYPES:
        raise PlanError(f"unsupported safetensors target type {target!r} for {tensor.name!r}")
    if target in SAFETENSORS_CLUSTER_TYPES and len(tensor.shape) < 2:
        raise PlanError(
            f"{target} requires a tensor with at least two dimensions; "
            f"{tensor.name!r} has shape {tensor.shape}"
        )
    if target in SAFETENSORS_CLUSTER_TYPES and any(dimension == 0 for dimension in tensor.shape):
        raise PlanError(f"{target} cannot quantize empty tensor {tensor.name!r}")
    if target in {"INT8", "INT8_CONVROT", "INT4_CONVROT", "INT4_CONVROT_SR"}:
        if len(tensor.shape) != 2:
            raise PlanError(
                f"{target} requires a two-dimensional weight; "
                f"{tensor.name!r} has shape {tensor.shape}"
            )
        if tensor.shape[-1] < 1:
            raise PlanError(f"{target} requires a non-empty final dimension for {tensor.name!r}")
    if target in {"MXFP4", "MXFP8_E4M3"} and tensor.shape[-1] % 32:
        raise PlanError(
            f"{target} requires the final dimension to be divisible by 32; "
            f"{tensor.name!r} has {tensor.shape[-1]} columns"
        )
    if target == "NVFP4":
        columns = tensor.shape[-1]
        rows = _elements(tensor) // columns if columns else 0
        if columns % 64 or rows % 128:
            raise PlanError(
                "NVFP4 requires the final dimension to be divisible by 64 and "
                f"flattened rows to be divisible by 128; {tensor.name!r} has "
                f"{rows} rows and {columns} columns"
            )
    if target in {"INT8_CONVROT", "INT4_CONVROT", "INT4_CONVROT_SR"}:
        columns = tensor.shape[-1]
        if columns % 256:
            raise PlanError(
                f"{target} requires the final dimension to be divisible by 256; "
                f"{tensor.name!r} has {columns} columns"
            )
    if target in {"INT4_CONVROT", "INT4_CONVROT_SR"} and tensor.shape[-1] % 2:
        raise PlanError(f"{target} requires an even final dimension for {tensor.name!r}")


def _default_target_eligible(
    tensor: TensorDefinition,
    target: str,
    output_format: ModelFormat,
) -> bool:
    cluster_types = (
        GGUF_QUANTIZED_TYPES if output_format == ModelFormat.GGUF else SAFETENSORS_CLUSTER_TYPES
    )
    if target not in cluster_types:
        return True
    if not tensor.name.endswith(".weight") or tensor.name.endswith(_EMBEDDING_SUFFIXES):
        return False
    elements = _elements(tensor)
    if elements < 256 * 256:
        return False

    if output_format == ModelFormat.GGUF:
        return len(tensor.shape) > 1 and elements % GGUF_BLOCK_SIZES[target] == 0

    columns = tensor.shape[-1] if tensor.shape else 0
    rows = elements // columns if columns else 0
    if target == "SCALED_F8_E4M3":
        return True
    if target in {"MXFP4", "MXFP8_E4M3"}:
        return columns >= 32 and columns % 32 == 0
    if target == "NVFP4":
        return columns >= 64 and columns % 64 == 0 and rows % 128 == 0
    if target == "INT8":
        return len(tensor.shape) == 2 and columns >= 1
    return len(tensor.shape) == 2 and columns % 256 == 0


def _validate_gguf_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    for key, value in metadata.items():
        if not key:
            raise PlanError("GGUF metadata keys must not be empty")
        if value is None or isinstance(value, dict):
            raise PlanError(f"GGUF metadata {key!r} cannot be represented by GGUFy's template")
        if isinstance(value, float) and (not math.isfinite(value) or abs(value) > 3.4028235e38):
            raise PlanError(f"GGUF metadata {key!r} is outside the finite float32 range")
        if isinstance(value, int) and not isinstance(value, bool):
            if value < -(2**63) or value > 2**63 - 1:
                raise PlanError(f"GGUF metadata {key!r} is outside the signed 64-bit range")
        if isinstance(value, list):
            if not value:
                raise PlanError(
                    f"GGUF metadata array {key!r} is empty and loses its element type in JSON"
                )
            first_type = type(value[0])
            if first_type not in (bool, int, float, str) or any(
                type(item) is not first_type for item in value
            ):
                raise PlanError(f"GGUF metadata array {key!r} must have one scalar type")
            if first_type is int and any(item < 0 or item > 2**31 - 1 for item in value):
                raise PlanError(
                    f"GGUF metadata integer array {key!r} is outside GGUFy's int32 range"
                )
            if first_type is float and any(
                not math.isfinite(item) or abs(item) > 3.4028235e38 for item in value
            ):
                raise PlanError(
                    f"GGUF metadata float array {key!r} is outside the finite float32 range"
                )
        elif not isinstance(value, (bool, int, float, str)):
            raise PlanError(f"GGUF metadata {key!r} has an unsupported JSON value")
    return dict(metadata)


def resolve_plan(source: TensorSchema, policy: QuantizationPolicy) -> ResolvedPlan:
    if source.format != ModelFormat.SAFETENSORS:
        raise PlanError("conversion inputs must use safetensors format")
    if (
        policy.reference_schema is not None
        and policy.reference_schema.format != policy.output_format
    ):
        raise PlanError(
            f"reference schema format {policy.reference_schema.format.value!r} does not match "
            f"output format {policy.output_format.value!r}"
        )

    resolved: list[TensorDefinition] = []
    matched_by_schema = 0
    matched_by_rule = 0
    preserved = 0
    reference_by_name = (
        {tensor.name: tensor for tensor in policy.reference_schema.tensors}
        if policy.reference_schema
        else {}
    )

    for source_tensor in source.tensors:
        target: str | None = None
        selection: str | None = None

        if policy.reference_schema is not None:
            reference_tensor = _reference_match(
                source_tensor,
                policy.reference_schema,
                reference_by_name,
                policy.schema_match,
            )
            if reference_tensor is not None:
                if reference_tensor.shape != source_tensor.shape:
                    raise PlanError(
                        f"shape mismatch for {source_tensor.name!r}: source "
                        f"{source_tensor.shape}, schema {reference_tensor.shape}"
                    )
                target = _normalize_target(reference_tensor.dtype, policy.output_format)
                selection = "schema"
                matched_by_schema += 1

        if target is None and policy.default_type is not None:
            target = (
                _preserved_type(source_tensor.dtype, policy.output_format)
                if policy.default_type == "PRESERVE"
                else _normalize_target(policy.default_type.value, policy.output_format)
            )
            selection = "default"

        for rule in policy.rules:
            if _rule_matches(source_tensor.name, rule):
                target = (
                    _preserved_type(source_tensor.dtype, policy.output_format)
                    if rule.target_type == "PRESERVE"
                    else _normalize_target(rule.target_type.value, policy.output_format)
                )
                selection = "rule"
                matched_by_rule += 1
                break

        if target is None:
            if policy.unmatched == UnmatchedPolicy.ERROR:
                raise PlanError(f"no tensor policy matched {source_tensor.name!r}")
            target = _preserved_type(source_tensor.dtype, policy.output_format)

        if selection == "default" and not _default_target_eligible(
            source_tensor, target, policy.output_format
        ):
            target = _preserved_type(source_tensor.dtype, policy.output_format)
        preserved_target = (
            source_tensor.dtype
            if policy.output_format == ModelFormat.SAFETENSORS
            else SAFETENSORS_TO_GGUF.get(source_tensor.dtype)
        )
        if target == preserved_target:
            preserved += 1
        _validate_target(source_tensor, target, policy.output_format)
        resolved.append(
            TensorDefinition(
                name=source_tensor.name,
                shape=source_tensor.shape,
                dtype=target,
            )
        )

    if not resolved:
        raise PlanError("input schema contains no tensors")

    type_counts = dict(sorted(Counter(tensor.dtype for tensor in resolved).items()))
    response = PlanResponse(
        source=source.source or "unknown",
        format=policy.output_format,
        tensors=resolved,
        type_counts=type_counts,
        matched_by_schema=matched_by_schema,
        matched_by_rule=matched_by_rule,
        preserved=preserved,
    )
    template: dict[str, Any] = {
        "tensors": {
            tensor.name: {
                # GGUFy templates use GGUF/innermost-first dimension order.
                "shape": list(reversed(tensor.shape)),
                "type": tensor.dtype,
            }
            for tensor in resolved
        }
    }
    if policy.output_format == ModelFormat.GGUF and policy.reference_schema is not None:
        template["metadata"] = _validate_gguf_metadata(policy.reference_schema.metadata)
    return ResolvedPlan(response=response, template=template)
