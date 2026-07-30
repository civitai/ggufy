from __future__ import annotations

import fnmatch
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import (
    OutputType,
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


SAFE_TEMPLATE_TYPES = {
    *(member.value for member in OutputType),
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
CLUSTER_TYPES = {
    "SCALED_F8_E4M3",
    "MXFP4",
    "MXFP8_E4M3",
    "NVFP4",
    "INT8",
    "INT8_CONVROT",
    "INT4_CONVROT",
    "INT4_CONVROT_SR",
}


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


def _preserved_type(dtype: str) -> str:
    if dtype not in SAFE_TEMPLATE_TYPES:
        raise PlanError(f"cannot preserve unsupported source dtype {dtype!r} in safetensors output")
    return dtype


def _validate_target(tensor: TensorDefinition, target: str) -> None:
    if target not in SAFE_TEMPLATE_TYPES:
        raise PlanError(f"unsupported safetensors target type {target!r} for {tensor.name!r}")
    if target in CLUSTER_TYPES and len(tensor.shape) < 2:
        raise PlanError(
            f"{target} requires a tensor with at least two dimensions; "
            f"{tensor.name!r} has shape {tensor.shape}"
        )
    if target in CLUSTER_TYPES and any(dimension == 0 for dimension in tensor.shape):
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
        rows = 1
        for dimension in tensor.shape[:-1]:
            rows *= dimension
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


def _default_cluster_eligible(tensor: TensorDefinition, target: str) -> bool:
    if target not in CLUSTER_TYPES:
        return True
    if not tensor.name.endswith(".weight"):
        return False
    embedding_suffixes = (
        ".embed.weight",
        ".embed_tokens.weight",
        ".token_embedding.weight",
        ".token_embed.weight",
        ".word_embeddings.weight",
        ".tok_embeddings.weight",
        ".wte.weight",
    )
    if tensor.name.endswith(embedding_suffixes):
        return False
    elements = 1
    for dimension in tensor.shape:
        elements *= dimension
    if elements < 256 * 256 or len(tensor.shape) < 1:
        return False
    columns = tensor.shape[-1]
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


def resolve_plan(source: TensorSchema, policy: QuantizationPolicy) -> ResolvedPlan:
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
        reference_tensor: TensorDefinition | None = None

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
                target = reference_tensor.dtype
                selection = "schema"
                matched_by_schema += 1

        if target is None and policy.default_type is not None:
            target = (
                _preserved_type(source_tensor.dtype)
                if policy.default_type == "PRESERVE"
                else policy.default_type.value
            )
            selection = "default"

        for rule in policy.rules:
            if _rule_matches(source_tensor.name, rule):
                target = (
                    _preserved_type(source_tensor.dtype)
                    if rule.target_type == "PRESERVE"
                    else rule.target_type.value
                )
                selection = "rule"
                matched_by_rule += 1
                break

        if target is None:
            if policy.unmatched == UnmatchedPolicy.ERROR:
                raise PlanError(f"no tensor policy matched {source_tensor.name!r}")
            target = _preserved_type(source_tensor.dtype)

        if selection == "default" and not _default_cluster_eligible(source_tensor, target):
            target = _preserved_type(source_tensor.dtype)
        if target == source_tensor.dtype:
            preserved += 1
        _validate_target(source_tensor, target)
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
        tensors=resolved,
        type_counts=type_counts,
        matched_by_schema=matched_by_schema,
        matched_by_rule=matched_by_rule,
        preserved=preserved,
    )
    template = {
        "tensors": {
            tensor.name: {
                # GGUFy templates use GGUF/innermost-first dimension order.
                "shape": list(reversed(tensor.shape)),
                "type": tensor.dtype,
            }
            for tensor in resolved
        }
    }
    return ResolvedPlan(response=response, template=template)
