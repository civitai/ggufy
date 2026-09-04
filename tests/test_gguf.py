from __future__ import annotations

from pathlib import Path

import pytest
from ggufy_api.gguf import GgufError, read_local_gguf_schema
from ggufy_api.models import PlanRequest, TensorDefinition, TensorSchema
from ggufy_api.planner import PlanError, resolve_plan

from .conftest import write_gguf


def test_reads_typed_metadata_and_natural_tensor_shapes(tmp_path: Path) -> None:
    path = tmp_path / "reference.gguf"
    write_gguf(
        path,
        [
            ("block.weight", "q4_k", [128, 256]),
            ("block.bias", "f32", [128]),
        ],
        metadata={
            "general.architecture": "flux",
            "general.quantization_version": 2,
            "model.enabled": True,
            "model.rates": [0.25, 0.5],
        },
    )

    schema = read_local_gguf_schema(path, 1024 * 1024)

    assert schema.format == "gguf"
    assert schema.metadata == {
        "general.architecture": "flux",
        "general.quantization_version": 2,
        "model.enabled": True,
        "model.rates": pytest.approx([0.25, 0.5]),
    }
    assert [(tensor.name, tensor.dtype, tensor.shape) for tensor in schema.tensors] == [
        ("block.bias", "f32", [128]),
        ("block.weight", "q4_k", [128, 256]),
    ]


def test_large_header_is_read_incrementally(tmp_path: Path) -> None:
    path = tmp_path / "large-header.gguf"
    write_gguf(
        path,
        [("weight", "q8_0", [256])],
        metadata={"model.description": "x" * (70 * 1024)},
    )

    schema = read_local_gguf_schema(path, 1024 * 1024)

    assert schema.tensors[0].dtype == "q8_0"
    assert len(schema.metadata["model.description"]) == 70 * 1024


def test_truncated_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truncated.gguf"
    path.write_bytes(b"GGUF\x03\x00")

    with pytest.raises(GgufError, match="truncated"):
        read_local_gguf_schema(path, 1024 * 1024)


def test_gguf_reference_schema_copies_types_and_metadata() -> None:
    source = TensorSchema(
        tensors=[
            TensorDefinition(name="block.weight", shape=[256, 256], dtype="F32"),
            TensorDefinition(name="block.bias", shape=[256], dtype="F32"),
        ]
    )
    reference = TensorSchema(
        format="gguf",
        metadata={"general.architecture": "flux", "general.file_type": 15},
        tensors=[
            TensorDefinition(name="block.weight", shape=[256, 256], dtype="q4_k"),
            TensorDefinition(name="block.bias", shape=[256], dtype="f32"),
        ],
    )
    request = PlanRequest(
        input_path="model.safetensors",
        output_format="gguf",
        schema=reference,
        unmatched="error",
    )

    plan = resolve_plan(source, request)

    assert plan.response.format == "gguf"
    assert plan.response.type_counts == {"f32": 1, "q4_k": 1}
    assert plan.template["metadata"] == reference.metadata
    assert plan.template["tensors"]["block.weight"] == {
        "shape": [256, 256],
        "type": "q4_k",
    }


def test_gguf_default_type_normalizes_and_preserves_small_tensors() -> None:
    source = TensorSchema(
        tensors=[
            TensorDefinition(name="large.weight", shape=[256, 256], dtype="BF16"),
            TensorDefinition(name="small.weight", shape=[1, 256], dtype="BF16"),
        ]
    )
    request = PlanRequest(
        input_path="model.safetensors",
        output_format="gguf",
        default_type="Q4_K",
    )

    plan = resolve_plan(source, request)

    assert {tensor.name: tensor.dtype for tensor in plan.response.tensors} == {
        "large.weight": "q4_k",
        "small.weight": "bf16",
    }


def test_unsupported_gguf_schema_type_fails_before_conversion() -> None:
    source = TensorSchema(tensors=[TensorDefinition(name="weight", shape=[256, 256], dtype="F32")])
    reference = TensorSchema(
        format="gguf",
        tensors=[TensorDefinition(name="weight", shape=[256, 256], dtype="iq2_xxs")],
    )
    request = PlanRequest(
        input_path="model.safetensors",
        output_format="gguf",
        schema=reference,
    )

    with pytest.raises(PlanError, match="IQ, TQ, NVFP4"):
        resolve_plan(source, request)


def test_reference_format_must_match_output_format() -> None:
    source = TensorSchema(tensors=[TensorDefinition(name="weight", shape=[256], dtype="F32")])
    reference = TensorSchema(
        format="gguf",
        tensors=[TensorDefinition(name="weight", shape=[256], dtype="q8_0")],
    )
    request = PlanRequest(input_path="model.safetensors", schema=reference)

    with pytest.raises(PlanError, match="does not match output format"):
        resolve_plan(source, request)
