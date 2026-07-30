from __future__ import annotations

import pytest
from ggufy_api.models import PlanRequest, TensorDefinition, TensorSchema
from ggufy_api.planner import PlanError, resolve_plan

SOURCE = TensorSchema(
    source="/data/input/model.safetensors",
    tensors=[
        TensorDefinition(name="model.block.attn.weight", shape=[4, 256], dtype="F32"),
        TensorDefinition(name="model.block.norm.weight", shape=[4], dtype="F32"),
    ],
)


def test_rules_compile_to_exact_mixed_template() -> None:
    request = PlanRequest.model_validate(
        {
            "input_path": "model.safetensors",
            "default_type": "SCALED_F8_E4M3",
            "rules": [
                {
                    "pattern": "*.attn.weight",
                    "match": "glob",
                    "target_type": "INT8_CONVROT",
                },
                {
                    "pattern": "*.norm.weight",
                    "match": "glob",
                    "target_type": "PRESERVE",
                },
            ],
        }
    )

    plan = resolve_plan(SOURCE, request)

    assert plan.response.type_counts == {"F32": 1, "INT8_CONVROT": 1}
    assert plan.template["tensors"]["model.block.attn.weight"] == {
        "shape": [256, 4],
        "type": "INT8_CONVROT",
    }


def test_reference_schema_requires_exact_shape() -> None:
    reference = TensorSchema(
        tensors=[
            TensorDefinition(
                name="block.attn.weight",
                shape=[8, 128],
                dtype="INT8_CONVROT",
            )
        ]
    )
    request = PlanRequest.model_validate(
        {
            "input_path": "model.safetensors",
            "schema": reference.model_dump(),
            "schema_match": "suffix",
        }
    )

    with pytest.raises(PlanError, match="shape mismatch"):
        resolve_plan(SOURCE, request)


def test_convrot_requires_256_column_groups() -> None:
    source = TensorSchema(tensors=[TensorDefinition(name="weight", shape=[4, 255], dtype="F32")])
    request = PlanRequest.model_validate(
        {
            "input_path": "model.safetensors",
            "default_type": "PRESERVE",
            "rules": [
                {
                    "pattern": "weight",
                    "match": "exact",
                    "target_type": "INT8_CONVROT",
                }
            ],
        }
    )

    with pytest.raises(PlanError, match="divisible by 256"):
        resolve_plan(source, request)


def test_default_cluster_type_preserves_small_and_non_weight_tensors() -> None:
    source = TensorSchema(
        tensors=[
            TensorDefinition(name="small.weight", shape=[4, 256], dtype="F32"),
            TensorDefinition(name="large.bias", shape=[65536], dtype="F32"),
        ]
    )
    request = PlanRequest.model_validate(
        {"input_path": "model.safetensors", "default_type": "SCALED_F8_E4M3"}
    )

    plan = resolve_plan(source, request)
    assert {tensor.name: tensor.dtype for tensor in plan.response.tensors} == {
        "large.bias": "F32",
        "small.weight": "F32",
    }


@pytest.mark.parametrize(
    ("target_type", "shape", "message"),
    [
        ("MXFP4", [128, 33], "divisible by 32"),
        ("MXFP8_E4M3", [128, 63], "divisible by 32"),
        ("NVFP4", [127, 64], "flattened rows"),
    ],
)
def test_explicit_cluster_rules_validate_storage_shape(
    target_type: str, shape: list[int], message: str
) -> None:
    source = TensorSchema(tensors=[TensorDefinition(name="model.weight", shape=shape, dtype="F32")])
    request = PlanRequest.model_validate(
        {
            "input_path": "model.safetensors",
            "rules": [
                {
                    "pattern": "model.weight",
                    "match": "exact",
                    "target_type": target_type,
                }
            ],
        }
    )

    with pytest.raises(PlanError, match=message):
        resolve_plan(source, request)
