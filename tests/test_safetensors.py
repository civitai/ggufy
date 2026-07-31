from __future__ import annotations

import json
import struct
from pathlib import Path

from ggufy_api.safetensors import (
    huggingface_resolve_url,
    logical_schema,
    read_local_header,
    read_local_markers,
)

from .conftest import write_safetensors


async def test_scaled_fp8_cluster_is_collapsed_independent_of_header_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cluster.safetensors"
    marker = json.dumps({"format": "float8_e4m3fn"}).encode()
    write_safetensors(
        path,
        [
            ("layer.comfy_quant", "U8", [len(marker)], marker),
            ("layer.weight_scale", "F32", [], struct.pack("<f", 0.25)),
            ("layer.weight", "F8_E4M3", [2, 4], b"\0" * 8),
            ("layer.bias", "F32", [2], b"\0" * 8),
        ],
    )

    header = read_local_header(path, 1024 * 1024)
    markers = await read_local_markers(path, header, max_marker_bytes=4096, max_marker_requests=100)
    schema = logical_schema(header, source=str(path), marker_values=markers)

    assert [(tensor.name, tensor.dtype, tensor.shape) for tensor in schema.tensors] == [
        ("layer.bias", "F32", [2]),
        ("layer.weight", "SCALED_F8_E4M3", [2, 4]),
    ]


def test_file_level_convrot_metadata_is_collapsed(tmp_path: Path) -> None:
    path = tmp_path / "convrot.safetensors"
    quantization_metadata = json.dumps(
        {
            "layers": {
                "layer": {
                    "format": "int8_tensorwise",
                    "per_row": True,
                    "convrot": True,
                    "convrot_groupsize": 256,
                }
            }
        }
    )
    write_safetensors(
        path,
        [
            ("model.layer.weight", "I8", [4, 256], b"\0" * 1024),
            ("model.layer.weight_scale", "F32", [4, 1], b"\0" * 16),
        ],
        metadata={"_quantization_metadata": quantization_metadata},
    )

    schema = logical_schema(read_local_header(path, 1024 * 1024), source=str(path))
    assert schema.tensors[0].dtype == "INT8_CONVROT"
    assert schema.tensors[0].shape == [4, 256]


async def test_nvfp4_cluster_restores_logical_column_count(tmp_path: Path) -> None:
    path = tmp_path / "nvfp4.safetensors"
    marker = json.dumps({"format": "nvfp4"}).encode()
    write_safetensors(
        path,
        [
            ("layer.weight", "U8", [128, 32], b"\0" * (128 * 32)),
            ("layer.weight_scale", "F8_E4M3", [128, 4], b"\0" * (128 * 4)),
            ("layer.weight_scale_2", "F32", [], struct.pack("<f", 1)),
            ("layer.comfy_quant", "U8", [len(marker)], marker),
        ],
    )

    header = read_local_header(path, 1024 * 1024)
    markers = await read_local_markers(path, header, max_marker_bytes=4096, max_marker_requests=100)
    schema = logical_schema(header, source=str(path), marker_values=markers)

    assert [(tensor.name, tensor.dtype, tensor.shape) for tensor in schema.tensors] == [
        ("layer.weight", "NVFP4", [128, 64])
    ]


def test_huggingface_url_encodes_revision() -> None:
    assert huggingface_resolve_url(
        "Comfy-Org/example", "diffusion_models/model.safetensors", "refs/pr/7"
    ) == (
        "https://huggingface.co/Comfy-Org/example/resolve/"
        "refs%2Fpr%2F7/diffusion_models/model.safetensors"
    )
