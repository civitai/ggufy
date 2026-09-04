from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from ggufy_api.config import Settings
from ggufy_api.gguf import GGML_TYPES


def write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    if metadata:
        header["__metadata__"] = metadata
    for name, dtype, shape, value in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(value)],
        }
        payload.extend(value)
        offset += len(value)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value: object) -> tuple[int, bytes]:
    if type(value) is bool:
        return 7, struct.pack("<B", value)
    if type(value) is int:
        return 4, struct.pack("<I", value)
    if type(value) is float:
        return 6, struct.pack("<f", value)
    if type(value) is str:
        return 8, _gguf_string(value)
    if type(value) is list and value:
        element_type, _ = _gguf_value(value[0])
        encoded = bytearray(struct.pack("<IQ", element_type, len(value)))
        for item in value:
            item_type, payload = _gguf_value(item)
            assert item_type == element_type
            encoded.extend(payload)
        return 9, bytes(encoded)
    raise ValueError(f"unsupported test GGUF metadata value: {value!r}")


def write_gguf(
    path: Path,
    tensors: list[tuple[str, str, list[int]]],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    metadata = metadata or {}
    encoded = bytearray(b"GGUF")
    encoded.extend(struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        value_type, payload = _gguf_value(value)
        encoded.extend(_gguf_string(key))
        encoded.extend(struct.pack("<I", value_type))
        encoded.extend(payload)
    for name, dtype, shape in tensors:
        encoded.extend(_gguf_string(name))
        encoded.extend(struct.pack("<I", len(shape)))
        for dimension in reversed(shape):
            encoded.extend(struct.pack("<Q", dimension))
        encoded.extend(struct.pack("<I", GGML_TYPES.index(dtype)))
        encoded.extend(struct.pack("<Q", 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_root = tmp_path / "data"
    return Settings(
        data_root=data_root,
        input_root=data_root / "input",
        output_root=data_root / "output",
        temp_root=data_root / "tmp",
        ggufy_binary=tmp_path / "fake-ggufy",
        max_concurrent_jobs=1,
        max_threads=4,
        max_header_bytes=1024 * 1024,
        max_marker_bytes=4096,
        max_marker_requests=100,
        remote_timeout_seconds=5,
        log_tail_bytes=4096,
    )
