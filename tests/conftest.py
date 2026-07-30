from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from ggufy_api.config import Settings


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
