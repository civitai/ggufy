from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import httpx

from .models import ModelFormat, TensorDefinition, TensorSchema
from .safetensors import SafetensorsError, _read_range, huggingface_resolve_url


class GgufError(ValueError):
    """Raised when a GGUF header is malformed, truncated, or too large."""


class _NeedMoreData(Exception):
    def __init__(self, required: int):
        self.required = required


GGML_TYPES = (
    "f32",
    "f16",
    "q4_0",
    "q4_1",
    "q4_2",
    "q4_3",
    "q5_0",
    "q5_1",
    "q8_0",
    "q8_1",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_k",
    "iq2_xxs",
    "iq2_xs",
    "iq3_xxs",
    "iq1_s",
    "iq4_nl",
    "iq3_s",
    "iq2_s",
    "iq4_xs",
    "i8",
    "i16",
    "i32",
    "i64",
    "f64",
    "iq1_m",
    "bf16",
    "q4_0_4_4",
    "q4_0_4_8",
    "q4_0_8_8",
    "tq1_0",
    "tq2_0",
    "iq4_nl_4_4",
    "iq4_nl_4_8",
    "iq4_nl_8_8",
    "mxfp4",
    "nvfp4",
    "q1_0",
)

_VALUE_UINT8 = 0
_VALUE_INT8 = 1
_VALUE_UINT16 = 2
_VALUE_INT16 = 3
_VALUE_UINT32 = 4
_VALUE_INT32 = 5
_VALUE_FLOAT32 = 6
_VALUE_BOOL = 7
_VALUE_STRING = 8
_VALUE_ARRAY = 9
_VALUE_UINT64 = 10
_VALUE_INT64 = 11
_VALUE_FLOAT64 = 12
_VALUE_TYPE_COUNT = 13

_SCALAR_FORMATS = {
    _VALUE_UINT8: "<B",
    _VALUE_INT8: "<b",
    _VALUE_UINT16: "<H",
    _VALUE_INT16: "<h",
    _VALUE_UINT32: "<I",
    _VALUE_INT32: "<i",
    _VALUE_FLOAT32: "<f",
    _VALUE_UINT64: "<Q",
    _VALUE_INT64: "<q",
    _VALUE_FLOAT64: "<d",
}


class _Reader:
    def __init__(self, data: bytes | bytearray, max_header_bytes: int):
        self.data = data
        self.max_header_bytes = max_header_bytes
        self.position = 0

    def take(self, size: int) -> memoryview:
        required = self.position + size
        if required > self.max_header_bytes:
            raise GgufError(
                f"GGUF header exceeds the {self.max_header_bytes}-byte configured limit"
            )
        if required > len(self.data):
            raise _NeedMoreData(required)
        start = self.position
        self.position = required
        return memoryview(self.data)[start:required]

    def scalar(self, fmt: str) -> int | float:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return int(self.scalar("<I"))

    def u64(self) -> int:
        return int(self.scalar("<Q"))

    def string(self, description: str, *, nonempty: bool = False) -> str:
        length = self.u64()
        if nonempty and length == 0:
            raise GgufError(f"GGUF {description} must not be empty")
        if length > self.max_header_bytes:
            raise GgufError(f"GGUF {description} length {length} exceeds the configured limit")
        raw = bytes(self.take(length))
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GgufError(f"GGUF {description} is not valid UTF-8") from exc

    def value(self, value_type: int) -> Any:
        if value_type < 0 or value_type >= _VALUE_TYPE_COUNT:
            raise GgufError(f"unknown GGUF metadata value type {value_type}")
        if value_type == _VALUE_STRING:
            return self.string("metadata string")
        if value_type == _VALUE_BOOL:
            value = int(self.scalar("<B"))
            if value not in (0, 1):
                raise GgufError(f"invalid GGUF boolean value {value}")
            return bool(value)
        if value_type == _VALUE_ARRAY:
            element_type = self.u32()
            if element_type == _VALUE_ARRAY or element_type >= _VALUE_TYPE_COUNT:
                raise GgufError(f"invalid GGUF metadata array element type {element_type}")
            count = self.u64()
            minimum_size = (
                8
                if element_type == _VALUE_STRING
                else struct.calcsize(_SCALAR_FORMATS.get(element_type, "<B"))
            )
            if count > (self.max_header_bytes - self.position) // minimum_size:
                raise GgufError(
                    f"GGUF metadata array with {count} entries exceeds the configured limit"
                )
            return [self.value(element_type) for _ in range(count)]

        fmt = _SCALAR_FORMATS[value_type]
        value = self.scalar(fmt)
        if isinstance(value, float) and not math.isfinite(value):
            raise GgufError("GGUF metadata contains a non-finite floating-point value")
        return value


def parse_gguf_schema(
    data: bytes | bytearray,
    *,
    source: str,
    max_header_bytes: int,
) -> TensorSchema:
    """Parse only GGUF metadata and tensor descriptors; tensor payloads are not needed."""

    reader = _Reader(data, max_header_bytes)
    if bytes(reader.take(4)) != b"GGUF":
        raise GgufError("invalid GGUF magic")
    version = reader.u32()
    if version not in (2, 3):
        raise GgufError(f"unsupported GGUF version {version}; expected version 2 or 3")

    tensor_count = reader.u64()
    metadata_count = reader.u64()
    maximum_entries = max_header_bytes // 16
    if tensor_count > maximum_entries:
        raise GgufError(f"GGUF tensor count {tensor_count} exceeds the configured limit")
    if metadata_count > maximum_entries:
        raise GgufError(f"GGUF metadata count {metadata_count} exceeds the configured limit")

    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        key = reader.string("metadata key", nonempty=True)
        if key in metadata:
            raise GgufError(f"GGUF metadata contains duplicate key {key!r}")
        metadata[key] = reader.value(reader.u32())

    tensors: list[TensorDefinition] = []
    tensor_names: set[str] = set()
    for _ in range(tensor_count):
        name = reader.string("tensor name", nonempty=True)
        if name in tensor_names:
            raise GgufError(f"GGUF header contains duplicate tensor {name!r}")
        tensor_names.add(name)

        dimension_count = reader.u32()
        if dimension_count > 16:
            raise GgufError(
                f"GGUF tensor {name!r} has unsupported dimension count {dimension_count}"
            )
        gguf_shape = [reader.u64() for _ in range(dimension_count)]
        type_id = reader.u32()
        if type_id >= len(GGML_TYPES):
            raise GgufError(f"GGUF tensor {name!r} has unknown GGML type {type_id}")
        reader.u64()  # Tensor-data offset, not needed for schema copying.
        tensors.append(
            TensorDefinition(
                name=name,
                shape=list(reversed(gguf_shape)),
                dtype=GGML_TYPES[type_id],
            )
        )

    tensors.sort(key=lambda tensor: tensor.name)
    return TensorSchema(
        source=source,
        format=ModelFormat.GGUF,
        metadata=metadata,
        tensors=tensors,
    )


def read_local_gguf_schema(path: Path, max_header_bytes: int) -> TensorSchema:
    data = bytearray()
    requested = 64 * 1024
    with path.open("rb") as file:
        while len(data) < max_header_bytes:
            chunk = file.read(min(requested, max_header_bytes - len(data)))
            if not chunk:
                raise GgufError("truncated GGUF header")
            data.extend(chunk)
            try:
                return parse_gguf_schema(
                    data,
                    source=str(path),
                    max_header_bytes=max_header_bytes,
                )
            except _NeedMoreData as exc:
                requested = max(64 * 1024, exc.required - len(data))
    raise GgufError(f"GGUF header exceeds the {max_header_bytes}-byte configured limit")


async def read_huggingface_gguf_schema(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    token: str | None,
    max_header_bytes: int,
    timeout_seconds: float,
) -> TensorSchema:
    url = huggingface_resolve_url(repo_id, filename, revision)
    headers = {"User-Agent": "ggufy-api/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = bytearray()
    requested = 64 * 1024
    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
        while len(data) < max_header_bytes:
            end = min(max_header_bytes, len(data) + requested) - 1
            try:
                chunk = await _read_range(
                    client,
                    url,
                    len(data),
                    end,
                    headers,
                    require_exact=False,
                )
            except SafetensorsError as exc:
                raise GgufError(str(exc)) from exc
            if not chunk:
                raise GgufError("remote file has a truncated GGUF header")
            data.extend(chunk)
            try:
                return parse_gguf_schema(
                    data,
                    source=url,
                    max_header_bytes=max_header_bytes,
                )
            except _NeedMoreData as exc:
                requested = max(64 * 1024, exc.required - len(data))

    raise GgufError(f"GGUF header exceeds the {max_header_bytes}-byte configured limit")
