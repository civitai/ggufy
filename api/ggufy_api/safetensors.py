from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .models import TensorDefinition, TensorSchema


class SafetensorsError(ValueError):
    """Raised when a safetensors header or tensor cluster is invalid."""


@dataclass(frozen=True, slots=True)
class RawTensor:
    name: str
    dtype: str
    shape: list[int]
    offsets: tuple[int, int]

    @property
    def size(self) -> int:
        return self.offsets[1] - self.offsets[0]


@dataclass(frozen=True, slots=True)
class SafetensorsHeader:
    metadata: dict[str, str]
    tensors: dict[str, RawTensor]
    header_length: int

    @property
    def data_start(self) -> int:
        return 8 + self.header_length


MarkerReader = Callable[[RawTensor], Awaitable[bytes]]


def _parse_header_bytes(header_bytes: bytes, header_length: int) -> SafetensorsHeader:
    try:
        document = json.loads(header_bytes.rstrip(b" ").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"invalid safetensors JSON header: {exc}") from exc
    if not isinstance(document, dict):
        raise SafetensorsError("safetensors header must be a JSON object")

    metadata_value = document.pop("__metadata__", {})
    if not isinstance(metadata_value, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata_value.items()
    ):
        raise SafetensorsError("__metadata__ must be a string-to-string object")

    tensors: dict[str, RawTensor] = {}
    for name, value in document.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise SafetensorsError("invalid tensor entry in safetensors header")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) and offset >= 0 for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise SafetensorsError(f"invalid tensor metadata for {name!r}")
        tensors[name] = RawTensor(
            name=name,
            dtype=dtype,
            shape=shape,
            offsets=(offsets[0], offsets[1]),
        )
    return SafetensorsHeader(
        metadata=dict(metadata_value),
        tensors=tensors,
        header_length=header_length,
    )


def read_local_header(path: Path, max_header_bytes: int) -> SafetensorsHeader:
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise SafetensorsError("file is too small to be safetensors")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length < 2 or header_length > max_header_bytes:
            raise SafetensorsError(
                f"safetensors header length {header_length} is outside the allowed range"
            )
        header_bytes = file.read(header_length)
        if len(header_bytes) != header_length:
            raise SafetensorsError("truncated safetensors header")
    return _parse_header_bytes(header_bytes, header_length)


async def read_local_markers(
    path: Path,
    header: SafetensorsHeader,
    *,
    max_marker_bytes: int,
    max_marker_requests: int,
) -> dict[str, bytes]:
    markers = [tensor for tensor in header.tensors.values() if tensor.name.endswith(".comfy_quant")]
    if len(markers) > max_marker_requests:
        raise SafetensorsError(
            f"file has {len(markers)} quantization markers; limit is {max_marker_requests}"
        )

    def read_all() -> dict[str, bytes]:
        values: dict[str, bytes] = {}
        with path.open("rb") as file:
            for tensor in markers:
                if tensor.size > max_marker_bytes:
                    raise SafetensorsError(
                        f"quantization marker {tensor.name!r} exceeds {max_marker_bytes} bytes"
                    )
                file.seek(header.data_start + tensor.offsets[0])
                payload = file.read(tensor.size)
                if len(payload) != tensor.size:
                    raise SafetensorsError(f"truncated quantization marker {tensor.name!r}")
                values[tensor.name] = payload
        return values

    return await asyncio.to_thread(read_all)


async def _read_range(
    client: httpx.AsyncClient,
    url: str,
    start: int,
    end: int,
    headers: dict[str, str],
) -> bytes:
    expected = end - start + 1
    request_headers = {**headers, "Range": f"bytes={start}-{end}"}
    try:
        async with client.stream("GET", url, headers=request_headers) as response:
            response.raise_for_status()
            if response.status_code not in (200, 206):
                raise SafetensorsError(
                    f"range request returned unexpected HTTP {response.status_code}"
                )
            if response.status_code == 200 and start != 0:
                raise SafetensorsError("remote host ignored a non-zero HTTP range request")

            result = bytearray()
            async for chunk in response.aiter_bytes():
                result.extend(chunk)
                if len(result) >= expected:
                    break
            if len(result) < expected:
                raise SafetensorsError(
                    f"range request returned {len(result)} bytes; expected {expected}"
                )
            return bytes(result[:expected])
    except httpx.HTTPError as exc:
        raise SafetensorsError(f"remote range request failed: {exc}") from exc


def huggingface_resolve_url(repo_id: str, filename: str, revision: str) -> str:
    encoded_repo = quote(repo_id, safe="/")
    encoded_revision = quote(revision, safe="")
    encoded_filename = quote(filename, safe="/")
    return f"https://huggingface.co/{encoded_repo}/resolve/{encoded_revision}/{encoded_filename}"


async def read_huggingface_header(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    token: str | None,
    max_header_bytes: int,
    timeout_seconds: float,
) -> tuple[SafetensorsHeader, str, httpx.AsyncClient, dict[str, str]]:
    url = huggingface_resolve_url(repo_id, filename, revision)
    headers = {"User-Agent": "ggufy-api/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    client = httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)
    try:
        prefix = await _read_range(client, url, 0, 7, headers)
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length < 2 or header_length > max_header_bytes:
            raise SafetensorsError(
                f"remote safetensors header length {header_length} is outside the allowed range"
            )
        header_bytes = await _read_range(client, url, 8, 7 + header_length, headers)
        return _parse_header_bytes(header_bytes, header_length), url, client, headers
    except BaseException:
        await client.aclose()
        raise


async def read_remote_markers(
    *,
    header: SafetensorsHeader,
    url: str,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    max_marker_bytes: int,
    max_marker_requests: int,
    concurrency: int = 16,
) -> dict[str, bytes]:
    markers = [tensor for tensor in header.tensors.values() if tensor.name.endswith(".comfy_quant")]
    if len(markers) > max_marker_requests:
        raise SafetensorsError(
            f"remote file has {len(markers)} quantization markers; limit is {max_marker_requests}"
        )
    semaphore = asyncio.Semaphore(concurrency)

    async def read_one(tensor: RawTensor) -> tuple[str, bytes]:
        if tensor.size > max_marker_bytes:
            raise SafetensorsError(
                f"quantization marker {tensor.name!r} exceeds {max_marker_bytes} bytes"
            )
        absolute_start = header.data_start + tensor.offsets[0]
        async with semaphore:
            payload = await _read_range(
                client,
                url,
                absolute_start,
                absolute_start + tensor.size - 1,
                headers,
            )
        return tensor.name, payload

    return dict(await asyncio.gather(*(read_one(tensor) for tensor in markers)))


def _parse_json_object(value: str | bytes | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_to_dtype(identity: dict[str, Any] | None) -> str | None:
    if identity is None:
        return None
    quant_format = identity.get("format")
    if quant_format == "float8_e4m3fn":
        return "SCALED_F8_E4M3"
    if quant_format == "mxfp4":
        return "MXFP4"
    if quant_format in ("mxfp8", "mxfp8_e4m3fn"):
        return "MXFP8_E4M3"
    if quant_format == "nvfp4":
        return "NVFP4"
    if quant_format == "int8_tensorwise":
        return "INT8_CONVROT" if identity.get("convrot") is True else "INT8"
    if quant_format == "convrot_w4a4":
        return "INT4_CONVROT"
    return None


def _central_identities(
    header: SafetensorsHeader,
) -> dict[str, dict[str, Any]]:
    raw = header.metadata.get("_quantization_metadata")
    root = _parse_json_object(raw)
    if root is None or not isinstance(root.get("layers"), dict):
        return {}
    return {
        name: value
        for name, value in root["layers"].items()
        if isinstance(name, str) and isinstance(value, dict)
    }


def _identity_for_base(
    base: str,
    marker_values: dict[str, bytes],
    central: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    marker = _parse_json_object(marker_values.get(f"{base}.comfy_quant"))
    if marker is not None:
        return marker
    if base in central:
        return central[base]
    suffix_matches = [
        value
        for name, value in central.items()
        if base.endswith(f".{name}") or name.endswith(f".{base}")
    ]
    return suffix_matches[0] if len(suffix_matches) == 1 else None


def _infer_cluster_dtype(
    weight: RawTensor,
    scale: RawTensor,
    identity: dict[str, Any] | None,
) -> str | None:
    identified = _format_to_dtype(identity)
    if identified:
        return identified
    if weight.dtype == "F8_E4M3" and scale.dtype == "F32" and scale.shape == []:
        return "SCALED_F8_E4M3"
    if weight.dtype == "I8" and scale.dtype == "F32":
        return "INT8"
    if weight.dtype == "U32" and scale.dtype == "U8":
        return "MXFP4"
    return None


def logical_schema(
    header: SafetensorsHeader,
    *,
    source: str,
    marker_values: dict[str, bytes] | None = None,
) -> TensorSchema:
    marker_values = marker_values or {}
    central = _central_identities(header)
    companions: set[str] = set()
    clustered_weights: set[str] = set()
    logical: list[TensorDefinition] = []

    for name, tensor in header.tensors.items():
        if not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        scale_name = f"{base}.weight_scale"
        scale = header.tensors.get(scale_name)
        if scale is None:
            continue

        identity = _identity_for_base(base, marker_values, central)
        cluster_dtype = _infer_cluster_dtype(tensor, scale, identity)
        if cluster_dtype is None:
            continue

        shape = list(tensor.shape)
        if cluster_dtype.startswith("INT4_CONVROT") and len(shape) == 2:
            shape[1] *= 2
        if cluster_dtype == "MXFP4" and len(shape) == 2:
            shape[1] *= 8
        if cluster_dtype == "NVFP4" and len(shape) == 2:
            shape[1] *= 2
        companions.update(
            {
                scale_name,
                f"{base}.weight_scale_2",
                f"{base}.comfy_quant",
            }
        )
        clustered_weights.add(name)
        logical.append(TensorDefinition(name=name, shape=shape, dtype=cluster_dtype))

    for name, tensor in header.tensors.items():
        if name in clustered_weights or name in companions or name.endswith(".comfy_quant"):
            continue
        logical.append(TensorDefinition(name=name, shape=tensor.shape, dtype=tensor.dtype))

    logical.sort(key=lambda item: item.name)
    return TensorSchema(source=source, metadata=header.metadata, tensors=logical)
