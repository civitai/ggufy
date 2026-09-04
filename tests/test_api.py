from __future__ import annotations

import asyncio
import json
import stat
import struct
from pathlib import Path

import httpx
from ggufy_api.gguf import read_local_gguf_schema
from ggufy_api.main import create_app
from ggufy_api.service import QuantizationService

from .conftest import write_gguf, write_safetensors


def _write_fake_ggufy(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import pathlib
import shutil
import sys

args = sys.argv[1:]
source = pathlib.Path(args[1])
output_dir = pathlib.Path(args[args.index("--output-dir") + 1])
output_name = args[args.index("--output-name") + 1]
shutil.copyfile(source, output_dir / f"{output_name}.safetensors")
print("fake ggufy conversion complete")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class _FakeGgufService(QuantizationService):
    command: list[str]

    async def _run(self, command: list[str]) -> tuple[int, str, str]:
        self.command = command
        template_path = Path(command[command.index("--template") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_name = command[command.index("--output-name") + 1]
        template = json.loads(await asyncio.to_thread(template_path.read_text, encoding="utf-8"))
        tensors = [
            (name, definition["type"], list(reversed(definition["shape"])))
            for name, definition in template["tensors"].items()
        ]
        await asyncio.to_thread(
            write_gguf,
            output_dir / f"{output_name}.gguf",
            tensors,
            metadata=template.get("metadata"),
        )
        return 0, "fake GGUF conversion complete", ""


async def test_health_and_preserving_conversion(settings) -> None:
    settings.prepare_directories()
    _write_fake_ggufy(settings.ggufy_binary)
    source = settings.input_root / "tiny.safetensors"
    write_safetensors(
        source,
        [
            ("layer.weight", "F32", [2, 2], struct.pack("<4f", 1, 2, 3, 4)),
            ("layer.bias", "F32", [2], struct.pack("<2f", 1, 2)),
        ],
    )
    app = create_app(settings=settings, service=QuantizationService(settings))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["cpu_only"] is True

            response = await client.post(
                "/v1/conversions",
                json={
                    "input_path": "tiny.safetensors",
                    "output_path": "tiny-copy.safetensors",
                    "default_type": "PRESERVE",
                },
            )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["output_path"] == str(settings.output_root / "tiny-copy.safetensors")
    assert result["type_counts"] == {"F32": 2}
    output_bytes, source_bytes = await asyncio.gather(
        asyncio.to_thread(Path(result["output_path"]).read_bytes),
        asyncio.to_thread(source.read_bytes),
    )
    assert output_bytes == source_bytes


async def test_paths_cannot_escape_roots(settings, tmp_path: Path) -> None:
    settings.prepare_directories()
    _write_fake_ggufy(settings.ggufy_binary)
    outside = tmp_path / "outside.safetensors"
    write_safetensors(outside, [("weight", "F32", [1], struct.pack("<f", 1))])
    app = create_app(settings=settings, service=QuantizationService(settings))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/plans/resolve",
                json={"input_path": str(outside), "default_type": "PRESERVE"},
            )

    assert response.status_code == 400
    assert "must be within" in response.json()["detail"]


async def test_output_escape_is_rejected_before_directories_are_created(
    settings, tmp_path: Path
) -> None:
    settings.prepare_directories()
    _write_fake_ggufy(settings.ggufy_binary)
    source = settings.input_root / "tiny.safetensors"
    write_safetensors(source, [("weight", "F32", [1], struct.pack("<f", 1))])
    outside_parent = tmp_path / "outside" / "created-by-mistake"
    app = create_app(settings=settings, service=QuantizationService(settings))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/conversions",
                json={
                    "input_path": "tiny.safetensors",
                    "output_path": str(outside_parent / "out.safetensors"),
                    "default_type": "PRESERVE",
                },
            )

    assert response.status_code == 400
    assert not outside_parent.exists()


async def test_gguf_conversion_uses_reference_metadata_and_extension(settings) -> None:
    settings.prepare_directories()
    source = settings.input_root / "tiny.safetensors"
    write_safetensors(
        source,
        [("layer.weight", "F32", [2, 256], b"\0" * (2 * 256 * 4))],
    )
    service = _FakeGgufService(settings)
    app = create_app(settings=settings, service=service)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/conversions",
                json={
                    "input_path": "tiny.safetensors",
                    "output_path": "tiny-q8.gguf",
                    "output_format": "gguf",
                    "schema": {
                        "format": "gguf",
                        "metadata": {"general.architecture": "flux"},
                        "tensors": [
                            {
                                "name": "layer.weight",
                                "shape": [2, 256],
                                "dtype": "q8_0",
                            }
                        ],
                    },
                    "unmatched": "error",
                },
            )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["format"] == "gguf"
    assert result["output_path"] == str(settings.output_root / "tiny-q8.gguf")
    assert result["type_counts"] == {"q8_0": 1}
    assert service.command[service.command.index("--filetype") + 1] == "gguf"
    assert service.command[service.command.index("--arch") + 1] == "flux"

    output_schema = read_local_gguf_schema(Path(result["output_path"]), settings.max_header_bytes)
    assert output_schema.format == "gguf"
    assert output_schema.tensors[0].dtype == "q8_0"
