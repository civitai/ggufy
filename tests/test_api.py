from __future__ import annotations

import asyncio
import stat
import struct
from pathlib import Path

import httpx
from ggufy_api.main import create_app
from ggufy_api.service import QuantizationService

from .conftest import write_safetensors


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
