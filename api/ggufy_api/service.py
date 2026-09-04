from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from .config import Settings
from .gguf import GgufError, read_huggingface_gguf_schema, read_local_gguf_schema
from .models import (
    ConversionRequest,
    ConversionResponse,
    HuggingFaceSchemaRequest,
    LocalSchemaRequest,
    ModelFormat,
    PlanRequest,
    PlanResponse,
    TensorSchema,
)
from .planner import ResolvedPlan, resolve_plan
from .safetensors import (
    SafetensorsError,
    logical_schema,
    read_huggingface_header,
    read_local_header,
    read_local_markers,
    read_remote_markers,
)


class PathPolicyError(ValueError):
    """Raised when a requested path escapes the configured data roots."""


class OutputConflictError(FileExistsError):
    """Raised when an output exists and overwrite was not requested."""


class ConversionError(RuntimeError):
    """Raised when GGUFy conversion or output validation fails."""


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class QuantizationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._conversion_slots = asyncio.Semaphore(settings.max_concurrent_jobs)

    def prepare(self) -> None:
        self.settings.prepare_directories()

    def binary_path(self) -> str:
        configured = str(self.settings.ggufy_binary)
        resolved = shutil.which(configured)
        return resolved or configured

    def resolve_input_path(self, raw_path: str) -> Path:
        resolved = self._resolve_existing_path(raw_path)
        if resolved.suffix.lower() != ".safetensors":
            raise PathPolicyError("conversion input must be a .safetensors file")
        return resolved

    def resolve_schema_path(self, raw_path: str) -> Path:
        resolved = self._resolve_existing_path(raw_path)
        if resolved.suffix.lower() not in (".safetensors", ".gguf"):
            raise PathPolicyError("schema input must be a .safetensors or .gguf file")
        return resolved

    def _resolve_existing_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.settings.input_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PathPolicyError(f"input file does not exist: {candidate}") from exc
        if not _inside(resolved, self.settings.input_root):
            raise PathPolicyError(f"input path must be within {self.settings.input_root}")
        if not resolved.is_file():
            raise PathPolicyError("input path must be a file")
        return resolved

    def resolve_output_path(
        self,
        raw_path: str | None,
        *,
        input_path: Path,
        plan: ResolvedPlan,
        output_format: ModelFormat,
        overwrite: bool,
    ) -> Path:
        extension = f".{output_format.value}"
        if raw_path is None:
            if len(plan.response.type_counts) == 1:
                label = next(iter(plan.response.type_counts)).lower()
            else:
                label = "mixed"
            candidate = self.settings.output_root / f"{input_path.stem}-{label}{extension}"
        else:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self.settings.output_root / candidate

        if candidate.suffix.lower() != extension:
            raise PathPolicyError(f"{output_format.value} output path must end with {extension}")
        parent = candidate.parent.resolve(strict=False)
        if not _inside(parent, self.settings.output_root):
            raise PathPolicyError(f"output path must be within {self.settings.output_root}")
        parent.mkdir(parents=True, exist_ok=True)
        parent = parent.resolve(strict=True)
        resolved = parent / candidate.name
        if resolved == input_path:
            raise PathPolicyError("output path must differ from input path")
        if resolved.exists() and not overwrite:
            raise OutputConflictError(f"output already exists: {resolved}")
        return resolved

    async def local_schema(self, request: LocalSchemaRequest) -> TensorSchema:
        path = self.resolve_schema_path(request.path)
        if path.suffix.lower() == ".gguf":
            return await asyncio.to_thread(
                read_local_gguf_schema,
                path,
                self.settings.max_header_bytes,
            )
        header = await asyncio.to_thread(read_local_header, path, self.settings.max_header_bytes)
        markers = (
            await read_local_markers(
                path,
                header,
                max_marker_bytes=self.settings.max_marker_bytes,
                max_marker_requests=self.settings.max_marker_requests,
            )
            if request.read_markers
            else {}
        )
        return logical_schema(header, source=str(path), marker_values=markers)

    async def huggingface_schema(self, request: HuggingFaceSchemaRequest) -> TensorSchema:
        if request.filename.lower().endswith(".gguf"):
            return await read_huggingface_gguf_schema(
                repo_id=request.repo_id,
                filename=request.filename,
                revision=request.revision,
                token=os.getenv("HF_TOKEN"),
                max_header_bytes=self.settings.max_header_bytes,
                timeout_seconds=self.settings.remote_timeout_seconds,
            )
        header, url, client, headers = await read_huggingface_header(
            repo_id=request.repo_id,
            filename=request.filename,
            revision=request.revision,
            token=os.getenv("HF_TOKEN"),
            max_header_bytes=self.settings.max_header_bytes,
            timeout_seconds=self.settings.remote_timeout_seconds,
        )
        try:
            markers = (
                await read_remote_markers(
                    header=header,
                    url=url,
                    client=client,
                    headers=headers,
                    max_marker_bytes=self.settings.max_marker_bytes,
                    max_marker_requests=self.settings.max_marker_requests,
                )
                if request.read_markers
                else {}
            )
        finally:
            await client.aclose()
        return logical_schema(header, source=url, marker_values=markers)

    async def _source_schema(self, input_path: Path) -> TensorSchema:
        return await self.local_schema(LocalSchemaRequest(path=str(input_path), read_markers=True))

    async def plan(self, request: PlanRequest) -> PlanResponse:
        input_path = self.resolve_input_path(request.input_path)
        source = await self._source_schema(input_path)
        return resolve_plan(source, request).response

    async def convert(self, request: ConversionRequest) -> ConversionResponse:
        input_path = self.resolve_input_path(request.input_path)
        source = await self._source_schema(input_path)
        plan = resolve_plan(source, request)
        output_path = self.resolve_output_path(
            request.output_path,
            input_path=input_path,
            plan=plan,
            output_format=request.output_format,
            overwrite=request.overwrite,
        )
        threads = min(request.threads or self.settings.max_threads, self.settings.max_threads)

        self.settings.temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="ggufy-plan-",
            dir=self.settings.temp_root,
            delete=False,
        ) as template_file:
            json.dump(plan.template, template_file, separators=(",", ":"))
            template_path = Path(template_file.name)

        temporary_name = f".{output_path.stem}.{uuid.uuid4().hex}.partial"
        temporary_output = output_path.parent / f"{temporary_name}.{request.output_format.value}"
        command = [
            self.binary_path(),
            "convert",
            str(input_path),
            "--filetype",
            request.output_format.value,
            "--template",
            str(template_path),
            "--output-dir",
            str(output_path.parent),
            "--output-name",
            temporary_name,
            "--threads",
            str(threads),
            # Every API request has already been resolved to an exact tensor
            # template. Architecture recognition is therefore not needed to
            # choose types, and should not block new diffusion architectures.
            "--allow-unknown-arch",
        ]
        if request.output_format == ModelFormat.GGUF and request.reference_schema is not None:
            architecture = request.reference_schema.metadata.get("general.architecture")
            if isinstance(architecture, str) and architecture:
                command.extend(("--arch", architecture))
        if request.allow_upscale:
            command.append("--allow-upscale")

        started = time.monotonic()
        try:
            async with self._conversion_slots:
                return_code, stdout_tail, stderr_tail = await self._run(command)
            if return_code:
                raise ConversionError(
                    f"ggufy exited with status {return_code}: "
                    f"{stderr_tail or stdout_tail or 'no log output'}"
                )
            if not temporary_output.is_file():
                raise ConversionError(
                    "ggufy completed without producing the expected output: "
                    f"{stderr_tail or stdout_tail or 'no log output'}"
                )

            await self._validate_output(temporary_output, plan, request.output_format)
            if output_path.exists() and not request.overwrite:
                raise OutputConflictError(f"output already exists: {output_path}")
            if request.overwrite:
                temporary_output.replace(output_path)
            else:
                try:
                    os.link(temporary_output, output_path)
                except FileExistsError as exc:
                    raise OutputConflictError(f"output already exists: {output_path}") from exc
                temporary_output.unlink()

            return ConversionResponse(
                output_path=str(output_path),
                format=request.output_format,
                output_size=output_path.stat().st_size,
                tensor_count=len(plan.response.tensors),
                type_counts=plan.response.type_counts,
                duration_seconds=round(time.monotonic() - started, 3),
                log_tail=(stdout_tail + "\n" + stderr_tail).strip(),
            )
        finally:
            await asyncio.to_thread(template_path.unlink, missing_ok=True)
            await asyncio.to_thread(temporary_output.unlink, missing_ok=True)

    async def _run(self, command: list[str]) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ConversionError(f"GGUFy executable was not found: {command[0]}") from exc

        stdout_task = asyncio.create_task(self._read_tail(process.stdout))
        stderr_task = asyncio.create_task(self._read_tail(process.stderr))
        try:
            return_code = await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
            raise
        finally:
            stdout_tail, stderr_tail = await asyncio.gather(stdout_task, stderr_task)
        return return_code, stdout_tail, stderr_tail

    async def _read_tail(self, stream: asyncio.StreamReader | None) -> str:
        if stream is None:
            return ""
        tail = bytearray()
        while chunk := await stream.read(8192):
            tail.extend(chunk)
            if len(tail) > self.settings.log_tail_bytes:
                del tail[: len(tail) - self.settings.log_tail_bytes]
        return tail.decode("utf-8", errors="replace")

    async def _validate_output(
        self,
        path: Path,
        plan: ResolvedPlan,
        output_format: ModelFormat,
    ) -> None:
        if output_format == ModelFormat.GGUF:
            try:
                actual = await asyncio.to_thread(
                    read_local_gguf_schema,
                    path,
                    self.settings.max_header_bytes,
                )
            except GgufError as exc:
                raise ConversionError(f"GGUFy output is not valid GGUF: {exc}") from exc
        else:
            try:
                header = await asyncio.to_thread(
                    read_local_header, path, self.settings.max_header_bytes
                )
                markers = await read_local_markers(
                    path,
                    header,
                    max_marker_bytes=self.settings.max_marker_bytes,
                    max_marker_requests=self.settings.max_marker_requests,
                )
                actual = logical_schema(header, source=str(path), marker_values=markers)
            except SafetensorsError as exc:
                raise ConversionError(f"GGUFy output is not valid safetensors: {exc}") from exc

        expected_by_name = {tensor.name: tensor for tensor in plan.response.tensors}
        actual_by_name = {tensor.name: tensor for tensor in actual.tensors}
        if expected_by_name.keys() != actual_by_name.keys():
            missing = sorted(expected_by_name.keys() - actual_by_name.keys())
            extra = sorted(actual_by_name.keys() - expected_by_name.keys())
            raise ConversionError(
                f"output tensor set differs from plan; missing={missing[:5]}, extra={extra[:5]}"
            )
        for name, expected in expected_by_name.items():
            observed = actual_by_name[name]
            normalized_expected = (
                "INT4_CONVROT" if expected.dtype == "INT4_CONVROT_SR" else expected.dtype
            )
            if observed.shape != expected.shape or observed.dtype != normalized_expected:
                raise ConversionError(
                    f"output tensor {name!r} differs from plan: expected "
                    f"{expected.shape}/{normalized_expected}, observed "
                    f"{observed.shape}/{observed.dtype}"
                )
