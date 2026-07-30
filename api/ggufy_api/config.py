from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    data_root: Path
    input_root: Path
    output_root: Path
    temp_root: Path
    ggufy_binary: Path
    max_concurrent_jobs: int
    max_threads: int
    max_header_bytes: int
    max_marker_bytes: int
    max_marker_requests: int
    remote_timeout_seconds: float
    log_tail_bytes: int

    @classmethod
    def from_env(cls) -> Settings:
        data_root = Path(os.getenv("GGUFY_DATA_ROOT", "/data")).resolve()
        return cls(
            data_root=data_root,
            input_root=Path(os.getenv("GGUFY_INPUT_ROOT", str(data_root / "input"))).resolve(),
            output_root=Path(os.getenv("GGUFY_OUTPUT_ROOT", str(data_root / "output"))).resolve(),
            temp_root=Path(os.getenv("GGUFY_TEMP_ROOT", str(data_root / "tmp"))).resolve(),
            ggufy_binary=Path(os.getenv("GGUFY_BINARY", "ggufy")),
            max_concurrent_jobs=_positive_int("GGUFY_MAX_CONCURRENT_JOBS", 1),
            max_threads=_positive_int("GGUFY_MAX_THREADS", os.cpu_count() or 1),
            max_header_bytes=_positive_int("GGUFY_MAX_HEADER_BYTES", 128 * 1024 * 1024),
            max_marker_bytes=_positive_int("GGUFY_MAX_MARKER_BYTES", 64 * 1024),
            max_marker_requests=_positive_int("GGUFY_MAX_MARKER_REQUESTS", 20_000),
            remote_timeout_seconds=float(os.getenv("GGUFY_REMOTE_TIMEOUT_SECONDS", "30")),
            log_tail_bytes=_positive_int("GGUFY_LOG_TAIL_BYTES", 64 * 1024),
        )

    def prepare_directories(self) -> None:
        for path in (self.input_root, self.output_root, self.temp_root):
            path.mkdir(parents=True, exist_ok=True)
