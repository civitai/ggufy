from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings
from .models import (
    ConversionRequest,
    ConversionResponse,
    HealthResponse,
    HuggingFaceSchemaRequest,
    LocalSchemaRequest,
    PlanRequest,
    PlanResponse,
    TensorSchema,
)
from .planner import PlanError
from .safetensors import SafetensorsError
from .service import (
    ConversionError,
    OutputConflictError,
    PathPolicyError,
    QuantizationService,
)


def create_app(
    *,
    settings: Settings | None = None,
    service: QuantizationService | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    active_service = service or QuantizationService(active_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        active_service.prepare()
        yield

    app = FastAPI(
        title="GGUFy Quantization API",
        description=(
            "CPU-only safetensors conversion service using GGUFy. "
            "Tensor policies are resolved to an immutable, exact-name template."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.service = active_service

    @app.exception_handler(PathPolicyError)
    @app.exception_handler(PlanError)
    @app.exception_handler(SafetensorsError)
    async def bad_request(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OutputConflictError)
    async def conflict(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ConversionError)
    async def conversion_failed(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            ggufy_binary=active_service.binary_path(),
            max_concurrent_jobs=active_settings.max_concurrent_jobs,
        )

    @app.post("/v1/schemas/local", response_model=TensorSchema)
    async def inspect_local(request: LocalSchemaRequest) -> TensorSchema:
        return await active_service.local_schema(request)

    @app.post("/v1/schemas/huggingface", response_model=TensorSchema)
    async def inspect_huggingface(
        request: HuggingFaceSchemaRequest,
    ) -> TensorSchema:
        return await active_service.huggingface_schema(request)

    @app.post("/v1/plans/resolve", response_model=PlanResponse)
    async def plan(request: PlanRequest) -> PlanResponse:
        return await active_service.plan(request)

    @app.post("/v1/conversions", response_model=ConversionResponse)
    async def convert(request: ConversionRequest) -> ConversionResponse:
        return await active_service.convert(request)

    return app


app = create_app()
