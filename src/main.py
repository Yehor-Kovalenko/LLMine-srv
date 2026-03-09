"""
main.py — FastAPI application entry point.

Responsibilities
----------------
* Configure structured, rolling-file logging before anything else.
* Create the FastAPI app with a lifespan that boots / shuts down ModelEngine.
* Register middleware (request logging, CORS, error formatting).
* Mount the API router.
* Expose a __main__ block for direct `python -m src.main` execution.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()  # load .env before reading env vars

from .api import router                      # noqa: E402 — must come after load_dotenv
from .engine import EmbedEngine, ModelEngine  # noqa: E402
from .model import ErrorResponse  # noqa: E402

# Logging setup

def _configure_logging() -> None:
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    max_bytes: int = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    level_name: str = os.getenv("LOG_LEVEL", "info").upper()
    level: int = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rolling file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "server.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s dir=%s maxBytes=%d backupCount=%d",
        level_name, log_dir, max_bytes, backup_count,
    )


_configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Start engines. Lazy load models and embedders
    inference_engine = ModelEngine()
    embed_engine = EmbedEngine()
    app.state.inference_engine = inference_engine
    app.state.embed_engine = embed_engine

    await inference_engine.start()
    await embed_engine.start()

    logger.info("Application startup complete. Models will load on first request.")
    try:
        yield
    finally:
        await inference_engine.stop()
        await embed_engine.stop()
        logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Inference Server",
        description="LLM inference API with lazy-load, streaming, and embeddings.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — tighten origins for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # # ── Request / response logging middleware ─────────────────────────────────
    # @app.middleware("http")
    # async def _log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    #     req_id = uuid.uuid4().hex[:8]
    #     start = time.perf_counter()
    #     logger.info(
    #         "[%s] → %s %s  client=%s",
    #         req_id, request.method, request.url.path,
    #         request.client.host if request.client else "unknown",
    #     )
    #     try:
    #         response = await call_next(request)
    #     except Exception:
    #         logger.error("[%s] Unhandled exception\n%s", req_id, traceback.format_exc())
    #         return JSONResponse(
    #             status_code=500,
    #             content=ErrorResponse(error="Internal server error").model_dump(),
    #         )
    #     elapsed_ms = (time.perf_counter() - start) * 1000
    #     logger.info("[%s] ← %d  %.1fms", req_id, response.status_code, elapsed_ms)
    #     return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def _global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error=str(exc)).model_dump(),
        )

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=os.getenv("SERVER_HOST"),
        port=int(os.getenv("SERVER_PORT")),
        log_config=None,   # custom logging
        reload=os.getenv("SERVER_DEV_RELOAD", "false").lower() == "true",
    )