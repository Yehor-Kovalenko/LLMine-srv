"""
api.py — Route definitions, Ollama-inspired style.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.backend.engine import LazyLoader
from .model import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    Message,
    StatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Auth

_bearer = HTTPBearer(auto_error=False)
_API_KEY: str = os.getenv("API_KEY", "")


def _verify_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Skip auth entirely when API_KEY env var is empty."""
    if not _API_KEY:
        return
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid or missing API key."},
        )


# Engine accessors

def _inference(request: Request) -> LazyLoader:
    return request.app.state.inference_engine


def _embedder(request: Request) -> LazyLoader:
    return request.app.state.embed_engine


# /health

@router.get("/health", tags=["server"])
async def status_endpoint(request: Request) -> StatusResponse:
    inf: LazyLoader = _inference(request)
    emb: LazyLoader = _embedder(request)
    return StatusResponse(
        status="ok",
        inference_model=inf.model_name,
        inference_loaded=inf.is_loaded,
        inference_idle_seconds=inf.idle_seconds,
        embed_model=emb.model_name,
        embed_loaded=emb.is_loaded,
        embed_idle_seconds=emb.idle_seconds,
    )


# /api/generate

@router.post("/api/generate", tags=["inference"], dependencies=[Depends(_verify_key)])
async def generate(req: GenerateRequest, request: Request):
    llm_loader: LazyLoader = _inference(request)
    opts = req.options
    model_name = req.model

    logger.info("generate model=%s stream=%s", model_name, req.stream)

    input_data = {
        "prompt": req.prompt,
        "system_prompt": req.system_prompt or "",
        "max_tokens": opts.max_tokens,
        "temperature": opts.temperature,
        "top_p": opts.top_p,
        "stop": opts.stop,
        "model": req.model,
    }

    if req.stream:
        return StreamingResponse(
            _generate_stream(llm_loader, input_data, model_name),
            media_type="application/x-ndjson",
        )

    t0 = time.perf_counter()
    try:
        result: dict = await llm_loader.generate(input_data)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("generate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    usage = result.get("usage", {})
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)

    return GenerateResponse(
        model=model_name,
        response=result.get("text"),
        done=True,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=p_tok + c_tok,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


async def _generate_stream(
    engine: LazyLoader,
    input_data: dict,
    model_name: str,
) -> AsyncIterator[bytes]:
    t0 = time.perf_counter()
    try:
        async for token in engine.generate_stream(input_data):
            chunk = GenerateResponse(model=model_name, response=token, done=False)
            yield (chunk.model_dump_json() + "\n").encode()

        # Final done object
        final = GenerateResponse(
            model=model_name,
            response="",
            done=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        yield (final.model_dump_json() + "\n").encode()

    except Exception:
        logger.exception("generate stream failed")
        err = ErrorResponse(error="Streaming error").model_dump_json()
        yield (err + "\n").encode()


# /api/chat

@router.post("/api/chat", tags=["inference"], dependencies=[Depends(_verify_key)])
async def chat(req: ChatRequest, request: Request):
    llm_loader: LazyLoader = _inference(request)
    opts = req.options
    model_name = req.model
    input_data = {
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "max_tokens": opts.max_tokens,
        "temperature": opts.temperature,
        "top_p": opts.top_p,
        "stop": opts.stop,
        "model": req.model,
    }

    logger.info("chat model=%s stream=%s messages=%d", model_name, req.stream, len(req.messages))

    if req.stream:
        return StreamingResponse(
            _chat_stream(llm_loader, input_data, model_name),
            media_type="application/x-ndjson",
        )

    t0 = time.perf_counter()
    try:
        result: dict = await llm_loader.generate(input_data)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    usage = result.get("usage", {})
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)

    return ChatResponse(
        model=req.model,
        message=Message(role="assistant", content=result["text"]),
        done=True,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=p_tok + c_tok,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


async def _chat_stream(
    engine: LazyLoader,
    input_data: dict,
    model_name: str
) -> AsyncIterator[bytes]:
    t0 = time.perf_counter()
    accumulated = []
    try:
        async for token in engine.generate_stream(input_data):
            accumulated.append(token)
            chunk = ChatResponse(
                model=model_name,
                message=Message(role="assistant", content=token),
                done=False,
            )
            yield (chunk.model_dump_json() + "\n").encode()

        final = ChatResponse(
            model=model_name,
            message=Message(role="assistant", content=""),
            done=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        yield (final.model_dump_json() + "\n").encode()

    except Exception:
        logger.exception("chat stream failed")
        err = ErrorResponse(error="Streaming error").model_dump_json()
        yield (err + "\n").encode()


# /api/embed

@router.post("/api/embed", tags=["inference"], dependencies=[Depends(_verify_key)])
async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    embedder_loader: LazyLoader = _embedder(request)
    texts = [req.input] if isinstance(req.input, str) else req.input
    model_name = req.model

    logger.info("embed model=%s inputs=%d", model_name, len(texts))

    t0 = time.perf_counter()
    try:
        result: dict = await embedder_loader.generate({
            "input": texts,
            "model": req.model
        })
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("embed failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EmbedResponse(
        model=model_name,
        embeddings=result["embeddings"],
        duration_ms=(time.perf_counter() - t0) * 1000,
    )

# TODO list all models, list currently runm model, health only for current resource usage also not only for models
#TODO refactor the api