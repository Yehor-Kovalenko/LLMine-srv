"""
api.py — Route definitions, Ollama-inspired style.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .engine import EmbedEngine, ModelEngine
from .model import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    Message,
    ModelSwitchRequest,
    ModelSwitchResponse,
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

def _inference(request: Request) -> ModelEngine:
    return request.app.state.inference_engine


def _embedder(request: Request) -> EmbedEngine:
    return request.app.state.embed_engine


# Helpers

def _build_prompt(prompt: str, system: str | None) -> str:
    """Prepend a system instruction to a raw prompt when provided."""
    if not system:
        return prompt
    return f"{system}\n\n{prompt}"


def _messages_to_prompt(messages: list[Message]) -> str:
    """
    Minimal chat-template formatter for backends that need a single string.
    Replace with tokenizer.apply_chat_template() for production use.
    """
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|{m.role}|>\n{m.content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


async def _ndjson_stream(chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Wrap an async token iterator into newline-delimited JSON bytes."""
    async for chunk in chunks:
        yield (json.dumps(chunk) + "\n").encode()


# /api/status

@router.get("/api/status", tags=["ops"])
async def status_endpoint(request: Request) -> StatusResponse:
    inf: ModelEngine = _inference(request)
    emb: EmbedEngine = _embedder(request)
    return StatusResponse(
        status="ok",
        inference_model=inf.model_name or inf.default_name,
        inference_loaded=inf.is_loaded,
        inference_idle_seconds=inf.idle_seconds,
        embed_model=emb.model_name or emb.default_name,
        embed_loaded=emb.is_loaded,
        embed_idle_seconds=emb.idle_seconds,
    )


# ── /api/generate ─────────────────────────────────────────────────────────────

@router.post("/api/generate", tags=["inference"], dependencies=[Depends(_verify_key)])
async def generate(req: GenerateRequest, request: Request):
    engine: ModelEngine = _inference(request)
    opts = req.options
    prompt = _build_prompt(req.prompt, req.system)
    model_name = req.model or engine.default_name

    logger.info("generate model=%s stream=%s", model_name, req.stream)

    if req.stream:
        return StreamingResponse(
            _generate_stream(engine, prompt, model_name, opts),
            media_type="application/x-ndjson",
        )

    t0 = time.perf_counter()
    try:
        text, p_tok, c_tok = await engine.generate(
            prompt,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            top_p=opts.top_p,
            stop=opts.stop,
            model=req.model,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("generate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(
        model=model_name,
        response=text,
        done=True,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=p_tok + c_tok,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


async def _generate_stream(
    engine: ModelEngine,
    prompt: str,
    model_name: str,
    opts,
) -> AsyncIterator[bytes]:
    t0 = time.perf_counter()
    try:
        async for token in engine.stream(
            prompt,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            top_p=opts.top_p,
            stop=opts.stop,
        ):
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


# ── /api/chat ─────────────────────────────────────────────────────────────────

@router.post("/api/chat", tags=["inference"], dependencies=[Depends(_verify_key)])
async def chat(req: ChatRequest, request: Request):
    engine: ModelEngine = _inference(request)
    opts = req.options
    prompt = _messages_to_prompt(req.messages)
    model_name = req.model or engine.default_name

    logger.info("chat model=%s stream=%s messages=%d", model_name, req.stream, len(req.messages))

    if req.stream:
        return StreamingResponse(
            _chat_stream(engine, prompt, model_name, opts),
            media_type="application/x-ndjson",
        )

    t0 = time.perf_counter()
    try:
        text, p_tok, c_tok = await engine.generate(
            prompt,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            top_p=opts.top_p,
            stop=opts.stop,
            model=req.model,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        model=model_name,
        message=Message(role="assistant", content=text),
        done=True,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=p_tok + c_tok,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


async def _chat_stream(
    engine: ModelEngine,
    prompt: str,
    model_name: str,
    opts,
) -> AsyncIterator[bytes]:
    t0 = time.perf_counter()
    accumulated = []
    try:
        async for token in engine.stream(
            prompt,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            top_p=opts.top_p,
            stop=opts.stop,
        ):
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


# ── /api/embed ────────────────────────────────────────────────────────────────

@router.post("/api/embed", tags=["inference"], dependencies=[Depends(_verify_key)])
async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    engine: EmbedEngine = _embedder(request)
    texts = [req.input] if isinstance(req.input, str) else req.input
    model_name = req.model or engine.default_name

    logger.info("embed model=%s inputs=%d", model_name, len(texts))

    t0 = time.perf_counter()
    try:
        vectors = await engine.embed(texts, model=req.model)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("embed failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EmbedResponse(
        model=model_name,
        embeddings=vectors,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


# ── /api/model ────────────────────────────────────────────────────────────────

@router.post("/api/model", tags=["management"], dependencies=[Depends(_verify_key)])
async def switch_model(req: ModelSwitchRequest, request: Request) -> ModelSwitchResponse:
    """
    Evict the current inference model and set a new default.
    The new model is loaded lazily on the next generate/chat request.
    """
    engine: ModelEngine = _inference(request)
    logger.info("model switch requested: %s → %s", engine.default_name, req.model)

    await engine.switch(req.model)

    return ModelSwitchResponse(
        model=req.model,
        status="scheduled",
        message=f"Model switched to '{req.model}'. It will load on the next inference request.",
    )