"""
model.py — Request / response schemas.

Ollama like api
Streaming and non-streaming share the same response shape — the final
streaming object just carries `done: true` and timing stats.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


# Shared generation options

class GenerateOptions(BaseModel):
    """Sampling knobs — all optional, sensible defaults assumed by the engine."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stop: list[str] | None = None


# /api/generate  (raw prompt → answer)

class GenerateRequest(BaseModel):
    """
    Simple single-turn completion.
    Optionally inject a system instruction without building a messages list.
    """
    model: str | None = None          # overrides the server's active model
    prompt: str
    system_prompt: str | None = None         # prepended as a system instruction
    stream: bool = False
    options: GenerateOptions = Field(default_factory=GenerateOptions)


class GenerateResponse(BaseModel):
    """
    Non-streaming: returned once with the full answer.
    Streaming: many objects with done=False, one final with done=True + stats.
    """
    model: str
    response: str                     # the generated text (token or full)
    done: bool = False # populated only on the final streaming chunk / non-streaming response
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: float | None = None


# /api/chat (messages list → assistant reply)

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """
    Multi-turn chat. Pass the full conversation history each time —
    the server is stateless between requests.
    """
    model: str | None = None
    messages: list[Message]
    stream: bool = False
    options: GenerateOptions = Field(default_factory=GenerateOptions)


class ChatResponse(BaseModel):
    """
    Non-streaming: one object with the complete assistant message.
    Streaming: many objects with done=False carrying partial content,
               one final object with done=True and token stats.
    """
    model: str
    message: Message                  # role always "assistant"
    done: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: float | None = None


# /api/embed

class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str | None = None          # overrides the server's active embedder


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]     # one vector per input string
    duration_ms: float | None = None


# /api/status

class StatusResponse(BaseModel):
    status: Literal["ok", "degraded"]
    inference_model: str | None
    inference_loaded: bool
    inference_idle_seconds: float | None
    embed_model: str | None
    embed_loaded: bool
    embed_idle_seconds: float | None
    timestamp: float = Field(default_factory=time.time)


# Error

class ErrorResponse(BaseModel):
    error: str