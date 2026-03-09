# my-llm-server

OpenAI-compatible LLM inference server built with **FastAPI**, **uv**, and **NGINX**.

## Stack

|Layer|Tech|
|---|---|
|API framework|FastAPI + uvicorn|
|Streaming|sse-starlette (SSE)|
|Dependency manager|uv|
|Gateway / proxy|NGINX (rate-limit + API-key auth)|
|Inference backend|vLLM **or** HuggingFace Transformers|
|Containerisation|Docker Compose|

---

## Quick start

### 1. Configure

bash

```bash
cp .env .env.local      # edit secrets — set API_KEY and MODEL_PATH
```

Key env vars:

|Variable|Default|Description|
|---|---|---|
|`API_KEY`|_(required)_|Bearer token checked by NGINX|
|`MODEL_PATH`|`gpt2`|HF model-id or local path|
|`ENGINE_BACKEND`|`huggingface`|`vllm` or `huggingface`|
|`MODEL_TTL_SECONDS`|`3600`|Seconds idle before model eviction|
|`LOG_DIR`|`logs`|Rolling log directory|
|`LOG_MAX_BYTES`|`10485760`|Max bytes per log file (10 MB)|
|`LOG_BACKUP_COUNT`|`5`|Number of rotated files to keep|

### 2. Wire up your backend

Open `src/engine.py` and implement the two methods:

- `load_model(name)` — return your vLLM `LLM` or HF `pipeline` object.
- `ModelEngine.generate()` — call the model and return `(text, prompt_tokens, completion_tokens)`.
- `ModelEngine.stream()` — `async for token in ...` yield one token string at a time.

The stubs contain commented-out example code for both backends.

### 3. Run locally (no Docker)

bash

```bash
uv sync
uv run python -m src.main
```

### 4. Run with Docker Compose

bash

```bash
docker compose up --build
```

The gateway listens on **port 80**. The FastAPI docs are available at `http://localhost/docs`.

---

## Endpoints

|Method|Path|Auth|Description|
|---|---|---|---|
|`GET`|`/health`|✗|Liveness + model state|
|`GET`|`/v1/models`|✓|List available models|
|`POST`|`/v1/chat/completions`|✓|Chat (streaming + batch)|
|`POST`|`/v1/completions`|✓|Text completion (streaming + batch)|

### Streaming

Set `"stream": true` in the request body. The server returns an SSE stream:

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"role":"assistant"}}],...}
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"}}],...}
...
data: [DONE]
```

---

## Logging

Logs are written to `./logs/server.log` with automatic rotation:

- Max file size: `LOG_MAX_BYTES` (default 10 MB)
- Backup files: `LOG_BACKUP_COUNT` (default 5)
- Format: `TIMESTAMP | LEVEL | logger | message`

---

## Project structure

```
my-llm-server/
├── .env                    # Secrets (API_KEY, MODEL_PATH, …)
├── pyproject.toml          # uv project definition
├── uv.lock                 # Deterministic lock file
├── Dockerfile
├── docker-compose.yml      # NGINX + FastAPI
├── gateway/
│   └── nginx.conf          # Rate-limit + API-key auth
├── logs/                   # Created at runtime
└── src/
    ├── __init__.py
    ├── main.py             # App factory, middleware, logging, lifespan
    ├── api.py              # Route handlers
    ├── engine.py           # Lazy-load + TTL eviction
    └── model.py            # Pydantic request/response schemas
```
##### Features:
- Lazy loading models - server starts (no model) -> first request made -> load model -> keep in cache until server is running (Lazy Loading)
- Health endpoint
- Live streaming response along side with standard response
- Api endpoint for choosing the adapter type (finetuned)
- Endpoint for loading different models
- Rate limiting using nginx
- Api key using nginx
- ngrok for remote tunnel sharing
-  **The Fix for prompt injection:** Use **Chat Templates**. Don't manually build strings; use the tokenizer's `apply_chat_template` method to ensure the model can distinguish between "System" instructions and "User" input.
- Use fastapi, safetensors type for saving model locally
- nd set a `max_model_len` or `max_tokens` limit in your inference engine (vLLM or Hugging Face) to prevent massive prompts from crashing the hardware.
- logs
- config


---

### The Workflow Loop

This is how your life changes once you adopt this:

1. **Fine-Tune:** You write a training script using `uv` and Hugging Face.
2. **Save:** You call `model.save_pretrained("./my-new-model")`.
    - _Time taken:_ A few seconds to write the files to your SSD. (save as safetensors)
3. **Serve:** You point your API server (vLLM or your custom FastAPI script) to `./my-new-model`.
4. **Test:** You send an HTTP request (`curl` or Postman).