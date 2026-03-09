# LLMine-srv

Local fully configurable LLM inference server built with **FastAPI**.
For loading all models you want!

## Stack

|Layer| Tech                                   |
|---|----------------------------------------|
|API framework| FastAPI + uvicorn                      |
|Streaming| sse-starlette (SSE)                    |
|Dependency manager| uv                                     |
|Inference backend| HuggingFace Transformers (safetensors) |

---

## Quick start

### 1. Configure

Create `.env` file with variables as in `example.env`

### 2. Run locally

```bash
uv sync
uv run python -m src.main
```
### Endpoints
Check `.env/SERVER_HOST`:`.env/SERVER_PORT`/docs for available endpoints

Set `"stream": true` in the request body. The server returns an SSE stream:

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
├── logs/                   # Created at runtime
└── src/
    ├── __init__.py
    |-- engine
         |-- __init__.py
         |- _base.py        # Base engine for lazy loading models
         |- embed_engine.py # for loading embedders
         |_ model_engine.py # engine for loading models
    ├── main.py             # App factory, middleware, logging, lifespan
    ├── api.py              # Route handlers
    ├── engine.py           # Lazy-load + TTL eviction
    └── model.py            # Pydantic request/response schemas
```
##### Features:
- Memory efficient lazy model loading that also handles model's eviction and freeing the machine's resources
- Live streaming response along side with standard response

[//]: # (- Api endpoint for choosing the adapter type &#40;finetuned&#41;)
[//]: # (- Endpoint for loading different models)
[//]: # (- Rate limiting using nginx)
[//]: # (- Api key using nginx)
[//]: # (- ngrok for remote tunnel sharing)
[//]: # (-  **The Fix for prompt injection:** Use **Chat Templates**. Don't manually build strings; use the tokenizer's `apply_chat_template` method to ensure the model can distinguish between "System" instructions and "User" input.)
- Loads only models in safetensors format to prevent from remote code execution attacks

[//]: # (- nd set a `max_model_len` or `max_tokens` limit in your inference engine &#40;vLLM or Hugging Face&#41; to prevent massive prompts from crashing the hardware.)
- Extensive configurable logging