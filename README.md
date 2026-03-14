# LLMine-srv
A lightweight, self-hosted inference server for running AI models locally. 
For loading all models you want!
---

## What it does

LLMine-srv lets you load and serve AI models from your own machine. Point it at a HuggingFace repo, an Ollama registry, or a local folder — and it handles the rest. Switch models on the fly, stream tokens as they're generated, and embed text without sending a single byte to an external API.

---

## Features

- **Multiple model formats** — Safetensors, GGUF, and ONNX all work as first-class citizens
- **Multiple registries** — pull from HuggingFace Hub, Ollama, or a local directory via `LOCAL_REGISTRY_DIR`
- **Streaming** — token-by-token streaming on all LLM endpoints via NDJSON
- **Lazy loading & auto-eviction** — models load on first request and are evicted from memory when idle, keeping resource usage lean
- **Windows-friendly** — GGUF inference runs via `ctransformers` (pre-built wheels, no C++ compiler needed)
- **Ollama-inspired API** — familiar endpoint structure if you've used Ollama before; easy to swap in

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server and model status |
| `POST` | `/api/generate` | Single-turn text generation |
| `POST` | `/api/chat` | Multi-turn chat with message history |
| `POST` | `/api/embed` | Text embeddings |
| `POST` | `/api/model` | Hot-swap the active model |

All write endpoints accept `"stream": true` for streaming responses.

Optional bearer token auth — set `API_KEY` env var to enable, leave it empty to skip.

---

## Supported backends

| Format | LLM | Embedder | Library |
|--------|-----|----------|---------|
| Safetensors | ✅ | ✅ | `transformers` + `torch` |
| GGUF | ✅ | ✅ | `ctransformers` / `sentence-transformers` |
| ONNX | ✅ | ✅ | `optimum` + `onnxruntime` |
| Ollama | ✅ | ✅ | Ollama daemon (HTTP) |


---

## Configuration

Example .env file was provided in the project, copy it in the root directory:
```cmd
cp ./example.env ./.env
```
Then modify variables as you wish, main ones described here:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(empty)* | Bearer token for auth. Auth is disabled when empty. |
| `LOCAL_REGISTRY_DIR` | *(none)* | Root directory for the local model registry |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon address |

---

## Quick start

### 1. Configure

Should have `.env` file with variables as in `example.env`

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