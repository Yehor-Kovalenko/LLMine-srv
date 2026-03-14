"""
Ollama provider.

Communicates with a running Ollama daemon (https://ollama.com) via its
REST API to pull a model and retrieve its metadata.  The model weights live
inside the Ollama data directory managed by the daemon itself;

Dependencies:
    pip install requests
    A running Ollama daemon  →  https://ollama.com/download

Environment variables (all optional):
    OLLAMA_HOST   – base URL of the daemon, default ``http://localhost:11434``
"""

import logging
import os
from typing import Any

import requests

from base_model_registry import BaseProvider
from src.backend.model_registry_dto import ModelFormat, ModelPackage

logger = logging.getLogger(__name__)

_PULL_TIMEOUT  = 1200
_SHOW_TIMEOUT  = 30


class OllamaProvider(BaseProvider):
    """
    Loads models via the Ollama REST API.
    """

    def __init__(self, host: str | None = None) -> None:
        self._host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def load(self, name: str) -> ModelPackage:
        logger.info("Loading Ollama model '%s' from %s", name, self._host)

        self._pull(name)
        info = self._show(name)

        model_store = self._model_store_path()
        metadata = self._build_metadata(name, info)
        model_type = self._infer_model_type(name, info)
        architecture = info.get("details", {}).get("family", "")
        context_length = self._extract_context_length(info)

        return ModelPackage(
            id=f"ollama/{name}",
            model_type=model_type,
            path=f"{model_store}/{name}",
            format=ModelFormat.GGUF,
            metadata=metadata,
            architecture=architecture,
            context_length=context_length,
            dimensions=self._extract_dimensions(info),
        )

    # private util helpers

    def _pull(self, name: str) -> None:
        """
        Pull the model if it is not already present.
        Streams the pull progress and raises on failure.
        """
        url = f"{self._host}/api/pull"
        logger.debug("POST %s  model=%s", url, name)

        with requests.post(
            url,
            json={"name": name, "stream": True},
            stream=True,
            timeout=_PULL_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        import json
                        data = json.loads(line)
                        status = data.get("status", "")
                        if status:
                            logger.debug("pull: %s", status)
                        if data.get("error"):
                            raise RuntimeError(f"Ollama pull error: {data['error']}")
                    except (ValueError, KeyError):
                        pass  # non-JSON line, ignore

    def _show(self, name: str) -> dict:
        """Return the ``/api/show`` response for ``name``."""
        url = f"{self._host}/api/show"
        resp = requests.post(url, json={"name": name}, timeout=_SHOW_TIMEOUT)
        resp.raise_for_status()
        return resp.json()


    @staticmethod
    def _model_store_path() -> str:
        """
        Return the Ollama model store directory.

        Ollama uses ``$OLLAMA_MODELS`` when set, otherwise OS defaults:
            Linux/macOS  ~/.ollama/models
            Windows      %USERPROFILE%\\.ollama\\models
        """
        env = os.getenv("OLLAMA_MODELS")
        if env:
            return env
        home = os.path.expanduser("~")
        return os.path.join(home, ".ollama", "models")

    @staticmethod
    def _build_metadata(name: str, info: dict) -> dict:
        metadata: dict = {"model_name": name}
        details: dict = info.get("details", {})
        metadata.update({
            "parameter_size": details.get("parameter_size", ""),
            "quantization_level": details.get("quantization_level", ""),
            "family": details.get("family", ""),
            "families": details.get("families", []),
            "format": details.get("format", ""),
        })
        # Modelfile and template are useful for engine configuration
        if "modelfile" in info:
            metadata["modelfile"] = info["modelfile"]
        if "template" in info:
            metadata["template"] = info["template"]
        if "system" in info:
            metadata["system_prompt"] = info["system"]
        return metadata

    @staticmethod
    def _infer_model_type(name: str, info: dict) -> str:
        family = info.get("details", {}).get("family", "").lower()
        name_lower = name.lower()

        embed_keywords = ("embed", "nomic", "minilm", "bge", "e5-")
        if any(k in name_lower for k in embed_keywords) or any(k in family for k in embed_keywords):
            return "embedder"

        return "llm"

    @staticmethod
    def _extract_context_length(info: dict) -> int:
        # Ollama exposes modelinfo (Ollama >= 0.1.38) with llama.context_length etc.
        model_info: dict = info.get("model_info", {})
        for key in ("llama.context_length", "context_length", "num_ctx"):
            if key in model_info:
                return int(model_info[key])

        # Fallback: scan the raw modelfile for `PARAMETER num_ctx`
        modelfile: str = info.get("modelfile", "")
        for line in modelfile.splitlines():
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].upper() == "PARAMETER" and parts[1] == "num_ctx":
                try:
                    return int(parts[2])
                except ValueError:
                    pass

        return 2048

    @staticmethod
    def _extract_dimensions(info: dict) -> int | None:
        model_info: dict = info.get("model_info", {})
        for key in ("llama.embedding_length", "embedding_length", "hidden_size"):
            if key in model_info:
                return int(model_info[key])
        return None