"""
Local filesystem provider.

Reads models from a local directory tree rooted at the path given by the
``LOCAL_REGISTRY_DIR`` environment variable.

Expected directory layout
--------------------------
Each model lives in its own subdirectory.  The subdirectory name is used as
the model identifier when no explicit id is found in the config.

    $LOCAL_REGISTRY_DIR/
    ├── my-llm/
    │   ├── model.safetensors   (or model.gguf / model.onnx)
    │   ├── config.json
    │   └── tokenizer.json
    ├── my-embedder/
    │   ├── model.onnx
    │   └── config.json
    └── quantised-7b/
        └── quantised-7b.Q4_K_M.gguf

Format detection order (per model directory):
    1. Any ``*.safetensors`` → SAFETENSORS
    2. Any ``*.onnx`` file   → ONNX
    3. Any ``*.gguf`` file   → GGUF

Usage
-----
    os.environ["LOCAL_REGISTRY_DIR"] = "/opt/models"
    provider = LocalProvider()
    pkg = provider.load("my-llm")          # loads /opt/models/my-llm/
    pkg = provider.load("quantised-7b")    # loads the .gguf inside
"""

import json
import logging
import os
from pathlib import Path

from src.backend.model_registry_dto import ModelFormat, ModelPackage
from .base_model_registry import BaseProvider

logger = logging.getLogger(__name__)

ENV_VAR = "LOCAL_REGISTRY_DIR"


class LocalProvider(BaseProvider):

    def __init__(self, registry_dir: str | None = None) -> None:
        raw = registry_dir or os.getenv(ENV_VAR)
        if not raw:
            raise EnvironmentError(
                f"Local model registry directory is not configured. "
                f"Set the '{ENV_VAR}' environment variable or pass 'registry_dir' "
                f"to LocalProvider()."
            )
        self._root = Path(raw).expanduser().resolve()
        if not self._root.exists():
            raise FileNotFoundError(
                f"Local registry root does not exist: {self._root}"
            )
        logger.info("LocalProvider rooted at '%s'", self._root)

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def load(self, name: str) -> ModelPackage:
        model_dir = self._resolve_model_dir(name)
        logger.info("Loading local model '%s' from '%s'", name, model_dir)

        model_path, fmt = self._detect_format(model_dir)
        metadata = self._collect_metadata(model_dir, name)
        architecture = metadata.get("architectures", [""])[0] if metadata.get("architectures") else ""
        context_length = (
            metadata.get("max_position_embeddings")
            or metadata.get("n_positions")
            or metadata.get("context_length")
            or 2048
        )
        dimensions = metadata.get("hidden_size") or metadata.get("dim")
        model_type = self._infer_model_type(name, fmt, metadata)

        return ModelPackage(
            id=f"local/{name}",
            model_type=model_type,
            path=str(model_path),
            format=fmt,
            metadata=metadata,
            architecture=architecture,
            context_length=int(context_length),
            dimensions=int(dimensions) if dimensions else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_dir(self, name: str) -> Path:
        """
        Resolve ``name`` to an absolute path inside the registry root.
        Raises ``ValueError`` if the result would escape the root (path
        traversal guard) and ``FileNotFoundError`` if the directory is absent.
        """
        # Normalise separators; reject traversal attempts
        clean = Path(name)
        if ".." in clean.parts:
            raise ValueError(
                f"Model name '{name}' contains '..' which is not allowed."
            )

        candidate = (self._root / clean).resolve()

        # Ensure the resolved path is still inside the registry root
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Model name '{name}' resolves outside the registry root '{self._root}'."
            )

        if not candidate.exists():
            available = self._list_models()
            raise FileNotFoundError(
                f"Model '{name}' not found in '{self._root}'. "
                f"Available models: {available}"
            )

        if not candidate.is_dir():
            raise NotADirectoryError(
                f"Expected a directory for model '{name}', found a file: {candidate}"
            )

        return candidate

    @staticmethod
    def _detect_format(model_dir: Path) -> tuple[Path, ModelFormat]:
        """
        Determine the model format from the files present in ``model_dir``.

        Returns ``(primary_path, ModelFormat)``.

        Priority: GGUF > ONNX > SAFETENSORS
        If no model file is found a ``FileNotFoundError`` is raised.
        """
        checks: list[tuple[str, ModelFormat]] = [
            ("*.safetensors", ModelFormat.SAFETENSORS),
            ("*.onnx",         ModelFormat.ONNX),
            ("*.gguf", ModelFormat.GGUF),
        ]

        for pattern, fmt in checks:
            matches = sorted(model_dir.rglob(pattern))
            if matches:
                logger.debug("Detected format %s via '%s' in '%s'", fmt, pattern, model_dir)
                return matches[0], fmt

        raise FileNotFoundError(
            f"No model file (.gguf / .onnx / .safetensors) found in '{model_dir}'."
        )

    @staticmethod
    def _collect_metadata(model_dir: Path, name: str) -> dict:
        metadata: dict = {"local_name": name, "local_dir": str(model_dir)}

        for fname in ("config.json", "tokenizer_config.json", "generation_config.json"):
            p = model_dir / fname
            if p.exists():
                try:
                    with p.open() as fh:
                        for key, value in json.load(fh).items():
                            metadata.setdefault(key, value)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not read %s: %s", p, exc)

        return metadata

    @staticmethod
    def _infer_model_type(name: str, fmt: ModelFormat, metadata: dict) -> str:
        # Config-based
        architectures: list[str] = metadata.get("architectures", [])
        arch_str = " ".join(architectures).lower()
        if any(k in arch_str for k in ("forcausallm", "forlm", "seq2seqlm")):
            return "llm"
        if any(k in arch_str for k in ("forsequenceclassification", "fortokenclassification")):
            return "classifier"
        if any(k in arch_str for k in ("formaskedlm", "bertmodel", "robertamodel")):
            return "embedder"

        # Name heuristics
        name_lower = name.lower()
        if any(k in name_lower for k in ("embed", "minilm", "bge", "e5-", "nomic")):
            return "embedder"
        if any(k in name_lower for k in ("classif", "ner", "sentiment")):
            return "classifier"

        # GGUF is almost always a generative LLM
        if fmt == ModelFormat.GGUF:
            return "llm"

        return metadata.get("model_type", "unknown")

    def _list_models(self) -> list[str]:
        """Return names of immediate subdirectories in the registry root."""
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())