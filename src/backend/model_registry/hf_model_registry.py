"""
Hugging face model registry providers
"""

import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from base_model_registry import BaseProvider
from src.backend.model_registry_dto import ModelFormat, ModelPackage

logger = logging.getLogger(__name__)


class HFSafetensorsProvider(BaseProvider):
    """
    Loads models stored in safetensors format from the HuggingFace Hub.

    The model is downloaded to the default HF cache directory unless the
    environment variable ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` is set.
    """

    _ALLOWED_PATTERNS = [
        "*.safetensors",
        "*.json",  # config, tokenizer, special_tokens_map, …
        "*.txt",  # vocab, merges...
        "*.model",  # sentencepiece vocab
        "tokenizer.model",
    ]

    def load(self, name: str) -> ModelPackage:
        repo_id, revision = self._parse_name(name)
        logger.info("Loading safetensors model '%s' (rev=%s) from HuggingFace Hub", repo_id, revision)

        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=self._ALLOWED_PATTERNS,
            local_files_only=False,
        )

        metadata = self._collect_metadata(local_dir, repo_id, revision)
        architecture = metadata.get("architectures", [""])[0] if metadata.get("architectures") else ""
        context_length = (
                metadata.get("max_position_embeddings")
                or metadata.get("n_positions")
                or metadata.get("seq_length")
                or 2048
        )
        dimensions = metadata.get("hidden_size") or metadata.get("dim")
        model_type = self._infer_model_type(metadata)

        return ModelPackage(
            id=f"hf/{repo_id}",
            model_type=model_type,
            path=local_dir,
            format=ModelFormat.SAFETENSORS,
            metadata=metadata,
            architecture=architecture,
            context_length=int(context_length),
            dimensions=int(dimensions) if dimensions else None,
        )

    # util methods

    @staticmethod
    def _parse_name(name: str) -> tuple[str, str | None]:
        """Split ``"owner/repo@revision"`` into ``(repo_id, revision)``."""
        if "@" in name:
            repo_id, revision = name.rsplit("@", 1)
            return repo_id, revision or None
        return name, None

    @staticmethod
    def _collect_metadata(local_dir: str, repo_id: str, revision: str | None) -> dict:
        """
        Merge config.json + tokenizer_config.json into a single metadata dict.
        Falls back gracefully if files are missing.
        """
        metadata: dict = {"repo_id": repo_id}
        if revision:
            metadata["revision"] = revision

        for filename in ("config.json", "tokenizer_config.json", "generation_config.json"):
            config_path = Path(local_dir) / filename
            if config_path.exists():
                try:
                    with config_path.open() as fh:
                        data = json.load(fh)
                    # config.json values take precedence; others fill missing keys
                    for key, value in data.items():
                        metadata.setdefault(key, value)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not read %s: %s", config_path, exc)

        return metadata

    @staticmethod
    def _infer_model_type(metadata: dict) -> str: #TODO reconsider type inferention
        """
        Heuristically determine whether the model is an LLM, embedder, or
        classifier based on the HF config.
        """
        architectures: list[str] = metadata.get("architectures", [])
        arch_str = " ".join(architectures).lower()

        if any(k in arch_str for k in ("forcausallm", "forlm", "seq2seqlm")):
            return "llm"
        if any(k in arch_str for k in ("forsequenceclassification", "fortokenclassification")):
            return "classifier"
        if any(k in arch_str for k in ("formaskedlm", "bertmodel", "robertamodel", "distilbert")):
            return "embedder"

        # Fall back to model_type field if present
        return metadata.get("model_type", "unknown")

#############################################################################

# Sub-directories that Optimum commonly places ONNX exports in.
_ONNX_SUBDIRS = ("onnx", ".", "")

# File patterns relevant to ONNX deployments.
_ALLOWED_PATTERNS = [
    "*.onnx",
    "*.onnx_data",  # external data file for large models
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer.model",
]


class HFONNXProvider(BaseProvider):
    """
    Loads ONNX models from the HuggingFace Hub.

    ``name`` format:
        ``"optimum/bert-base-uncased"``
        ``"Xenova/all-MiniLM-L6-v2@main"``          – with revision
        ``"optimum/gpt2::decoder_model.onnx"``       – specific ONNX file
    """

    def load(self, name: str) -> ModelPackage:
        repo_id, revision, file_hint = self._parse_name(name)
        logger.info(
            "Loading ONNX model '%s' (rev=%s, hint=%s) from HuggingFace Hub",
            repo_id, revision, file_hint,
        )

        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=_ALLOWED_PATTERNS,
        )

        onnx_path = self._resolve_onnx_path(local_dir, file_hint)
        metadata = self._collect_metadata(local_dir, repo_id, revision, onnx_path)

        architecture = metadata.get("architectures", [""])[0] if metadata.get("architectures") else ""
        context_length = (
                metadata.get("max_position_embeddings")
                or metadata.get("n_positions")
                or 512  # transformers / ONNX models are commonly seq-512
        )
        dimensions = metadata.get("hidden_size") or metadata.get("dim")
        model_type = self._infer_model_type(metadata)

        return ModelPackage(
            id=f"hf/{repo_id}",
            model_type=model_type,
            path=str(onnx_path),  # path to the primary .onnx file
            format=ModelFormat.ONNX,
            metadata=metadata,
            architecture=architecture,
            context_length=int(context_length),
            dimensions=int(dimensions) if dimensions else None,
        )

    # private utils

    @staticmethod
    def _parse_name(name: str) -> tuple[str, str | None, str | None]:
        """Parse ``"owner/repo[@rev][::onnx_filename]"``."""
        file_hint: str | None = None
        if "::" in name:
            name, file_hint = name.split("::", 1)
            file_hint = file_hint.strip() or None

        revision: str | None = None
        if "@" in name:
            name, revision = name.rsplit("@", 1)
            revision = revision.strip() or None

        return name.strip(), revision, file_hint

    @staticmethod
    def _resolve_onnx_path(local_dir: str, hint: str | None) -> Path:
        """
        Locate the primary ONNX file inside a downloaded snapshot directory.

        Priority:
        1. Explicit hint (filename or relative path).
        2. ``onnx/model.onnx`` (Optimum default).
        3. ``model.onnx`` at repo root.
        4. Any ``encoder_model.onnx`` or ``decoder_model.onnx``.
        5. First ``.onnx`` file found anywhere.
        """
        root = Path(local_dir)

        if hint:
            candidate = root / hint
            if candidate.exists():
                return candidate
            # Search recursively for the basename
            matches = list(root.rglob(hint))
            if matches:
                return matches[0]
            raise FileNotFoundError(
                f"ONNX file '{hint}' not found in {local_dir}"
            )

        for subdir in _ONNX_SUBDIRS:
            for fname in ("model.onnx", "model_quantized.onnx"):
                p = root / subdir / fname
                if p.exists():
                    return p

        # Prefer encoder over decoder for embedding/classification use-cases
        for pattern in ("encoder_model.onnx", "model_encoder.onnx", "decoder_model.onnx"):
            matches = list(root.rglob(pattern))
            if matches:
                return matches[0]

        # Fallback: first .onnx file
        all_onnx = sorted(root.rglob("*.onnx"))
        if all_onnx:
            return all_onnx[0]

        raise FileNotFoundError(f"No .onnx file found in snapshot at {local_dir}")

    @staticmethod
    def _collect_metadata(
            local_dir: str,
            repo_id: str,
            revision: str | None,
            onnx_path: Path,
    ) -> dict:
        metadata: dict = {
            "repo_id": repo_id,
            "onnx_file": str(onnx_path),
        }
        if revision:
            metadata["revision"] = revision

        root = Path(local_dir)
        for fname in ("config.json", "tokenizer_config.json"):
            p = root / fname
            if not p.exists():
                # Check inside onnx/ subdirectory too
                p = root / "onnx" / fname
            if p.exists():
                try:
                    with p.open() as fh:
                        for key, value in json.load(fh).items():
                            metadata.setdefault(key, value)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not read %s: %s", p, exc)

        return metadata

    @staticmethod
    def _infer_model_type(metadata: dict) -> str:
        architectures: list[str] = metadata.get("architectures", [])
        arch_str = " ".join(architectures).lower()

        if any(k in arch_str for k in ("forcausallm", "forlm", "seq2seqlm")):
            return "llm"
        if any(k in arch_str for k in ("forsequenceclassification", "fortokenclassification")):
            return "classifier"
        if any(k in arch_str for k in ("formaskedlm", "bertmodel", "robertamodel")):
            return "embedder"

        return metadata.get("model_type", "unknown")
