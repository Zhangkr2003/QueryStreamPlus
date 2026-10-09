"""Resolve default and custom model paths for QueryStream++."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "models" / "QueryStreamPlus-7B"
DEFAULT_SIGLIP2_PATH = REPOSITORY_ROOT / "models" / "siglip2-large-patch16-256"
MANIFEST_NAME = "querystream_plus_config.json"


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path
    router: Path
    adapter: Path
    siglip2: Path


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _manifest(model: Path) -> dict:
    path = model / MANIFEST_NAME
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def _relative_to_model(model: Path, value: str, default: str) -> Path:
    selected = Path(value or default).expanduser()
    return selected.resolve() if selected.is_absolute() else (model / selected).resolve()


def resolve_artifacts(
    *,
    model_path: Optional[str] = None,
    router_checkpoint: Optional[str] = None,
    lora_adapter: Optional[str] = None,
    siglip2_checkpoint: Optional[str] = None,
) -> ArtifactPaths:
    """Resolve model paths."""
    model = _path(model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH)
    manifest = _manifest(model)

    router_value = router_checkpoint or os.environ.get("ROUTER_CHECKPOINT")
    router = (
        _path(router_value)
        if router_value
        else _relative_to_model(model, manifest.get("router_checkpoint", ""), "router/router.pt")
    )

    adapter_value = lora_adapter or os.environ.get("LORA_ADAPTER")
    adapter = (
        _path(adapter_value)
        if adapter_value
        else _relative_to_model(model, manifest.get("lora_adapter", ""), "adapter")
    )

    siglip_value = siglip2_checkpoint or os.environ.get("SIGLIP2_MODEL_PATH")
    if siglip_value:
        siglip2 = _path(siglip_value)
    elif manifest.get("semantic_encoder"):
        siglip2 = _relative_to_model(model, manifest["semantic_encoder"], "")
    else:
        siglip2 = DEFAULT_SIGLIP2_PATH.resolve()

    return ArtifactPaths(model=model, router=router, adapter=adapter, siglip2=siglip2)


def validate_inference_artifacts(paths: ArtifactPaths) -> None:
    """Check required inference files."""
    required = {
        "QueryStreamPlus-7B model configuration": paths.model / "config.json",
        "QueryStream++ router": paths.router,
        "LoRA adapter configuration": paths.adapter / "adapter_config.json",
        "LoRA adapter weights": paths.adapter / "adapter_model.safetensors",
        "SigLIP2 model configuration": paths.siglip2 / "config.json",
    }
    missing = [(label, path) for label, path in required.items() if not path.is_file()]
    if missing:
        details = "\n".join(f"  - {label}: {path}" for label, path in missing)
        raise FileNotFoundError(
            "QueryStream++ inference artifacts are incomplete. Expected the published "
            f"models under {REPOSITORY_ROOT / 'models'}:\n{details}"
        )
