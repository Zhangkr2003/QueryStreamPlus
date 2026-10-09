"""Cache for QueryStream++ visual encoders."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch


def frame_content_digest(frames) -> str:
    """Hash resized RGB frames."""
    digest = hashlib.sha256()
    for frame in frames:
        array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class UniversalVisualFeatureCache:
    """Cache visual backbone and SigLIP2 features."""

    def __init__(self, root: str, encoder_signature: dict, mode: str = "readwrite") -> None:
        if mode not in {"readwrite", "readonly", "refresh"}:
            raise ValueError("feature cache mode must be readwrite, readonly, or refresh")
        self.root = Path(root).expanduser()
        self.encoder_signature = encoder_signature
        self.mode = mode
        self.hits = 0
        self.misses = 0

    def _identity(self, frames):
        identity = {
            "content_digest": frame_content_digest(frames),
            "encoder_signature": self.encoder_signature,
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest(), identity

    def _path(self, digest: str) -> Path:
        return self.root / "features" / digest[:2] / f"{digest}.pt"

    def load(self, frames):
        digest, identity = self._identity(frames)
        path = self._path(digest)
        if self.mode == "refresh" or not path.is_file():
            self.misses += 1
            return None, digest
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["identity"] != identity:
                raise ValueError("identity mismatch")
            self.hits += 1
            return payload, digest
        except Exception as exc:
            warnings.warn(f"Ignoring invalid visual feature cache entry {path}: {exc}")
            self.misses += 1
            return None, digest

    def store(self, frames, video_grid_thw, qwen_values, aligned_semantic_tokens):
        if self.mode == "readonly":
            return None
        digest, identity = self._identity(frames)
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "identity": identity,
            "video_grid_thw": video_grid_thw.detach().cpu().to(torch.int64),
            "qwen_values": qwen_values.detach().cpu().to(torch.float16),
            "aligned_semantic_tokens": aligned_semantic_tokens.detach().cpu().to(torch.float16),
        }
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path
