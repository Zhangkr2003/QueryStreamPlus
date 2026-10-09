"""Lossless cache for uniformly sampled and resized benchmark frames."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class VideoFrameCache:
    def __init__(self, root: str, mode: str = "readwrite") -> None:
        if mode not in {"readwrite", "readonly", "refresh"}:
            raise ValueError("frame cache mode must be readwrite, readonly, or refresh")
        self.root = Path(root).expanduser()
        self.mode = mode
        self.hits = 0
        self.misses = 0

    def _identity(self, video_path: str, parameters: dict):
        path = Path(video_path).expanduser().resolve()
        stat = path.stat()
        identity = {
            "video_path": str(path),
            "video_size": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "sampling": parameters,
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest(), identity

    def _path(self, digest: str) -> Path:
        return self.root / "frames" / digest[:2] / f"{digest}.pt"

    @staticmethod
    def _pack(frames) -> torch.Tensor:
        values = [torch.from_numpy(np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()) for frame in frames]
        if not values:
            raise ValueError("cannot cache an empty frame sequence")
        return torch.stack(values)

    @staticmethod
    def _unpack(value: torch.Tensor):
        if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != torch.uint8:
            raise ValueError("invalid cached frame tensor")
        return [Image.fromarray(frame.numpy(), mode="RGB") for frame in value]

    def load_or_compute(self, video_path: str, parameters: dict, decoder):
        digest, identity = self._identity(video_path, parameters)
        path = self._path(digest)
        if self.mode != "refresh" and path.is_file():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["identity"] != identity:
                    raise ValueError("identity mismatch")
                self.hits += 1
                return self._unpack(payload["frames"]), True
            except Exception as exc:
                warnings.warn(f"Ignoring invalid frame cache entry {path}: {exc}")
        self.misses += 1
        frames = decoder()
        if self.mode != "readonly":
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".tmp.{os.getpid()}")
            torch.save({"identity": identity, "frames": self._pack(frames)}, temporary)
            os.replace(temporary, path)
        return frames, False
