#!/usr/bin/env python3
"""Download the checkpoints used by QueryStream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


DEFAULT_TIMECHAT_REPO = "wyccccc/TimeChatOnline-7B"
DEFAULT_QWEN_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_TAG = "openai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download QueryStream backbone and OpenCLIP weights.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        choices=("timechat", "qwen2.5-vl", "both"),
        default="timechat",
        help="Video-language backbone to download (default: timechat).",
    )
    parser.add_argument("--timechat-repo", default=DEFAULT_TIMECHAT_REPO)
    parser.add_argument("--timechat-revision", default="main")
    parser.add_argument("--qwen-repo", default=DEFAULT_QWEN_REPO)
    parser.add_argument("--qwen-revision", default="main")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-tag", default=DEFAULT_CLIP_TAG)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-openclip", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_backbone(path: Path, name: str) -> dict:
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]
    missing = [filename for filename in required if not (path / filename).is_file()]
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    if not shards:
        missing.append("model-*-of-*.safetensors")
    if missing:
        raise RuntimeError(f"Incomplete {name} checkpoint at {path}: {missing}")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    indexed_shards = sorted(set(index.get("weight_map", {}).values()))
    missing_shards = [filename for filename in indexed_shards if not (path / filename).is_file()]
    if missing_shards:
        raise RuntimeError(f"{name} index references missing shards: {missing_shards}")
    return {
        "path": str(path.resolve()),
        "shards": len(shards),
        "bytes": sum(item.stat().st_size for item in shards),
    }


def download_backbone(
    args: argparse.Namespace,
    target: Path,
    name: str,
    repo_id: str,
    revision: str,
) -> dict:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the repository requirements.txt first.") from exc
    print(f"[{name}] downloading {repo_id}@{revision} to {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=target,
        token=args.hf_token,
        max_workers=args.workers,
        ignore_patterns=("*.msgpack", "*.h5", "*.ot"),
    )
    result = verify_backbone(target, name)
    result.update(
        {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": HfApi().model_info(
                repo_id, revision=revision, token=args.hf_token
            ).sha,
        }
    )
    return result


def verify_openclip(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"OpenCLIP checkpoint is missing or unexpectedly small: {path}")
    checkpoint_format = detect_checkpoint_format(path)
    expected_suffix = ".safetensors" if checkpoint_format == "safetensors" else ".pt"
    if path.suffix != expected_suffix:
        raise RuntimeError(
            f"OpenCLIP checkpoint content is {checkpoint_format}, but its suffix is {path.suffix}: {path}. "
            f"Rename it to {path.with_suffix(expected_suffix).name}."
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "checkpoint_format": checkpoint_format,
    }


def detect_checkpoint_format(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(9)
    if len(prefix) == 9:
        header_length = int.from_bytes(prefix[:8], byteorder="little", signed=False)
        if 0 < header_length < min(path.stat().st_size - 8, 100 * 1024 * 1024) and prefix[8:9] == b"{":
            return "safetensors"
    return "pytorch"


def download_openclip(args: argparse.Namespace, target_base: Path) -> dict:
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("Install the repository requirements.txt first.") from exc
    config = open_clip.get_pretrained_cfg(args.clip_model, args.clip_tag)
    if not config:
        raise ValueError(f"Unknown OpenCLIP pair: {args.clip_model}/{args.clip_tag}")
    cache_dir = args.output_root / ".download_cache" / "openclip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OpenCLIP] downloading {args.clip_model}/{args.clip_tag}")
    downloaded = Path(open_clip.download_pretrained(config, cache_dir=str(cache_dir)))
    if not downloaded.is_file():
        raise RuntimeError("OpenCLIP downloader did not return a checkpoint file.")
    checkpoint_format = detect_checkpoint_format(downloaded)
    target = target_base.with_suffix(".safetensors" if checkpoint_format == "safetensors" else ".pt")
    print(f"[OpenCLIP] storing {checkpoint_format} checkpoint at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or sha256(target) != sha256(downloaded):
        partial = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(downloaded, partial)
        partial.replace(target)
    result = verify_openclip(target)
    result.update(
        {
            "model": args.clip_model,
            "pretrained_tag": args.clip_tag,
            "source_hf_hub": config.get("hf_hub"),
            "source_url": config.get("url"),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    timechat_target = args.output_root / "TimeChatOnline-7B"
    qwen_target = args.output_root / "Qwen2.5-VL-7B-Instruct"
    clip_target_base = args.output_root / "openclip" / "ViT-L-14-openai"
    manifest = {"backbones": {}}
    if args.backbone in {"timechat", "both"}:
        manifest["backbones"]["timechat"] = (
            verify_backbone(timechat_target, "TimeChat-Online-7B")
            if args.verify_only
            else download_backbone(
                args,
                timechat_target,
                "TimeChat-Online-7B",
                args.timechat_repo,
                args.timechat_revision,
            )
        )
    if args.backbone in {"qwen2.5-vl", "both"}:
        manifest["backbones"]["qwen2.5-vl"] = (
            verify_backbone(qwen_target, "Qwen2.5-VL-7B-Instruct")
            if args.verify_only
            else download_backbone(
                args,
                qwen_target,
                "Qwen2.5-VL-7B-Instruct",
                args.qwen_repo,
                args.qwen_revision,
            )
        )
    if not args.skip_openclip:
        if args.verify_only:
            candidates = [clip_target_base.with_suffix(".safetensors"), clip_target_base.with_suffix(".pt")]
            clip_target = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            manifest["openclip"] = verify_openclip(clip_target)
        else:
            manifest["openclip"] = download_openclip(args, clip_target_base)
    manifest_path = args.output_root / "model_manifest.json"
    if not args.verify_only:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if "timechat" in manifest["backbones"]:
        print(f"export TIMECHAT_MODEL_PATH={timechat_target.resolve()}")
    if "qwen2.5-vl" in manifest["backbones"]:
        print(f"export QWEN_MODEL_PATH={qwen_target.resolve()}")
    selected_target = qwen_target if args.backbone == "qwen2.5-vl" else timechat_target
    print(f"export MODEL_PATH={selected_target.resolve()}")
    if "openclip" in manifest:
        print(f"export CLIP_PRETRAINED={manifest['openclip']['path']}")


if __name__ == "__main__":
    main()
