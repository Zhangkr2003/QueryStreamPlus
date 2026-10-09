#!/usr/bin/env python3
"""Apply the QueryStream++ adapter to the pinned lmms-eval checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parent
EXPECTED_COMMIT = "f7a6d6bf6b1622e08ebf5336a48f98e69d503ea6"
MODEL_ENTRY = '    "qwen2_5_vl_querystream_plus": "Qwen2_5_VL_QueryStreamPlus",\n'
MODEL_ANCHOR = '    "qwen2_5_vl": "Qwen2_5_VL",\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lmms_eval_checkout", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the patch is installed without changing the checkout.",
    )
    return parser.parse_args()


def git_head(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Not a Git checkout: {checkout}")
    return result.stdout.strip()


def expected_files(checkout: Path) -> tuple[Path, Path, Path]:
    package = checkout / "lmms_eval"
    registry = package / "models" / "__init__.py"
    model = package / "models" / "qwen2_5_vl_querystream_plus.py"
    task = package / "tasks" / "longvideobench"
    if not registry.is_file() or not task.is_dir():
        raise RuntimeError(
            "The checkout does not have the lmms-eval v0.3.5 package layout: "
            f"{checkout}"
        )
    return registry, model, task


def same_bytes(left: Path, right: Path) -> bool:
    return left.is_file() and left.read_bytes() == right.read_bytes()


def verify_installation(checkout: Path) -> list[str]:
    registry, model, task = expected_files(checkout)
    errors = []
    if MODEL_ENTRY.strip() not in registry.read_text(encoding="utf-8"):
        errors.append("QueryStream++ model is not registered")
    copies = (
        (PATCH_ROOT / "models" / model.name, model),
        (PATCH_ROOT / "tasks" / "longvideobench" / "utils.py", task / "utils.py"),
        (
            PATCH_ROOT / "tasks" / "longvideobench" / "longvideobench_val_v.yaml",
            task / "longvideobench_val_v.yaml",
        ),
    )
    for source, destination in copies:
        if not same_bytes(source, destination):
            errors.append(f"Patch file is missing or differs: {destination}")
    return errors


def install(checkout: Path) -> None:
    registry, model, task = expected_files(checkout)
    registry_text = registry.read_text(encoding="utf-8")
    if MODEL_ENTRY.strip() not in registry_text:
        if MODEL_ANCHOR not in registry_text:
            raise RuntimeError("Cannot find the qwen2_5_vl registry entry to patch")
        registry.write_text(
            registry_text.replace(MODEL_ANCHOR, MODEL_ANCHOR + MODEL_ENTRY, 1),
            encoding="utf-8",
        )
    shutil.copy2(PATCH_ROOT / "models" / model.name, model)
    shutil.copy2(PATCH_ROOT / "tasks" / "longvideobench" / "utils.py", task / "utils.py")
    shutil.copy2(
        PATCH_ROOT / "tasks" / "longvideobench" / "longvideobench_val_v.yaml",
        task / "longvideobench_val_v.yaml",
    )


def main() -> int:
    args = parse_args()
    checkout = args.lmms_eval_checkout.expanduser().resolve()
    head = git_head(checkout)
    if head != EXPECTED_COMMIT:
        raise RuntimeError(
            f"Expected lmms-eval {EXPECTED_COMMIT}, but {checkout} is at {head}. "
            "Check out the pinned commit before applying this patch."
        )
    if not args.check:
        install(checkout)
    errors = verify_installation(checkout)
    if errors:
        raise RuntimeError("; ".join(errors))
    action = "verified" if args.check else "installed and verified"
    print(f"QueryStream++ lmms-eval patch {action} at {head}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
