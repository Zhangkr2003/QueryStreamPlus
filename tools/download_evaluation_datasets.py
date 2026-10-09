#!/usr/bin/env python3
"""Download QueryStream++ evaluation datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import string
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set


STREAMINGBENCH_REPO = "mjuicem/StreamingBench"
OVOBENCH_REPO = "JoeLeelyf/OVO-Bench"
OVOBENCH_ANNOTATION_URL = (
    "https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/main/data/ovo_bench_new.json"
)
VIDEOMME_REPO = "lmms-lab/Video-MME"
LONGVIDEOBENCH_REPO = "longvideobench/LongVideoBench"
SVBENCH_REPO = "yzy666/SVBench"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download official QueryStream++ evaluation datasets."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=package_root / "data" / "benchmarks",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("streamingbench", "ovobench", "videomme", "longvideobench", "svbench"),
        default=("streamingbench", "ovobench", "videomme", "longvideobench", "svbench"),
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--accept-licenses",
        action="store_true",
        help="Confirm that all selected dataset licenses and access terms were reviewed.",
    )
    parser.add_argument(
        "--delete-archives-after-extract",
        action="store_true",
        help="Remove downloaded archives after verification.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download or extract; verify the requested local datasets only.",
    )
    return parser.parse_args()


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install requirements.txt first."
        ) from exc
    return HfApi, hf_hub_download, snapshot_download


def normalized_relative(path: str) -> str:
    value = str(path).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe archive path: {path!r}")
    return pure.as_posix()


def atomic_copy(source, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def repo_files(api, repo_id: str, token: Optional[str]) -> List[str]:
    return list(api.list_repo_files(repo_id, repo_type="dataset", token=token))


def download_url(url: str, target: Path) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "QueryStreamPlus-dataset-helper"})
    with urllib.request.urlopen(request) as source:
        atomic_copy(source, target)
    return target


def streamingbench_sessions(annotation: Path) -> Set[str]:
    sessions: Set[str] = set()
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            question_id = str(row.get("question_id", ""))
            if not question_id:
                continue
            session = f"sample_{question_id.split('_')[-2]}"
            sessions.add(session)
    if not sessions:
        raise RuntimeError(f"No StreamingBench sessions were found in {annotation}.")
    return sessions


def extract_streamingbench_zip(archive: Path, video_root: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(normalized_relative(member.filename)).parts
            sample_index = next(
                (index for index, part in enumerate(parts) if part.startswith("sample_")),
                None,
            )
            if sample_index is None or PurePosixPath(*parts).suffix.lower() not in VIDEO_SUFFIXES:
                continue
            relative = PurePosixPath(*parts[sample_index:])
            target = video_root / Path(relative)
            if target.is_file() and target.stat().st_size > 0:
                continue
            with handle.open(member) as source:
                atomic_copy(source, target)
            extracted += 1
    return extracted


def verify_streamingbench(root: Path) -> None:
    annotation = root / "StreamingBench" / "Real_Time_Visual_Understanding.csv"
    if not annotation.is_file():
        raise RuntimeError(f"StreamingBench annotation is missing: {annotation}")
    video_root = root / "Real-Time Visual Understanding"
    sessions = streamingbench_sessions(annotation)
    missing = [
        session
        for session in sorted(sessions)
        if not (video_root / session / "video.mp4").is_file()
        or (video_root / session / "video.mp4").stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"StreamingBench is missing {len(missing)} official source videos; first: {missing[0]}"
        )
    print(f"[StreamingBench] verified {len(sessions)} official source videos.")


def download_streamingbench(
    output_root: Path,
    token: Optional[str],
    delete_archives: bool,
    verify_only: bool,
) -> None:
    root = output_root / "streamingbench"
    if verify_only:
        verify_streamingbench(root)
        return
    HfApi, hf_hub_download, _ = require_huggingface_hub()
    files = repo_files(HfApi(), STREAMINGBENCH_REPO, token)
    annotation_name = "StreamingBench/Real_Time_Visual_Understanding.csv"
    if annotation_name not in files:
        raise RuntimeError("The official StreamingBench annotation was not found.")
    hf_hub_download(
        repo_id=STREAMINGBENCH_REPO,
        repo_type="dataset",
        filename=annotation_name,
        local_dir=root,
        token=token,
    )
    try:
        verify_streamingbench(root)
        return
    except RuntimeError:
        pass
    archives = sorted(
        path
        for path in files
        if path.startswith("Real-Time Visual Understanding_") and path.endswith(".zip")
    )
    if not archives:
        raise RuntimeError("The official StreamingBench real-time video archives were not found.")
    archive_root = output_root / "_archives" / "streamingbench"
    for index, filename in enumerate(archives, 1):
        print(f"[StreamingBench] downloading archive {index}/{len(archives)}: {filename}")
        archive = Path(
            hf_hub_download(
                repo_id=STREAMINGBENCH_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=archive_root,
                token=token,
            )
        )
        count = extract_streamingbench_zip(archive, root / "Real-Time Visual Understanding")
        print(f"[StreamingBench] {filename}: materialized {count} new video files.")
        if delete_archives:
            archive.unlink(missing_ok=True)
    verify_streamingbench(root)


def ovobench_video_paths(annotation: Path) -> Set[str]:
    with annotation.open(encoding="utf-8") as handle:
        items = json.load(handle)
    paths = {
        normalized_relative(item["video"])
        for item in items
        if isinstance(item, dict) and item.get("video")
    }
    if not paths:
        raise RuntimeError(f"No OVO-Bench video paths were found in {annotation}.")
    return paths


def extract_ovobench_parts(
    parts: Sequence[Path], video_root: Path, requested_paths: Set[str]
) -> int:
    by_basename: Dict[str, List[str]] = {}
    for relative in requested_paths:
        by_basename.setdefault(PurePosixPath(relative).name, []).append(relative)
    missing = {
        relative
        for relative in requested_paths
        if not (video_root / Path(relative)).is_file()
    }
    if not missing:
        return 0
    extracted = 0
    stream = ConcatenatedReader(parts)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as handle:
            for member in handle:
                if not member.isfile():
                    continue
                archive_name = normalized_relative(member.name)
                candidates = by_basename.get(PurePosixPath(archive_name).name, [])
                relative = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate in missing
                        and (archive_name == candidate or archive_name.endswith("/" + candidate))
                    ),
                    None,
                )
                if relative is None:
                    continue
                source = handle.extractfile(member)
                if source is None:
                    continue
                with source:
                    atomic_copy(source, video_root / Path(relative))
                missing.remove(relative)
                extracted += 1
                if not missing:
                    break
    finally:
        stream.close()
    if missing:
        first = sorted(missing)[0]
        raise RuntimeError(
            f"OVO-Bench archives did not contain {len(missing)} source videos; first: {first}"
        )
    return extracted


def verify_ovobench(root: Path) -> None:
    annotation = root / "ovo_bench_new.json"
    if not annotation.is_file():
        raise RuntimeError(f"OVO-Bench annotation is missing: {annotation}")
    expected = ovobench_video_paths(annotation)
    missing = sorted(path for path in expected if not (root / Path(path)).is_file())
    if missing:
        raise RuntimeError(
            f"OVO-Bench is missing {len(missing)}/{len(expected)} source videos; first: {missing[0]}"
        )
    print(f"[OVO-Bench] verified all {len(expected)} source videos.")


def download_ovobench(
    output_root: Path,
    token: Optional[str],
    delete_archives: bool,
    verify_only: bool,
) -> None:
    root = output_root / "ovobench"
    if verify_only:
        verify_ovobench(root)
        return
    annotation = download_url(OVOBENCH_ANNOTATION_URL, root / "ovo_bench_new.json")
    requested = ovobench_video_paths(annotation)
    if all((root / Path(path)).is_file() for path in requested):
        verify_ovobench(root)
        return
    HfApi, hf_hub_download, _ = require_huggingface_hub()
    files = repo_files(HfApi(), OVOBENCH_REPO, token)
    part_names = sorted(path for path in files if path.startswith("src_videos.tar.part"))
    if not part_names:
        raise RuntimeError("The official OVO-Bench source-video archive parts were not found.")
    archive_root = output_root / "_archives" / "ovobench"
    local_parts = []
    for index, filename in enumerate(part_names, 1):
        print(f"[OVO-Bench] downloading part {index}/{len(part_names)}: {filename}")
        local_parts.append(
            Path(
                hf_hub_download(
                    repo_id=OVOBENCH_REPO,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=archive_root,
                    token=token,
                )
            )
        )
    count = extract_ovobench_parts(local_parts, root, requested)
    print(f"[OVO-Bench] materialized {count} new source videos.")
    verify_ovobench(root)
    if delete_archives:
        for part in local_parts:
            part.unlink(missing_ok=True)


def extract_videomme_zip(archive: Path, video_root: Path) -> int:
    """Extract videos as ``<videoID>.mp4``."""
    extracted = 0
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            member_path = PurePosixPath(normalized_relative(member.filename))
            if member_path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            target = video_root / f"{member_path.stem}.mp4"
            if target.is_file() and target.stat().st_size > 0:
                continue
            with handle.open(member) as source:
                atomic_copy(source, target)
            extracted += 1
    return extracted


def videomme_expected_ids(annotation: Path) -> Optional[Set[str]]:
    try:
        import pandas as pd
    except ImportError:
        return None
    frame = pd.read_parquet(annotation, columns=["videoID"])
    return {str(value) for value in frame["videoID"].tolist()}


def verify_videomme(root: Path) -> None:
    annotation = root / "test.parquet"
    if not annotation.is_file():
        raise RuntimeError(f"Video-MME annotation is missing: {annotation}")
    video_root = root / "videos"
    expected = videomme_expected_ids(annotation)
    if expected is None:
        count = sum(1 for path in video_root.glob("*.mp4") if path.stat().st_size > 0)
        if count < 900:
            raise RuntimeError(f"Video-MME has only {count}/900 non-empty videos.")
        print(f"[Video-MME] verified {count} videos (install pandas+pyarrow for ID-level checks).")
        return
    missing = sorted(video_id for video_id in expected if not (video_root / f"{video_id}.mp4").is_file())
    if missing:
        raise RuntimeError(f"Video-MME is missing {len(missing)} videos; first: {missing[0]}")
    print(f"[Video-MME] verified {len(expected)} unique videos and {len(expected) * 3} QA rows.")


def download_videomme(
    output_root: Path,
    token: Optional[str],
    delete_archives: bool,
    verify_only: bool,
) -> None:
    root = output_root / "video_mme"
    if verify_only:
        verify_videomme(root)
        return
    HfApi, hf_hub_download, _ = require_huggingface_hub()
    api = HfApi()
    files = repo_files(api, VIDEOMME_REPO, token)
    parquet_files = sorted(path for path in files if path.endswith(".parquet") and "test" in path)
    if not parquet_files:
        raise RuntimeError("The official Video-MME repository contains no test parquet.")
    annotation_source = Path(
        hf_hub_download(
            repo_id=VIDEOMME_REPO,
            repo_type="dataset",
            filename=parquet_files[0],
            local_dir=root,
            token=token,
        )
    )
    shutil.copy2(annotation_source, root / "test.parquet")

    archives = sorted(path for path in files if path.startswith("videos_chunked_") and path.endswith(".zip"))
    if not archives:
        raise RuntimeError("The official Video-MME repository contains no videos_chunked ZIPs.")
    archive_root = output_root / "_archives" / "video_mme"
    for index, filename in enumerate(archives, 1):
        print(f"[Video-MME] downloading archive {index}/{len(archives)}: {filename}")
        archive = Path(
            hf_hub_download(
                repo_id=VIDEOMME_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=archive_root,
                token=token,
            )
        )
        count = extract_videomme_zip(archive, root / "videos")
        print(f"[Video-MME] {filename}: materialized {count} new videos.")
        if delete_archives:
            archive.unlink(missing_ok=True)
    verify_videomme(root)


def iter_annotation_dicts(value) -> Iterable[dict]:
    if isinstance(value, dict):
        if "video_path" in value:
            yield value
        for child in value.values():
            yield from iter_annotation_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_annotation_dicts(child)


def longvideobench_video_paths(annotation: Path) -> Set[str]:
    with annotation.open(encoding="utf-8") as handle:
        data = json.load(handle)
    paths = {normalized_relative(item["video_path"]) for item in iter_annotation_dicts(data)}
    if not paths:
        raise RuntimeError(f"No video_path entries were found in {annotation}.")
    return paths


class ConcatenatedReader(io.RawIOBase):
    """Sequentially expose split binary files as one non-seekable stream."""

    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths)
        self.index = 0
        self.handle = None
        self.position = 0

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def _advance(self) -> bool:
        if self.handle is not None:
            self.handle.close()
        if self.index >= len(self.paths):
            self.handle = None
            return False
        self.handle = self.paths[self.index].open("rb")
        self.index += 1
        return True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        # Keep reads bounded for split archives.
        if size < 0:
            size = 8 * 1024 * 1024
        chunks = []
        remaining = size
        while remaining > 0:
            if self.handle is None and not self._advance():
                break
            chunk = self.handle.read(remaining)
            if chunk:
                chunks.append(chunk)
                self.position += len(chunk)
                remaining -= len(chunk)
                continue
            self.handle.close()
            self.handle = None
        return b"".join(chunks)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        super().close()


def extract_longvideobench_parts(
    parts: Sequence[Path], video_root: Path, requested_paths: Set[str]
) -> int:
    by_basename: Dict[str, str] = {}
    for relative in requested_paths:
        basename = PurePosixPath(relative).name
        previous = by_basename.setdefault(basename, relative)
        if previous != relative:
            raise RuntimeError(f"Ambiguous LongVideoBench basename: {basename}")

    missing = {
        relative
        for relative in requested_paths
        if not (video_root / Path(relative)).is_file()
    }
    if not missing:
        return 0
    extracted = 0
    stream = ConcatenatedReader(parts)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as handle:
            for member in handle:
                if not member.isfile():
                    continue
                archive_name = normalized_relative(member.name)
                if PurePosixPath(archive_name).suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                relative = by_basename.get(PurePosixPath(archive_name).name)
                if relative is None or relative not in missing:
                    continue
                source = handle.extractfile(member)
                if source is None:
                    continue
                with source:
                    atomic_copy(source, video_root / Path(relative))
                missing.remove(relative)
                extracted += 1
                if not missing:
                    break
    finally:
        stream.close()
    if missing:
        first = sorted(missing)[0]
        raise RuntimeError(
            f"LongVideoBench archives did not contain {len(missing)} validation videos; first: {first}"
        )
    return extracted


def default_lvb_part_names() -> List[str]:
    names = [f"videos.tar.part.a{letter}" for letter in string.ascii_lowercase]
    names.extend(f"videos.tar.part.b{letter}" for letter in "abcde")
    return names


def verify_longvideobench(root: Path) -> None:
    annotation = root / "lvb_val.json"
    if not annotation.is_file():
        raise RuntimeError(f"LongVideoBench validation annotation is missing: {annotation}")
    expected = longvideobench_video_paths(annotation)
    missing = sorted(path for path in expected if not (root / "videos" / Path(path)).is_file())
    if missing:
        raise RuntimeError(
            f"LongVideoBench is missing {len(missing)}/{len(expected)} validation videos; first: {missing[0]}"
        )
    print(f"[LongVideoBench] verified all {len(expected)} validation videos.")


def download_longvideobench(
    output_root: Path,
    token: Optional[str],
    delete_archives: bool,
    verify_only: bool,
) -> None:
    root = output_root / "longvideobench"
    if verify_only:
        verify_longvideobench(root)
        return
    if not token:
        raise RuntimeError(
            "LongVideoBench is gated. Accept its terms at "
            "https://huggingface.co/datasets/longvideobench/LongVideoBench and set HF_TOKEN."
        )
    HfApi, hf_hub_download, _ = require_huggingface_hub()
    api = HfApi()
    try:
        files = repo_files(api, LONGVIDEOBENCH_REPO, token)
    except Exception as exc:
        raise RuntimeError(
            "Cannot access LongVideoBench. Confirm that the HF account associated with "
            "HF_TOKEN accepted the dataset terms."
        ) from exc
    annotation = Path(
        hf_hub_download(
            repo_id=LONGVIDEOBENCH_REPO,
            repo_type="dataset",
            filename="lvb_val.json",
            local_dir=root,
            token=token,
        )
    )
    requested = longvideobench_video_paths(annotation)
    if all((root / "videos" / Path(path)).is_file() for path in requested):
        print(f"[LongVideoBench] all {len(requested)} validation videos already exist.")
        verify_longvideobench(root)
        return
    part_names = sorted(path for path in files if path.startswith("videos.tar.part."))
    if not part_names:
        part_names = default_lvb_part_names()
    archive_root = output_root / "_archives" / "LongVideoBench"
    local_parts = []
    for index, filename in enumerate(part_names, 1):
        print(f"[LongVideoBench] downloading part {index}/{len(part_names)}: {filename}")
        local_parts.append(
            Path(
                hf_hub_download(
                    repo_id=LONGVIDEOBENCH_REPO,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=archive_root,
                    token=token,
                )
            )
        )
    print(
        f"[LongVideoBench] streaming {len(local_parts)} tar parts; extracting only "
        f"{len(requested)} validation videos."
    )
    count = extract_longvideobench_parts(local_parts, root / "videos", requested)
    print(f"[LongVideoBench] materialized {count} new validation videos.")
    verify_longvideobench(root)
    if delete_archives:
        for part in local_parts:
            part.unlink(missing_ok=True)


def verify_svbench(root: Path) -> None:
    if not (root / "Meta" / "Meta_EN" / "meta_test.csv").is_file() and (root / "SVBench").is_dir():
        root = root / "SVBench"
    expected = (
        root / "Meta" / "Meta_EN" / "meta_test.csv",
        root / "Dialogue" / "Dialogue_EN",
        root / "Streaming" / "Streaming_EN",
        root / "Path",
        root / "Video",
    )
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(
            f"SVBench official release is incomplete; first missing path: {missing[0]}"
        )
    print(f"[SVBench] verified official Dialogue/Streaming layout at {root}.")


def download_svbench(
    output_root: Path,
    token: Optional[str],
    workers: int,
    verify_only: bool,
) -> None:
    root = output_root / "svbench"
    if not verify_only:
        _, _, snapshot_download = require_huggingface_hub()
        print("[SVBench] downloading the official Dialogue/Streaming release.")
        snapshot_download(
            repo_id=SVBENCH_REPO,
            repo_type="dataset",
            local_dir=root,
            token=token,
            max_workers=workers,
        )
    verify_svbench(root)


def print_layout(output_root: Path) -> None:
    print("\nEvaluation data layout:")
    print(f"  StreamingBench root: {output_root / 'streamingbench'}")
    print(f"  OVO-Bench root     : {output_root / 'ovobench'}")
    print(f"  Video-MME parquet : {output_root / 'video_mme' / 'test.parquet'}")
    print(f"  Video-MME videos  : {output_root / 'video_mme' / 'videos'}")
    print(f"  LongVideoBench HF_HOME: {output_root}")
    print(f"  LongVideoBench videos : {output_root / 'longvideobench' / 'videos'}")
    print(f"  SVBench root      : {output_root / 'svbench'}")


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if not args.verify_only and not args.accept_licenses:
        raise SystemExit(
            "Review the official dataset licenses/access terms, then rerun with --accept-licenses."
        )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = list(dict.fromkeys(args.datasets))
    for dataset in selected:
        if dataset == "streamingbench":
            download_streamingbench(
                output_root,
                args.hf_token,
                args.delete_archives_after_extract,
                args.verify_only,
            )
        elif dataset == "ovobench":
            download_ovobench(
                output_root,
                args.hf_token,
                args.delete_archives_after_extract,
                args.verify_only,
            )
        elif dataset == "videomme":
            download_videomme(
                output_root,
                args.hf_token,
                args.delete_archives_after_extract,
                args.verify_only,
            )
        elif dataset == "longvideobench":
            download_longvideobench(
                output_root,
                args.hf_token,
                args.delete_archives_after_extract,
                args.verify_only,
            )
        elif dataset == "svbench":
            download_svbench(
                output_root,
                args.hf_token,
                args.workers,
                args.verify_only,
            )
    print_layout(output_root)


if __name__ == "__main__":
    main()
