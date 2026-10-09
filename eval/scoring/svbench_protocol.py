"""Load official SVBench Dialogue and Streaming annotations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def resolve_annotation_root(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.name == "meta_test.csv":
            candidate = candidate.parents[2]
        else:
            raise ValueError(
                "Official SVBench evaluation expects the annotation root or Meta/Meta_EN/meta_test.csv."
            )
    roots = [candidate, candidate / "SVBench"]
    for root in roots:
        if (
            (root / "Meta" / "Meta_EN" / "meta_test.csv").is_file()
            and (root / "Dialogue" / "Dialogue_EN").is_dir()
            and (root / "Streaming" / "Streaming_EN").is_dir()
        ):
            return root
    raise FileNotFoundError(
        f"Cannot locate official SVBench annotations below {candidate}. Expected "
        "Meta/Meta_EN, Dialogue/Dialogue_EN, Streaming/Streaming_EN, and Path."
    )


def _test_meta(root: Path) -> List[dict]:
    path = root / "Meta" / "Meta_EN" / "meta_test.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if str(row.get("Sort_of_Set", "Test")).lower() == "test"]
    rows.sort(key=lambda row: str(row["Video_Name"]))
    return rows


def _video_path(video_dir: str, video_id: str, meta: dict) -> str:
    root = Path(video_dir).expanduser().resolve()
    annotated = Path(str(meta.get("Path_of_Video", "")))
    # Remove a redundant dataset prefix when present.
    annotated_without_dataset = annotated
    if annotated.parts and annotated.parts[0].lower() == "svbench":
        annotated_without_dataset = Path(*annotated.parts[1:])
    candidates = [
        root / annotated,
        root / annotated_without_dataset,
        root / "Video" / annotated.name,
        root / annotated.name,
        root / "SVBench" / annotated_without_dataset,
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        # Accept an unambiguous container rename.
        matches = sorted((root / "Video").glob(f"{video_id}.*"))
        if len(matches) == 1 and matches[0].is_file():
            path = matches[0]
        else:
            expected = root / annotated_without_dataset
            raise FileNotFoundError(
                f"Missing official SVBench video {video_id}. Meta expects {expected}."
            )
    return str(path.resolve())


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dialogue_rows(annotation_root: str, video_dir: str) -> Iterable[dict]:
    root = resolve_annotation_root(annotation_root)
    for meta in _test_meta(root):
        video_id = str(meta["Video_Name"])
        video = _video_path(video_dir, video_id, meta)
        chains = _load_json(root / "Dialogue" / "Dialogue_EN" / f"{video_id}.json")
        sequence_index = 0
        for chain_index, entry in enumerate(chains):
            chain = entry.get("chain", {})
            questions = list(chain.get("questions", []))
            answers = list(chain.get("answers", []))
            if not questions or len(questions) != len(answers):
                raise ValueError(
                    f"Invalid Dialogue chain {video_id}:{chain_index}: question/answer mismatch."
                )
            start = float(entry.get("qac_timestamps_start", 0.0))
            end = float(entry.get("qac_timestamps_end", start))
            for question_index, (question, answer) in enumerate(zip(questions, answers)):
                yield {
                    "sample_id": (
                        f"svbench_dialogue_{video_id}_c{chain_index:04d}_q{question_index:02d}"
                    ),
                    "session_id": f"svbench_dialogue_{video_id}",
                    "video": video,
                    "visible_end": end,
                    "question": str(question),
                    "options": None,
                    "answer": str(answer),
                    "sequence_index": sequence_index,
                    "reuse_visible_state": question_index > 0,
                    "metadata": {
                        "source_video_id": video_id,
                        "official_split": "Test",
                        "svbench_eval_mode": "dialogue",
                        "chain_index": chain_index,
                        "question_index": question_index,
                        "turn_index": sequence_index,
                        "timestamp_start": start,
                        "timestamp_end": end,
                        "source_dataset": "SVBench",
                    },
                }
                sequence_index += 1


def _transition_relationships(con_entries: Sequence[dict]) -> Dict[Tuple[int, int, int, int], list]:
    relationships: Dict[Tuple[int, int, int, int], list] = {}
    for chain_index, entry in enumerate(con_entries):
        relation = entry.get("relationship", {})
        before = [int(value) for value in relation.get("chainBefore", [])]
        after = [int(value) for value in relation.get("chainAfter", [])]
        labels = list(relation.get("relationship", []))
        for before_index in before:
            for after_index in after:
                relationships[(chain_index, before_index, chain_index + 1, after_index)] = labels
    return relationships


def streaming_rows(annotation_root: str, video_dir: str) -> Iterable[dict]:
    root = resolve_annotation_root(annotation_root)
    for meta in _test_meta(root):
        video_id = str(meta["Video_Name"])
        video = _video_path(video_dir, video_id, meta)
        paths = _load_json(root / "Streaming" / "Streaming_EN" / f"{video_id}.json")
        coordinate_path = root / "Path" / f"{video_id}.json"
        coordinates = _load_json(coordinate_path).get("Paths", [])
        if len(paths) != len(coordinates):
            raise ValueError(
                f"SVBench Streaming/Path count mismatch for {video_id}: "
                f"{len(paths)} versus {len(coordinates)}."
            )
        con_path = root / "Con" / "Con_EN" / f"{video_id}.json"
        relation_map = _transition_relationships(_load_json(con_path)) if con_path.is_file() else {}
        for path_index, (path, path_coordinates) in enumerate(zip(paths, coordinates)):
            questions = list(path.get("questions", []))
            answers = list(path.get("answers", []))
            timestamps = list(path.get("timestamps", []))
            if not (len(questions) == len(answers) == len(timestamps) == len(path_coordinates)):
                raise ValueError(
                    f"Invalid Streaming path {video_id}:{path_index}: field lengths differ."
                )
            previous_coordinate: Optional[Tuple[int, int]] = None
            previous_end: Optional[float] = None
            for sequence_index, (question, answer, timestamp, coordinate) in enumerate(
                zip(questions, answers, timestamps, path_coordinates)
            ):
                if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                    raise ValueError(
                        f"Invalid timestamp in Streaming path {video_id}:{path_index}:{sequence_index}."
                    )
                start, end = float(timestamp[0]), float(timestamp[1])
                chain_index, question_index = int(coordinate[0]), int(coordinate[1])
                relationship = []
                if previous_coordinate is not None:
                    relationship = relation_map.get(
                        (
                            previous_coordinate[0],
                            previous_coordinate[1],
                            chain_index,
                            question_index,
                        ),
                        [],
                    )
                yield {
                    "sample_id": (
                        f"svbench_streaming_{video_id}_p{path_index:02d}_t{sequence_index:04d}"
                    ),
                    "session_id": f"svbench_streaming_{video_id}_p{path_index:02d}",
                    "video": video,
                    "visible_end": end,
                    "question": str(question),
                    "options": None,
                    "answer": str(answer),
                    "sequence_index": sequence_index,
                    "reuse_visible_state": previous_end is not None and abs(end - previous_end) < 1e-6,
                    "metadata": {
                        "source_video_id": video_id,
                        "official_split": "Test",
                        "svbench_eval_mode": "streaming",
                        "path_index": path_index,
                        "chain_index": chain_index,
                        "question_index": question_index,
                        "turn_index": sequence_index,
                        "timestamp_start": start,
                        "timestamp_end": end,
                        "relationship_from_previous": relationship,
                        "source_dataset": "SVBench",
                    },
                }
                previous_coordinate = (chain_index, question_index)
                previous_end = end


def official_rows(mode: str, annotation_root: str, video_dir: str) -> Iterable[dict]:
    if mode == "dialogue":
        return dialogue_rows(annotation_root, video_dir)
    if mode == "streaming":
        return streaming_rows(annotation_root, video_dir)
    raise ValueError(f"Unsupported official SVBench mode: {mode}")
