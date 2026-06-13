from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from core.data import AntiUAVSequence
from core.metrics import SequenceMetrics
from core.motion import MotionCandidateDetector, detections_in_roi
from core.tracker import OcclusionAwareSingleTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the lightweight UAV tracker on multiple sequences.")
    parser.add_argument("--split-dir", type=Path, default=Path().cwd().parent / "anti_uav_project/Anti-UAV-RGBT/val")
    parser.add_argument("--modality", default="visible", choices=["visible", "infrared"])
    parser.add_argument("--max-sequences", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--csv", type=Path, default=Path("outputs/lightweight_benchmark.csv"))
    parser.add_argument("--device", default="auto", help="Patch classifier device: auto, cuda, or cpu.")
    parser.add_argument("--patch-model", type=Path, default=Path().cwd() / "weights/tiny_patch_cnn.pt", help="Optional tiny patch CNN checkpoint.")
    parser.add_argument("--patch-threshold", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def evaluate_sequence(
    sequence_dir: Path,
    modality: str,
    max_frames: int,
    iou_threshold: float,
    patch_classifier,
    top_k: int,
) -> dict[str, float | str]:
    sequence = AntiUAVSequence.from_dir(sequence_dir, modality)
    detector = MotionCandidateDetector()
    tracker = OcclusionAwareSingleTracker()
    metrics = SequenceMetrics()

    start = time.perf_counter()
    for record in sequence.iter_frames(max_frames=max_frames):
        height, width = record.image.shape[:2]
        all_detections = detector.detect(record.image, roi=None)
        roi = tracker.search_roi(width, height)
        if patch_classifier is not None:
            global_cnn_detections = patch_classifier.filter(record.image, all_detections, top_k=top_k)
        else:
            global_cnn_detections = all_detections
        local_detections = detections_in_roi(global_cnn_detections, roi)
        state = tracker.update(
            primary_detections=local_detections,
            fallback_detections=global_cnn_detections,
            frame=record.image,
        )
        metrics.update(
            exists=record.exists,
            gt_box=record.gt_box,
            pred_box=state.box,
            status=state.status,
            event=state.event,
            iou_threshold=iou_threshold,
        )

    elapsed = max(1e-9, time.perf_counter() - start)
    result = metrics.as_dict()
    result["fps"] = result["frames"] / elapsed
    result["sequence"] = sequence.name
    return result


def mean_numeric(rows: list[dict[str, float | str]]) -> dict[str, float]:
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, float)]
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in numeric_keys
    }

def main() -> None:
    args = parse_args()
    patch_classifier = None
    if args.patch_model is not None:
        from core.patch_classifier import PatchClassifier

        patch_classifier = PatchClassifier(
            checkpoint_path=args.patch_model,
            device=args.device,
            threshold=args.patch_threshold,
        )
        print(f"patch_classifier_device: {patch_classifier.device}")

    sequence_dirs = [
        path for path in sorted(args.split_dir.iterdir())
        if path.is_dir() and (path / f"{args.modality}.mp4").exists()
    ][: args.max_sequences]

    if not sequence_dirs:
        raise RuntimeError(f"No sequences found under {args.split_dir}")

    rows = [
        evaluate_sequence(
            path,
            args.modality,
            args.max_frames,
            args.iou_threshold,
            patch_classifier,
            args.top_k,
        )
        for path in sequence_dirs
    ]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = mean_numeric(rows)
    print(f"wrote: {args.csv}")
    print(f"sequences: {len(rows)}")
    for key, value in summary.items():
        print(f"mean_{key}: {value:.4f}")


if __name__ == "__main__":
    main()
