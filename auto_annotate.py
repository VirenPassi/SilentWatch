"""Create synthetic motion-onset ToA labels from SPHAR videos.

These labels mark the first large visual change, not a human-verified action
onset. They are useful for exercising the dual-head pipeline, but must not be
reported as ground truth in a paper without manual validation.

Example:
    python auto_annotate.py --input-dir D:/SPHAR-subset --output-csv D:/toa_auto.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def detect_motion_onset(
    video_path: Path,
    threshold: float,
    resize_width: int = 160,
) -> tuple[int, float, int]:
    """Return (source-frame index, delta score, frame count) for one video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    previous = None
    frame_index = 0
    frame_count = 0
    onset_frame = 0
    onset_score = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resize_height = max(1, round(height * resize_width / width))
            gray = cv2.resize(gray, (resize_width, resize_height))

            if previous is not None:
                # Mean absolute grayscale change is normalized to [0, 255].
                score = float(cv2.absdiff(gray, previous).mean())
                if score >= threshold:
                    onset_frame = frame_index
                    onset_score = score
                    break
            previous = gray
            frame_index += 1
    finally:
        capture.release()

    return onset_frame, onset_score, frame_count


def annotate_directory(
    input_dir: Path,
    output_csv: Path,
    threshold: float,
) -> int:
    """Annotate videos recursively and write source-frame labels to CSV."""
    videos = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_path in videos:
        try:
            onset_frame, score, frames_examined = detect_motion_onset(
                video_path, threshold
            )
        except RuntimeError as error:
            print(f"Skipping {video_path}: {error}")
            continue

        # preprocess.py joins annotations by basename and converts source frames
        # into sampled-frame coordinates using dataset.sampling_rate.
        rows.append(
            {
                "video_name": video_path.name,
                "toa_frame": onset_frame,
                "annotation_type": "synthetic_motion_delta",
                "delta_score": f"{score:.6f}",
                "frames_examined": frames_examined,
            }
        )
        print(f"{video_path.name}: ToA source frame {onset_frame} (delta={score:.3f})")

    if not rows:
        raise RuntimeError("No readable videos were annotated")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Mean grayscale delta in [0, 255] that marks motion onset.",
    )
    args = parser.parse_args()
    count = annotate_directory(args.input_dir, args.output_csv, args.threshold)
    print(f"Wrote {count} synthetic ToA annotations to {args.output_csv}")


if __name__ == "__main__":
    main()
