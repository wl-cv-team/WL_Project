"""Audit archive plate-detection labels and write a JSON report plus a markdown summary."""

import argparse
import json
from pathlib import Path

from wl_snatch.label_audit import (
    ISSUE_KINDS,
    PLATE_PAIRS as PAIRS,
    filename_overlap,
    list_images,
    summarize_directory,
)


def _fmt(d: dict) -> str:
    def f(v):
        return "N/A" if v is None else f"{v:.4g}"

    return f"min={f(d['min'])} median={f(d['median'])} max={f(d['max'])}"


def build_markdown(result: dict) -> str:
    lines = ["# Plate label audit", ""]
    lines += ["Box sizes are normalized (fraction of image width or height).", ""]
    lines += ["| directory | images | labeled boxes | " + " | ".join(ISSUE_KINDS) + " |"]
    lines += ["|---|---|---|" + "---|" * len(ISSUE_KINDS)]
    for name, s in result["directories"].items():
        counts = " | ".join(str(s["issues_by_kind"].get(k, 0)) for k in ISSUE_KINDS)
        lines.append(f"| {name} | {s['images']} | {s['labeled_boxes']} | {counts} |")
    for name, s in result["directories"].items():
        lines += ["", f"## {name}", ""]
        lines.append(f"- box width (normalized): {_fmt(s['box_width_normalized'])}")
        lines.append(f"- box height (normalized): {_fmt(s['box_height_normalized'])}")
        lines.append(f"- boxes per image: {_fmt(s['boxes_per_image'])}")
        lines.append(f"- boxes per image histogram (boxes: images): {s['boxes_per_image_histogram']}")
        lines.append(f"- videos: {len(s['frames_per_video'])}, persons: {len(s['frames_per_person'])}")
        lines.append(f"- frames per person_id: {s['frames_per_person']}")
        lines.append(f"- frames per video_id: {s['frames_per_video']}")
        if s["unparsed_names"]:
            lines.append(f"- file names without video id / person id: {s['unparsed_names']}")
        if s["issues"]:
            lines.append("- issues:")
            lines += [f"  - {i['image']} [{i['kind']}] {i['detail']}" for i in s["issues"]]
    lines += ["", "## Cross-directory facts", ""]
    ov = result["overlap_plate_dataset_vs_v2"]
    lines.append(f"- file names in both plate_dataset and plate_dataset_v2 (train or validation): {len(ov)}")
    if ov:
        lines.append(f"  - {ov}")
    shared = result["video_ids_in_both_v2_train_and_validation"]
    lines.append(f"- video_id in both v2 train and v2 validation: {len(shared)}")
    if shared:
        lines.append(f"  - {shared}")
    shared_p = result["person_ids_in_both_v2_train_and_validation"]
    lines.append(f"- person_id in both v2 train and v2 validation: {len(shared_p)}")
    if shared_p:
        lines.append(f"  - {shared_p}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/label_audit"))
    args = parser.parse_args()

    allowed = frozenset({0})
    directories = {}
    names = {}
    for name, (img, lbl) in PAIRS.items():
        directories[name] = summarize_directory(args.data_dir / img, args.data_dir / lbl, allowed)
        names[name] = [p.name for p in list_images(args.data_dir / img)]
    v2_names = names["plate_dataset_v2/train"] + names["plate_dataset_v2/validation"]
    train_videos = set(directories["plate_dataset_v2/train"]["frames_per_video"])
    val_videos = set(directories["plate_dataset_v2/validation"]["frames_per_video"])
    train_persons = set(directories["plate_dataset_v2/train"]["frames_per_person"])
    val_persons = set(directories["plate_dataset_v2/validation"]["frames_per_person"])
    result = {
        "allowed_classes": sorted(allowed),
        "directories": directories,
        "overlap_plate_dataset_vs_v2": filename_overlap(names["plate_dataset"], v2_names),
        "video_ids_in_both_v2_train_and_validation": sorted(train_videos & val_videos),
        "person_ids_in_both_v2_train_and_validation": sorted(train_persons & val_persons),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "label_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = build_markdown(result)
    (args.out_dir / "label_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {args.out_dir / 'label_audit.json'} and {args.out_dir / 'label_audit.md'}")


if __name__ == "__main__":
    main()
