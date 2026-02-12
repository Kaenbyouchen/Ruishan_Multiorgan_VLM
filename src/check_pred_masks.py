import argparse
import json
import os
from typing import Dict, List

import nibabel as nib
import numpy as np
from PIL import Image


ORGAN_MAP = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gall Bladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava",
    10: "Pancreas",
    11: "Right Adrenal Gland",
    12: "Left Adrenal Gland",
    13: "Duodenum",
}


def read_yaml_list(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            ids.append(value.lstrip("-").strip())
    return ids


def load_nifti(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata()).astype(np.int32)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_missed_slices(
    gt: np.ndarray, pred: np.ndarray, organ_id: int
) -> Dict[str, object]:
    gt_mask = gt == organ_id
    pred_mask = pred == organ_id
    intersection = int(np.logical_and(gt_mask, pred_mask).sum())
    gt_total = int(gt_mask.sum())
    total_slices = gt.shape[2]
    missed = []
    gt_present = 0
    pred_present = 0
    for idx in range(total_slices):
        gt_slice = gt_mask[:, :, idx]
        pred_slice = pred_mask[:, :, idx]
        gt_has = bool(gt_slice.any())
        pred_has = bool(pred_slice.any())
        if gt_has:
            gt_present += 1
        if pred_has:
            pred_present += 1
        if gt_has and not pred_has:
            missed.append(idx + 1)  # 1-based slice index
    missed_count = len(missed)
    missed_ratio = (missed_count / gt_present) if gt_present else 0.0
    overlap_ratio = (intersection / gt_total) if gt_total else 0.0
    return {
        "gt_slices": gt_present,
        "pred_slices": pred_present,
        "missed_slices": missed,
        "missed_count": missed_count,
        "missed_ratio": missed_ratio,
        "overlap_ratio": overlap_ratio,
        "intersection": intersection,
        "gt_total": gt_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check missed organ slices in pred masks.")
    parser.add_argument(
        "--gt_dir",
        default="data/gt_amos_mr",
        help="Ground truth NIfTI directory",
    )
    parser.add_argument(
        "--pred_dir",
        default="data/pred_amos_mr",
        help="Predicted NIfTI directory",
    )
    parser.add_argument(
        "--list_file",
        default="data/img_amos_mr/list/dataset.yaml",
        help="List of case IDs",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/check_pred",
        help="Output directory for JSON report",
    )
    parser.add_argument(
        "--write_heatmap",
        action="store_true",
        help="Write a heatmap PNG for missed ratios",
    )
    args = parser.parse_args()

    case_ids = read_yaml_list(args.list_file)
    ensure_dir(args.output_dir)

    summary = {
        "gt_dir": args.gt_dir,
        "pred_dir": args.pred_dir,
        "cases": [],
        "organ_totals": {},
    }

    organ_totals = {
        organ_name: {
            "gt_slices": 0,
            "pred_slices": 0,
            "missed_count": 0,
            "intersection": 0,
            "gt_total": 0,
        }
        for organ_name in ORGAN_MAP.values()
    }

    for case_id in case_ids:
        gt_path = os.path.join(args.gt_dir, f"{case_id}.nii.gz")
        pred_path = os.path.join(args.pred_dir, f"{case_id}.nii.gz")
        if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
            continue
        gt = load_nifti(gt_path)
        pred = load_nifti(pred_path)
        case_entry = {"case_id": case_id, "organs": {}}
        for organ_id, organ_name in ORGAN_MAP.items():
            stats = compute_missed_slices(gt, pred, organ_id)
            case_entry["organs"][organ_name] = stats
            organ_totals[organ_name]["gt_slices"] += stats["gt_slices"]
            organ_totals[organ_name]["pred_slices"] += stats["pred_slices"]
            organ_totals[organ_name]["missed_count"] += stats["missed_count"]
            organ_totals[organ_name]["intersection"] += stats["intersection"]
            organ_totals[organ_name]["gt_total"] += stats["gt_total"]
        summary["cases"].append(case_entry)

    for organ_name, totals in organ_totals.items():
        gt_slices = totals["gt_slices"]
        missed_count = totals["missed_count"]
        totals["missed_ratio"] = (missed_count / gt_slices) if gt_slices else 0.0
        totals["overlap_ratio"] = (
            totals["intersection"] / totals["gt_total"] if totals["gt_total"] else 0.0
        )
    summary["organ_totals"] = organ_totals

    output_path = os.path.join(args.output_dir, "check_pred_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Build matrix summary for visualization
    case_order = [c["case_id"] for c in summary["cases"]]
    organ_order = list(ORGAN_MAP.values())
    missed_ratio_matrix = []
    missed_count_matrix = []
    gt_slices_matrix = []
    overlap_ratio_matrix = []
    for case in summary["cases"]:
        ratios = []
        counts = []
        gts = []
        overlaps = []
        for organ_name in organ_order:
            stats = case["organs"][organ_name]
            ratios.append(stats["missed_ratio"])
            counts.append(stats["missed_count"])
            gts.append(stats["gt_slices"])
            overlaps.append(stats["overlap_ratio"])
        missed_ratio_matrix.append(ratios)
        missed_count_matrix.append(counts)
        gt_slices_matrix.append(gts)
        overlap_ratio_matrix.append(overlaps)

    matrix_summary = {
        "case_order": case_order,
        "organ_order": organ_order,
        "missed_ratio_matrix": missed_ratio_matrix,
        "missed_count_matrix": missed_count_matrix,
        "gt_slices_matrix": gt_slices_matrix,
        "overlap_ratio_matrix": overlap_ratio_matrix,
    }
    matrix_path = os.path.join(args.output_dir, "check_pred_matrix.json")
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix_summary, f, ensure_ascii=False, indent=2)

    dice_summary = {
        "case_order": case_order,
        "organ_order": organ_order,
        "overlap_ratio_matrix": overlap_ratio_matrix,
    }
    dice_path = os.path.join(args.output_dir, "check_pred_dice.json")
    with open(dice_path, "w", encoding="utf-8") as f:
        json.dump(dice_summary, f, ensure_ascii=False, indent=2)

    if args.write_heatmap:
        cell = 12
        height = max(1, len(case_order)) * cell
        width = max(1, len(organ_order)) * cell
        img = Image.new("RGB", (width, height), (255, 255, 255))
        pixels = img.load()
        for r, row in enumerate(missed_ratio_matrix):
            for c, ratio in enumerate(row):
                ratio = max(0.0, min(1.0, float(ratio)))
                red = 255
                green = int(round(255 * (1.0 - ratio)))
                blue = int(round(255 * (1.0 - ratio)))
                for y in range(r * cell, (r + 1) * cell):
                    for x in range(c * cell, (c + 1) * cell):
                        pixels[x, y] = (red, green, blue)
        heatmap_path = os.path.join(args.output_dir, "missed_ratio_heatmap.png")
        img.save(heatmap_path)


if __name__ == "__main__":
    main()
