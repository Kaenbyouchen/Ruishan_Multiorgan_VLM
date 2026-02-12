import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_percent_range(range_str: str, total_slices: int) -> Tuple[int, int]:
    if not range_str or "%" not in range_str or "-" not in range_str:
        return 0, 0
    parts = range_str.replace("%", "").split("-")
    start_percent = float(parts[0].strip())
    end_percent = float(parts[1].strip())
    start_frac = min(max(start_percent / 100.0, 0.0), 1.0)
    end_frac = min(max(end_percent / 100.0, 0.0), 1.0)
    start_slice = int(round(start_frac * total_slices))
    end_slice = int(round(end_frac * total_slices))
    start_slice = max(1, min(total_slices, start_slice))
    end_slice = max(1, min(total_slices, end_slice))
    if end_slice < start_slice:
        end_slice = start_slice
    return start_slice, end_slice


def build_slice_range_name(case_id: str, slice_start: int, slice_end: int) -> str:
    return f"{case_id}_slice{slice_start}_to_{slice_end}.json"


def compute_square_hard_classes_from_pred_summary(
    pred_summary: Dict[str, object],
    range_str: str,
    topk: int = 5,
    include_absent_to_pad: bool = True,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, Optional[float]]]]:
    """
    Use outputs/organ_square_distribution/.../pred_summary.json.
    Rank organs by bin-level median_percent (area/max_area %) ascending.
    """
    organs = pred_summary.get("organs", {})
    if not isinstance(organs, dict):
        return [], [], {}

    present_scores: List[Tuple[float, str]] = []
    absent_names: List[str] = []
    stats: Dict[str, Dict[str, Optional[float]]] = {}

    for _, organ_name in ORGAN_MAP.items():
        entry = organs.get(organ_name, None)
        median_percent_val: float = 0.0
        representative_position: Optional[int] = None

        if isinstance(entry, dict):
            bins = entry.get("bins", [])
            if isinstance(bins, list):
                for b in bins:
                    if not isinstance(b, dict):
                        continue
                    if str(b.get("range", "")) == range_str:
                        median_percent_val = float(b.get("median_percent", 0.0) or 0.0)
                        representative_position = b.get("representative_position", None)
                        if representative_position is not None:
                            representative_position = int(representative_position)
                        break

        stats[organ_name] = {
            "median_percent": float(median_percent_val),
            "representative_position": representative_position,
        }

        if median_percent_val > 0:
            present_scores.append((median_percent_val, organ_name))
        else:
            absent_names.append(organ_name)

    present_scores.sort(key=lambda x: x[0])  # smaller => harder
    present_only = [n for _, n in present_scores[:topk]]

    padded = list(present_only)
    if include_absent_to_pad and len(padded) < topk:
        for n in absent_names:
            if len(padded) >= topk:
                break
            if n not in padded:
                padded.append(n)

    return present_only, padded, stats


## NOTE: 旧的“从 NIfTI 逐 slice 统计面积”的实现曾在沙箱环境触发 exit 139；
## 这里改为直接复用 square distribution 已输出的 pred_summary.json（按 median_percent 排序）。


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VLM hard_classes vs square(area) hard_classes per 10% bin."
    )
    parser.add_argument(
        "--case_id",
        default="",
        help="If set, only evaluate this case id; otherwise evaluate all cases in positions dir.",
    )
    parser.add_argument(
        "--vlm_output_dir",
        default="outputs/ten_percent_eachclass_prior_case/each_class",
        help="Directory containing VLM per-bin JSON outputs.",
    )
    parser.add_argument(
        "--pred_dir",
        default="data/pred_amos_mr",
        help="(Unused) kept for backward compatibility.",
    )
    parser.add_argument(
        "--positions_dir",
        default="outputs/organ_square_distribution",
        help="Directory containing pred_representative_positions.json per case.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/hardclass_compare",
        help="Directory to write per-case comparison JSON.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # Determine cases
    case_ids: List[str] = []
    if args.case_id:
        case_ids = [str(args.case_id)]
    else:
        for name in os.listdir(args.positions_dir):
            if name.startswith("case_") and os.path.isdir(os.path.join(args.positions_dir, name)):
                case_ids.append(name.replace("case_", ""))
        case_ids = sorted(case_ids)

    for case_id in case_ids:
        positions_path = os.path.join(
            args.positions_dir, f"case_{case_id}", "pred_representative_positions.json"
        )
        pred_summary_path = os.path.join(args.positions_dir, f"case_{case_id}", "pred_summary.json")
        if not (os.path.exists(positions_path) and os.path.exists(pred_summary_path)):
            continue

        positions = load_json(positions_path)
        pred_summary = load_json(pred_summary_path)
        total_slices = int(positions.get("total_slices") or 0)
        pos_items = positions.get("positions", [])
        if not total_slices or not isinstance(pos_items, list):
            continue

        bins_report = []
        ordered_match = 0
        set_match = 0
        overlap_sum = 0.0
        valid_bins = 0
        comparable_bins = 0

        for item in pos_items:
            range_str = str(item.get("range", ""))
            avg_slice = item.get("average_slice", None)
            if avg_slice is None:
                continue
            slice_number = int(round(float(avg_slice)))
            slice_start, slice_end = parse_percent_range(range_str, total_slices)
            if slice_start == 0 or slice_end == 0:
                slice_start = slice_number
                slice_end = slice_number

            fname = build_slice_range_name(case_id, slice_start, slice_end)
            vlm_path = os.path.join(args.vlm_output_dir, fname)
            if not os.path.exists(vlm_path):
                continue
            vlm = load_json(vlm_path)
            vlm_hard = vlm.get("hard_classes", [])
            if not isinstance(vlm_hard, list):
                vlm_hard = []
            vlm_hard = [str(x) for x in vlm_hard][:5]

            square_hard_present, square_hard_padded, square_stats = compute_square_hard_classes_from_pred_summary(
                pred_summary, range_str, topk=5, include_absent_to_pad=True
            )

            # Metrics
            valid_bins += 1
            inter = len(set(vlm_hard) & set(square_hard_padded))
            overlap = inter / 5.0 if 5 else 0.0
            overlap_sum += overlap
            if len(vlm_hard) == 5:
                comparable_bins += 1
                if vlm_hard == square_hard_padded:
                    ordered_match += 1
                if set(vlm_hard) == set(square_hard_padded):
                    set_match += 1

            bins_report.append(
                {
                    "range": range_str,
                    "average_slice": slice_number,
                    "file": fname,
                    "vlm_hard_classes": vlm_hard,
                    "square_hard_classes_present_only": square_hard_present,
                    "square_hard_classes_padded": square_hard_padded,
                    "square_stats": square_stats,
                    "overlap_at_5": overlap,
                    "ordered_match": len(vlm_hard) == 5 and vlm_hard == square_hard_padded,
                    "set_match": len(vlm_hard) == 5 and set(vlm_hard) == set(square_hard_padded),
                }
            )

        if valid_bins == 0:
            continue

        report = {
            "case_id": case_id,
            "vlm_output_dir": args.vlm_output_dir,
            "pred_dir": args.pred_dir,
            "positions_path": positions_path,
            "pred_summary_path": pred_summary_path,
            "square_basis": "pred_summary.json bins[].median_percent (area/max_area %) ascending",
            "valid_bins": valid_bins,
            "comparable_bins": comparable_bins,
            "ordered_match_rate": (ordered_match / comparable_bins) if comparable_bins else 0.0,
            "set_match_rate": (set_match / comparable_bins) if comparable_bins else 0.0,
            "mean_overlap_at_5": overlap_sum / valid_bins,
            "ordered_match_rate_percent": (
                (ordered_match / comparable_bins) * 100.0 if comparable_bins else 0.0
            ),
            "set_match_rate_percent": ((set_match / comparable_bins) * 100.0 if comparable_bins else 0.0),
            "mean_overlap_at_5_percent": (overlap_sum / valid_bins) * 100.0,
            "bins": bins_report,
        }

        out_path = os.path.join(args.output_dir, f"{case_id}_hardclass_compare.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

