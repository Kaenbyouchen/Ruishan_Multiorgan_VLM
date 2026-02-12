import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


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


def list_case_ids(mask_dir: str) -> List[str]:
    ids = []
    for name in os.listdir(mask_dir):
        if name.endswith(".nii.gz"):
            ids.append(name.replace(".nii.gz", ""))
    return sorted(ids)


def load_mask(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata())


def slice_count_for_plane(volume: np.ndarray, plane: str) -> int:
    if plane == "axial":
        return volume.shape[2]
    if plane == "coronal":
        return volume.shape[1]
    if plane == "sagittal":
        return volume.shape[0]
    raise ValueError(f"Unsupported plane: {plane}")


def extract_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[:, :, index]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[index, :, :]
    raise ValueError(f"Unsupported plane: {plane}")


def positions_percent() -> List[int]:
    return list(range(0, 101))


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def collect_slice_counts(volume: np.ndarray, plane: str) -> np.ndarray:
    total = slice_count_for_plane(volume, plane)
    max_label = max(ORGAN_MAP.keys())
    counts = np.zeros((total, max_label + 1), dtype=np.int32)
    for index in range(total):
        slice_data = extract_slice(volume, plane, index)
        slice_labels = slice_data.astype(int)
        slice_labels[slice_labels > max_label] = 0
        bincount = np.bincount(slice_labels.ravel(), minlength=max_label + 1)
        counts[index, :] = bincount
    return counts


def per_position_area(
    counts_per_slice: np.ndarray,
    positions: List[int],
) -> Dict[int, np.ndarray]:
    total = counts_per_slice.shape[0]
    areas = {organ_id: np.zeros(len(positions), dtype=np.float32) for organ_id in ORGAN_MAP}
    for i, pos in enumerate(positions):
        frac = pos / 100.0
        number = int(round(frac * total))
        number = max(1, min(total, number))
        index = number - 1
        for organ_id in ORGAN_MAP:
            areas[organ_id][i] = float(counts_per_slice[index, organ_id])
    return areas


def normalize_by_max(areas: Dict[int, np.ndarray], counts_per_slice: np.ndarray) -> Dict[int, np.ndarray]:
    normalized = {}
    for organ_id, values in areas.items():
        max_area = float(np.max(counts_per_slice[:, organ_id]))
        if max_area <= 0:
            normalized[organ_id] = np.zeros_like(values, dtype=np.float32)
        else:
            normalized[organ_id] = values / max_area * 100.0
    return normalized


def plot_single_organ(
    positions: List[int],
    values: np.ndarray,
    organ_name: str,
    output_path: str,
    title_prefix: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(positions, values, linewidth=2)
    plt.ylim(0, 120)
    plt.xlabel("Case position (%)")
    plt.ylabel("Area / Max Area (%)")
    plt.title(f"{title_prefix} - {organ_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_all_organs(
    positions: List[int],
    values_by_organ: Dict[int, np.ndarray],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(10, 5))
    for organ_id, values in values_by_organ.items():
        plt.plot(positions, values, linewidth=1.8, label=ORGAN_MAP[organ_id])
    plt.ylim(0, 120)
    plt.xlabel("Case position (%)")
    plt.ylabel("Area / Max Area (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def median_position_in_bin(
    positions: List[int],
    values: np.ndarray,
    start: int,
    end: int,
) -> Tuple[Optional[int], float]:
    indices = [i for i, pos in enumerate(positions) if start <= pos <= end and values[i] > 0]
    if not indices:
        return None, 0.0
    bin_values = values[indices]
    median_value = float(np.median(bin_values))
    closest_idx = min(
        indices,
        key=lambda i: (abs(values[i] - median_value), -values[i], positions[i]),
    )
    return positions[closest_idx], median_value


def summarize_bins(
    positions: List[int],
    values: np.ndarray,
    window_size: int,
    total_slices: int,
) -> List[Dict[str, object]]:
    summaries = []
    for start in range(0, 100, window_size):
        if start + window_size >= 100:
            end = 100
        else:
            end = start + window_size - 1
        rep_pos, median_value = median_position_in_bin(positions, values, start, end)
        if rep_pos is None:
            rep_slice = None
        else:
            number = int(round((rep_pos / 100.0) * total_slices))
            rep_slice = max(1, min(total_slices, number))
        summaries.append(
            {
                "range": f"{start}%-{end}%",
                "median_percent": round(median_value, 2),
                "representative_position": rep_slice,
            }
        )
    return summaries


def process_case(
    case_id: str,
    mask_dir: str,
    plane: str,
    output_dir: str,
    window_size: int,
    tag: str,
) -> None:
    path = os.path.join(mask_dir, f"{case_id}.nii.gz")
    volume = load_mask(path)
    positions = positions_percent()

    counts_per_slice = collect_slice_counts(volume, plane)
    total_slices = counts_per_slice.shape[0]
    areas = per_position_area(counts_per_slice, positions)
    normalized = normalize_by_max(areas, counts_per_slice)

    for organ_id, values in normalized.items():
        organ_name = ORGAN_MAP[organ_id]
        filename = f"{tag}_{organ_id}_{sanitize_name(organ_name)}.png"
        plot_single_organ(
            positions,
            values,
            organ_name,
            os.path.join(output_dir, filename),
            tag.upper(),
        )

    plot_all_organs(
        positions,
        normalized,
        os.path.join(output_dir, f"{tag}_all_organs.png"),
        f"{tag.upper()} - All Organs",
    )

    summary = {}
    representative_by_organ = {}
    for organ_id, values in normalized.items():
        organ_name = ORGAN_MAP[organ_id]
        max_area = float(np.max(counts_per_slice[:, organ_id]))
        bins_summary = summarize_bins(positions, values, window_size, total_slices)
        summary[organ_name] = {
            "max_area": int(max_area),
            "bins": bins_summary,
        }
        representative_by_organ[organ_id] = bins_summary

    with open(os.path.join(output_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "plane": plane,
                "window_size_percent": window_size,
                "organs": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    representative_positions = []
    bin_count = len(next(iter(representative_by_organ.values())))
    for bin_index in range(bin_count):
        slice_values = []
        for organ_id in ORGAN_MAP:
            rep_slice = representative_by_organ[organ_id][bin_index]["representative_position"]
            if rep_slice is not None:
                slice_values.append(rep_slice)
        if slice_values:
            avg_slice = float(np.mean(slice_values))
            avg_percent = avg_slice / total_slices * 100.0
            representative_positions.append(
                {
                    "range": representative_by_organ[1][bin_index]["range"],
                    "average_percent": round(avg_percent, 2),
                    "average_slice": int(round(avg_slice)),
                    "used_organs": len(slice_values),
                }
            )
        else:
            representative_positions.append(
                {
                    "range": representative_by_organ[1][bin_index]["range"],
                    "average_percent": None,
                    "average_slice": None,
                    "used_organs": 0,
                }
            )

    with open(os.path.join(output_dir, f"{tag}_representative_positions.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "plane": plane,
                "window_size_percent": window_size,
                "total_slices": total_slices,
                "positions": representative_positions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organ area distribution per case (0-100%).")
    parser.add_argument("--gt_dir", default="", help="GT mask directory (NIfTI .nii.gz).")
    parser.add_argument("--pred_dir", default="", help="Pred mask directory (NIfTI .nii.gz).")
    parser.add_argument(
        "--case_id",
        default="",
        help="If provided, process only this case id.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/organ_square_distribution",
        help="Output directory for plots and JSON.",
    )
    parser.add_argument(
        "--plane",
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Slice plane for position statistics.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Bin size in percent (e.g. 10 => 0%-9%, 10%-19%).",
    )
    return parser.parse_args()


def resolve_sources(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if not args.gt_dir or not args.pred_dir:
        raise ValueError("Both gt_dir and pred_dir are required.")
    return [("gt", args.gt_dir), ("pred", args.pred_dir)]


def main() -> None:
    args = parse_args()
    sources = resolve_sources(args)
    ensure_dir(args.output_dir)

    for tag, mask_dir in sources:
        case_ids = [args.case_id] if args.case_id else list_case_ids(mask_dir)
        for case_id in case_ids:
            case_output = os.path.join(args.output_dir, f"case_{case_id}")
            ensure_dir(case_output)
            process_case(case_id, mask_dir, args.plane, case_output, args.window_size, tag)

    print(f"Saved plots and JSON to: {args.output_dir}")


if __name__ == "__main__":
    main()
