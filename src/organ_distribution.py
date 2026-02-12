import argparse
import json
import os
from typing import Dict, List, Tuple

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


def count_presence_by_position(mask_dir: str, plane: str) -> Tuple[Dict[int, np.ndarray], int, List[int]]:
    case_ids = list_case_ids(mask_dir)
    positions = positions_percent()
    counts = {organ_id: np.zeros(len(positions), dtype=np.int32) for organ_id in ORGAN_MAP}

    for case_id in case_ids:
        path = os.path.join(mask_dir, f"{case_id}.nii.gz")
        volume = load_mask(path)
        total = slice_count_for_plane(volume, plane)
        for i, pos in enumerate(positions):
            frac = pos / 100.0
            number = int(round(frac * total))
            number = max(1, min(total, number))
            index = number - 1
            slice_data = extract_slice(volume, plane, index)
            labels = set(np.unique(slice_data).astype(int).tolist())
            for organ_id in ORGAN_MAP:
                if organ_id in labels:
                    counts[organ_id][i] += 1

    return counts, len(case_ids), positions


def to_percentages(counts: Dict[int, np.ndarray], total_cases: int) -> Dict[int, np.ndarray]:
    if total_cases == 0:
        return {organ_id: np.zeros_like(values, dtype=float) for organ_id, values in counts.items()}
    return {organ_id: values / total_cases * 100.0 for organ_id, values in counts.items()}


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_")


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
    plt.ylabel("Cases with organ (%)")
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
    plt.ylabel("Cases with organ (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def best_range(values: np.ndarray, window_size: int) -> Tuple[str, float]:
    window_len = window_size + 1
    best_start = 0
    best_mean = -1.0
    for start in range(0, len(values) - window_len + 1):
        mean_value = float(np.mean(values[start : start + window_len]))
        if mean_value > best_mean:
            best_mean = mean_value
            best_start = start
    return f"{best_start}%-{best_start + window_size}%", best_mean


def process_dataset(
    tag: str,
    mask_dir: str,
    plane: str,
    output_dir: str,
    window_size: int,
) -> Dict[str, Dict[str, object]]:
    counts, case_count, positions = count_presence_by_position(mask_dir, plane)
    percentages = to_percentages(counts, case_count)

    for organ_id, values in percentages.items():
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
        percentages,
        os.path.join(output_dir, f"{tag}_all_organs.png"),
        f"{tag.upper()} - All Organs",
    )

    ranges = {}
    for organ_id, values in percentages.items():
        organ_name = ORGAN_MAP[organ_id]
        range_text, mean_value = best_range(values, window_size)
        ranges[organ_name] = {
            "range": range_text,
            "mean_percent": round(mean_value, 2),
        }

    return {
        "case_count": case_count,
        "ranges": ranges,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organ distribution over case positions (0-100%).")
    parser.add_argument("--gt_dir", required=True, help="GT mask directory (NIfTI .nii.gz).")
    parser.add_argument("--pred_dir", required=True, help="Predicted mask directory (NIfTI .nii.gz).")
    parser.add_argument(
        "--output_dir",
        default="outputs/organ_distribution",
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
        help="Window size in percent for peak range (e.g. 10 => 20%-30%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    gt_result = process_dataset("gt", args.gt_dir, args.plane, args.output_dir, args.window_size)
    pred_result = process_dataset("pred", args.pred_dir, args.plane, args.output_dir, args.window_size)

    summary = {
        "window_size_percent": args.window_size,
        "gt": gt_result,
        "pred": pred_result,
    }
    with open(os.path.join(args.output_dir, "organ_peak_ranges.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved plots and JSON to: {args.output_dir}")


if __name__ == "__main__":
    main()
