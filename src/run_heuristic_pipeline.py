import argparse
import json
import os
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import yaml

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


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_nifti(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata()).astype(np.int32)


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


def read_case_ids(list_file: str, ids: List[str], use_all_cases: bool, single_id: str) -> List[str]:
    if ids:
        return [str(v) for v in ids]
    if not use_all_cases and single_id:
        return [str(single_id)]
    data = read_yaml(list_file)
    return [str(v) for v in data]


def connected_components_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def classify_position(center: Tuple[float, float], shape: Tuple[int, int]) -> str:
    row, col = center
    h, w = shape
    vertical = "up"
    if row > h * 0.66:
        vertical = "down"
    elif row > h * 0.33:
        vertical = "center"
    horizontal = "left"
    if col > w * 0.66:
        horizontal = "right"
    elif col > w * 0.33:
        horizontal = "center"
    if vertical == "center" and horizontal == "center":
        return "center"
    if vertical == "center":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{horizontal}-{vertical}"


def classify_morphology(mask: np.ndarray) -> Tuple[str, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return "", 0.0
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    ratio = max(width, height) / max(1, min(width, height))
    components = connected_components_count(mask)
    if components >= 2:
        return "irregular", ratio
    if ratio >= 2.0:
        return "elongated", ratio
    if ratio >= 1.2:
        return "oval", ratio
    return "round", ratio


def size_bucket(area: int, max_area: int) -> str:
    if max_area <= 0:
        return ""
    frac = area / max_area
    if frac <= 0.2:
        return "very small"
    if frac <= 0.4:
        return "small"
    if frac <= 0.6:
        return "medium"
    if frac <= 0.8:
        return "large"
    return "very large"


def compute_hard_score(area: int, max_area: int, components: int, elong_ratio: float) -> float:
    area_norm = area / max_area if max_area else 0.0
    comp_norm = min(1.0, (components - 1) / 4.0)
    elong_norm = min(1.0, (elong_ratio - 1.0) / 2.0)
    return (1.0 - area_norm) * 0.5 + comp_norm * 0.3 + elong_norm * 0.2


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic hard class + description pipeline")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    ids = read_case_ids(
        cfg["dataset"]["list_file"],
        cfg["dataset"].get("ids", []),
        bool(cfg["dataset"].get("use_all_cases", True)),
        str(cfg["dataset"].get("single_id", "")),
    )
    image_dir = cfg["dataset"]["image_dir"]
    label_dir = cfg["dataset"]["label_dir"]
    output_dir = cfg["pipeline"]["output_dir"]
    plane = cfg["pipeline"]["plane"]
    positions_from_json = cfg["pipeline"]["positions_from_json"]
    positions_json_key = cfg["pipeline"].get("positions_json_key", "average_slice")
    run_per_bin = bool(cfg["pipeline"].get("run_per_bin", True))

    ensure_dir(output_dir)
    outputs_dir = os.path.join(output_dir, "heuristic_outputs")
    ensure_dir(outputs_dir)

    for case_id in ids:
        image_path = os.path.join(image_dir, f"{case_id}.nii.gz")
        label_path = os.path.join(label_dir, f"{case_id}.nii.gz")
        if not (os.path.exists(image_path) and os.path.exists(label_path)):
            continue
        label = load_nifti(label_path)
        total_slices = slice_count_for_plane(label, plane)

        json_path = positions_from_json.format(case_id=case_id)
        with open(json_path, "r", encoding="utf-8") as f:
            pos_data = json.load(f)
        bins = pos_data.get("positions", [])

        for bin_entry in bins:
            slice_number = int(round(float(bin_entry.get(positions_json_key))))
            slice_number = max(1, min(total_slices, slice_number))
            slice_index = slice_number - 1
            slice_start, slice_end = parse_percent_range(bin_entry.get("range", ""), total_slices)
            if slice_start == 0 or slice_end == 0:
                slice_start = slice_number
                slice_end = slice_number

            descriptions = {}
            hard_scores = {}
            for organ_id, organ_name in ORGAN_MAP.items():
                mask_slice = extract_slice(label == organ_id, plane, slice_index)
                area = int(mask_slice.sum())
                if area == 0:
                    descriptions[organ_name] = ""
                    hard_scores[organ_name] = 1.0
                    continue
                morph, elong_ratio = classify_morphology(mask_slice)
                coords = np.argwhere(mask_slice)
                center = coords.mean(axis=0)
                position = classify_position((center[0], center[1]), mask_slice.shape)

                # max area over all slices for this organ in this case
                organ_mask = label == organ_id
                max_area = 0
                for idx in range(total_slices):
                    s = extract_slice(organ_mask, plane, idx)
                    max_area = max(max_area, int(s.sum()))

                size = size_bucket(area, max_area)
                desc = f" Morphology: {morph}; Size: {size}; Position: {position}"
                descriptions[organ_name] = desc.strip()

                components = connected_components_count(mask_slice)
                hard_scores[organ_name] = compute_hard_score(area, max_area, components, elong_ratio)

            hard_classes = [k for k, _ in sorted(hard_scores.items(), key=lambda x: x[1], reverse=True)][:5]

            output = {"hard_classes": hard_classes, "descriptions": descriptions}
            output_name = f"{case_id}_slice{slice_start}_to_{slice_end}.json"
            output_path = os.path.join(outputs_dir, output_name)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
