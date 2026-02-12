import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import nibabel as nib
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

ORGAN_NAME_TO_ID = {v: k for k, v in ORGAN_MAP.items()}

RANGE_FILE_RE = re.compile(r"^(?P<case_id>\d+)_slice(?P<start>\d+)_to_(?P<end>\d+)\.json$")
CASE_FILE_RE = re.compile(r"^(?P<case_id>\d+)\.json$")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_nifti(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata()).astype(np.int32)


def slice_count_for_plane(shape: Tuple[int, int, int], plane: str) -> int:
    if plane == "axial":
        return int(shape[2])
    if plane == "coronal":
        return int(shape[1])
    if plane == "sagittal":
        return int(shape[0])
    raise ValueError(f"Unsupported plane: {plane}")


def parse_case_and_range(filename: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    m = RANGE_FILE_RE.match(filename)
    if m:
        return m.group("case_id"), int(m.group("start")), int(m.group("end"))
    m = CASE_FILE_RE.match(filename)
    if m:
        return m.group("case_id"), None, None
    return None, None, None


def count_organ_voxels_in_range(
    volume: np.ndarray,
    organ_id: int,
    plane: str,
    slice_start: int,
    slice_end: int,
) -> int:
    s = max(1, int(slice_start))
    e = max(1, int(slice_end))
    total_slices = slice_count_for_plane(volume.shape, plane)
    s = min(total_slices, s)
    e = min(total_slices, e)
    if e < s:
        e = s
    s0 = s - 1
    e0 = e - 1

    if plane == "axial":
        block = volume[:, :, s0 : e0 + 1]
    elif plane == "coronal":
        block = volume[:, s0 : e0 + 1, :]
    elif plane == "sagittal":
        block = volume[s0 : e0 + 1, :, :]
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    return int(np.sum(block == organ_id))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查每个 VLM JSON 的 hard_classes 是否在 GT 中可找到"
    )
    parser.add_argument(
        "--vlm_dir",
        default="outputs/ten_percent_eachclass_prior_case/all_case/each_class",
        help="VLM 输出 JSON 目录（如 each_class 或 each_class_3words）",
    )
    parser.add_argument(
        "--gt_dir",
        default="data/gt_amos_mr",
        help="GT NIfTI 目录",
    )
    parser.add_argument(
        "--plane",
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="切片方向",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/check_hardclass",
        help="输出目录",
    )
    parser.add_argument(
        "--output_name",
        default="hardclass_in_gt_check.json",
        help="输出 JSON 文件名",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.vlm_dir):
        raise FileNotFoundError(f"vlm_dir 不存在: {args.vlm_dir}")

    ensure_dir(args.output_dir)

    names = sorted([n for n in os.listdir(args.vlm_dir) if n.endswith(".json")])
    gt_cache: Dict[str, np.ndarray] = {}

    files_result: List[dict] = []
    organ_stats = {
        name: {"occurrences": 0, "found": 0, "missing": 0}
        for name in ORGAN_NAME_TO_ID
    }
    unknown_organ_stats: Dict[str, int] = {}

    for name in names:
        in_path = os.path.join(args.vlm_dir, name)
        case_id, s, e = parse_case_and_range(name)
        entry = {
            "file": name,
            "case_id": case_id,
            "slice_start": s,
            "slice_end": e,
            "hard_classes": [],
            "checks": [],
            "all_hard_classes_found_in_gt": False,
            "error": "",
        }
        files_result.append(entry)

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            entry["error"] = f"json_load_error:{ex}"
            continue

        hard_classes = data.get("hard_classes", [])
        if not isinstance(hard_classes, list):
            entry["error"] = "hard_classes_not_list"
            continue
        entry["hard_classes"] = [str(x) for x in hard_classes]

        if case_id is None:
            entry["error"] = "filename_not_supported"
            continue

        gt_path = os.path.join(args.gt_dir, f"{case_id}.nii.gz")
        if not os.path.exists(gt_path):
            entry["error"] = f"gt_missing:{gt_path}"
            continue

        if case_id not in gt_cache:
            gt_cache[case_id] = load_nifti(gt_path)
        volume = gt_cache[case_id]
        total_slices = slice_count_for_plane(volume.shape, args.plane)

        if s is None or e is None:
            s = 1
            e = total_slices
            entry["slice_start"] = s
            entry["slice_end"] = e

        all_found = True
        for organ_name in entry["hard_classes"]:
            item = {
                "organ_name": organ_name,
                "organ_id": None,
                "found_in_gt": False,
                "gt_voxel_count_in_range": 0,
            }
            organ_id = ORGAN_NAME_TO_ID.get(organ_name)
            if organ_id is None:
                unknown_organ_stats[organ_name] = unknown_organ_stats.get(organ_name, 0) + 1
                all_found = False
                entry["checks"].append(item)
                continue

            item["organ_id"] = organ_id
            voxels = count_organ_voxels_in_range(volume, organ_id, args.plane, s, e)
            item["gt_voxel_count_in_range"] = voxels
            item["found_in_gt"] = voxels > 0
            entry["checks"].append(item)

            organ_stats[organ_name]["occurrences"] += 1
            if item["found_in_gt"]:
                organ_stats[organ_name]["found"] += 1
            else:
                organ_stats[organ_name]["missing"] += 1
                all_found = False

        entry["all_hard_classes_found_in_gt"] = all_found

    total_files = len(files_result)
    parsed_files = len([x for x in files_result if not x["error"]])
    ok_files = len([x for x in files_result if x["all_hard_classes_found_in_gt"] and not x["error"]])
    failed_files = parsed_files - ok_files
    total_hard_items = sum(len(x["hard_classes"]) for x in files_result if not x["error"])
    total_found_items = sum(
        sum(1 for c in x["checks"] if c["found_in_gt"])
        for x in files_result
        if not x["error"]
    )
    total_missing_items = total_hard_items - total_found_items

    result = {
        "vlm_dir": args.vlm_dir,
        "gt_dir": args.gt_dir,
        "plane": args.plane,
        "summary": {
            "total_files": total_files,
            "parsed_files": parsed_files,
            "parse_error_files": total_files - parsed_files,
            "files_all_hardclass_found": ok_files,
            "files_with_missing_hardclass": failed_files,
            "total_hardclass_items": total_hard_items,
            "found_items": total_found_items,
            "missing_items": total_missing_items,
            "unknown_organ_items": int(sum(unknown_organ_stats.values())),
        },
        "organ_stats": organ_stats,
        "unknown_organ_stats": unknown_organ_stats,
        "files": files_result,
    }

    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写入: {out_path}")
    print(
        "[统计] total_files={total_files}, parsed_files={parsed_files}, "
        "files_with_missing_hardclass={failed_files}, missing_items={total_missing_items}".format(
            total_files=total_files,
            parsed_files=parsed_files,
            failed_files=failed_files,
            total_missing_items=total_missing_items,
        )
    )


if __name__ == "__main__":
    main()
