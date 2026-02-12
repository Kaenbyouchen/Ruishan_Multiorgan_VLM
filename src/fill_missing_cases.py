import argparse
import copy
import json
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Tuple

import nibabel as nib
import yaml


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def parse_ids(list_file: str, ids: List[str]) -> List[str]:
    if ids:
        return [str(v) for v in ids]
    data = read_yaml(list_file)
    return [str(v) for v in data]


def resolve_case_ids(cfg: dict) -> List[str]:
    dataset = cfg["dataset"]
    ids = dataset.get("ids", [])
    if ids:
        return [str(v) for v in ids]
    if not bool(dataset.get("use_all_cases", True)) and str(dataset.get("single_id", "")):
        return [str(dataset.get("single_id", ""))]
    return parse_ids(dataset["list_file"], [])


def slice_count_for_plane(shape: Tuple[int, int, int], plane: str) -> int:
    if plane == "axial":
        return int(shape[2])
    if plane == "coronal":
        return int(shape[1])
    if plane == "sagittal":
        return int(shape[0])
    raise ValueError(f"Unsupported plane: {plane}")


def parse_percent_range(range_str: str, total_slices: int) -> Tuple[int, int]:
    if not range_str or "%" not in range_str or "-" not in range_str:
        return 0, 0
    try:
        parts = range_str.replace("%", "").split("-")
        start_percent = float(parts[0].strip())
        end_percent = float(parts[1].strip())
    except (ValueError, IndexError):
        return 0, 0
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


def infer_final_output_dir(cfg: dict) -> str:
    output_dir = str(cfg["pipeline"]["output_dir"])
    keep_output_dir = bool(cfg["pipeline"].get("keep_output_dir", False))
    use_all_slices = bool(cfg["pipeline"].get("use_all_slices", False))
    use_all_cases = bool(cfg["dataset"].get("use_all_cases", True))
    per_class = bool(cfg["vlm"].get("per_class", False))

    if use_all_cases and per_class:
        output_dir = os.path.join(output_dir, "all_case")
    if (
        use_all_slices
        and not keep_output_dir
        and not os.path.basename(output_dir).startswith("full_sequence_")
    ):
        output_dir = os.path.join(
            os.path.dirname(output_dir),
            f"full_sequence_{os.path.basename(output_dir)}",
        )
    if (
        use_all_cases
        and not keep_output_dir
        and not os.path.basename(output_dir).startswith("all_cases_")
    ):
        output_dir = os.path.join(
            os.path.dirname(output_dir),
            f"all_cases_{os.path.basename(output_dir)}",
        )
    return output_dir


def load_bins(positions_json_path: str, key: str, total_slices: int) -> List[dict]:
    with open(positions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = data.get("positions", [])
    bins = []
    for item in positions:
        value = item.get(key)
        if value is None:
            continue
        number = int(round(float(value)))
        number = max(1, min(total_slices, number))
        bins.append({"range": str(item.get("range", "")), "slice_number": number})
    return bins


def expected_outputs_for_case(
    cfg: dict,
    case_id: str,
    image_path: str,
    positions_json_template: str,
    positions_json_key: str,
) -> Tuple[List[str], List[str]]:
    run_per_bin = bool(cfg["pipeline"].get("run_per_bin", False))
    positions_from_json = str(cfg["pipeline"].get("positions_from_json", ""))
    if not run_per_bin or not positions_from_json:
        return [f"{case_id}.json"], []

    json_path = positions_json_template.format(case_id=case_id)
    if not os.path.exists(json_path):
        return [], [f"positions_json_missing:{json_path}"]

    shape = nib.load(image_path).shape
    total_slices = slice_count_for_plane(shape, str(cfg["pipeline"]["plane"]))
    bins = load_bins(json_path, positions_json_key, total_slices)
    if not bins:
        return [], [f"positions_json_has_no_bins:{json_path}"]

    outputs = []
    for bin_entry in bins:
        slice_number = int(bin_entry["slice_number"])
        slice_start, slice_end = parse_percent_range(bin_entry.get("range", ""), total_slices)
        if slice_start == 0 or slice_end == 0:
            slice_start = slice_number
            slice_end = slice_number
        outputs.append(build_slice_range_name(case_id, slice_start, slice_end))
    return sorted(set(outputs)), []


def scan_missing_cases(cfg: dict) -> Dict[str, object]:
    case_ids = resolve_case_ids(cfg)
    final_output_dir = infer_final_output_dir(cfg)
    per_class = bool(cfg["vlm"].get("per_class", False))
    outputs_dir = os.path.join(final_output_dir, "each_class" if per_class else "vlm_outputs")

    image_dir = str(cfg["dataset"]["image_dir"])
    label_dir = str(cfg["dataset"]["label_dir"])
    positions_json_template = str(cfg["pipeline"].get("positions_from_json", ""))
    positions_json_key = str(cfg["pipeline"].get("positions_json_key", "average_slice"))

    missing_cases = []
    checked_cases = 0
    for case_id in case_ids:
        image_path = os.path.join(image_dir, f"{case_id}.nii.gz")
        label_path = os.path.join(label_dir, f"{case_id}.nii.gz")
        if not (os.path.exists(image_path) and os.path.exists(label_path)):
            missing_cases.append(
                {
                    "case_id": case_id,
                    "missing_files": [],
                    "errors": [f"missing_input:{image_path} or {label_path}"],
                }
            )
            continue

        checked_cases += 1
        expected_files, errors = expected_outputs_for_case(
            cfg, case_id, image_path, positions_json_template, positions_json_key
        )
        if errors:
            missing_cases.append(
                {"case_id": case_id, "missing_files": expected_files, "errors": errors}
            )
            continue

        missing_files = [
            name for name in expected_files if not os.path.exists(os.path.join(outputs_dir, name))
        ]
        if missing_files:
            missing_cases.append(
                {
                    "case_id": case_id,
                    "missing_files": missing_files,
                    "errors": [],
                }
            )

    return {
        "final_output_dir": final_output_dir,
        "outputs_dir": outputs_dir,
        "total_cases": len(case_ids),
        "checked_cases": checked_cases,
        "missing_case_count": len(missing_cases),
        "missing_cases": missing_cases,
    }


def run_single_case(config_path: str, base_cfg: dict, case_id: str) -> int:
    cfg_copy = copy.deepcopy(base_cfg)
    base_use_all_cases = bool(base_cfg["dataset"].get("use_all_cases", True))
    if base_use_all_cases:
        cfg_copy["dataset"]["ids"] = [str(case_id)]
        cfg_copy["dataset"]["use_all_cases"] = True
        cfg_copy["dataset"]["single_id"] = ""
    else:
        cfg_copy["dataset"]["ids"] = []
        cfg_copy["dataset"]["use_all_cases"] = False
        cfg_copy["dataset"]["single_id"] = str(case_id)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
    try:
        write_yaml(tmp_path, cfg_copy)
        cmd = ["python", "src/run_pipeline.py", "--config", tmp_path]
        print(f"[补齐] 运行 case {case_id}: {' '.join(cmd)}")
        env = os.environ.copy()
        env["VLM_SKIP_AUTO_FILL"] = "1"
        proc = subprocess.run(cmd, check=False, env=env)
        return int(proc.returncode)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描并一键补齐缺失 case 输出")
    parser.add_argument("--config", required=True, help="pipeline yaml 配置文件路径")
    parser.add_argument(
        "--run",
        action="store_true",
        help="执行补齐（不加该参数时仅扫描并打印缺失情况）",
    )
    parser.add_argument(
        "--report",
        default="",
        help="可选：将扫描结果写入 JSON 报告文件路径",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="可选：仅处理指定 case，逗号分隔，例如 580,581",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="补齐轮数上限（每轮会重新扫描后仅重跑仍缺失的 case）",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="每个 case 之间暂停秒数（用于降低限流风险）",
    )
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    target_case_ids = []
    if args.case_ids.strip():
        target_case_ids = [v.strip() for v in args.case_ids.split(",") if v.strip()]

    report = scan_missing_cases(cfg)
    if target_case_ids:
        report["missing_cases"] = [
            item for item in report["missing_cases"] if item["case_id"] in target_case_ids
        ]
        report["missing_case_count"] = len(report["missing_cases"])
    missing_case_ids = [item["case_id"] for item in report["missing_cases"]]

    print(
        f"[扫描] total={report['total_cases']} checked={report['checked_cases']} "
        f"missing_cases={report['missing_case_count']}"
    )
    if missing_case_ids:
        print("[扫描] 缺失 case:", ", ".join(missing_case_ids))
    else:
        print("[扫描] 没有缺失 case。")

    report_path = args.report
    if report_path:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[扫描] 报告已写入: {report_path}")

    if not args.run or not missing_case_ids:
        return

    max_rounds = max(1, int(args.max_rounds))
    sleep_seconds = max(0.0, float(args.sleep_seconds))

    all_failed_cases = set()
    remaining_ids = missing_case_ids
    for round_idx in range(1, max_rounds + 1):
        if not remaining_ids:
            break
        print(
            f"[补齐] round={round_idx}/{max_rounds}, "
            f"待补齐 case 数={len(remaining_ids)}"
        )
        round_failed = []
        for i, case_id in enumerate(remaining_ids, start=1):
            print(f"[补齐] ({i}/{len(remaining_ids)}) case={case_id}")
            code = run_single_case(args.config, cfg, case_id)
            if code != 0:
                round_failed.append(case_id)
                all_failed_cases.add(case_id)
            if sleep_seconds > 0 and i < len(remaining_ids):
                time.sleep(sleep_seconds)

        after = scan_missing_cases(cfg)
        if target_case_ids:
            after["missing_cases"] = [
                item for item in after["missing_cases"] if item["case_id"] in target_case_ids
            ]
            after["missing_case_count"] = len(after["missing_cases"])
        remaining_ids = [item["case_id"] for item in after["missing_cases"]]
        print(f"[补齐后] round={round_idx}, remaining_missing_cases={len(remaining_ids)}")
        if remaining_ids:
            print("[补齐后] 仍缺失:", ", ".join(remaining_ids))
        if round_failed:
            print("[补齐后] 本轮执行失败 case:", ", ".join(round_failed))

    if all_failed_cases:
        print("[补齐后] 累计执行失败 case:", ", ".join(sorted(all_failed_cases)))


if __name__ == "__main__":
    main()
