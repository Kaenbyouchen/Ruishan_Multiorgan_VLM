import argparse
import json
import os
from typing import Dict, Tuple


def parse_description(desc: str) -> Dict[str, str]:
    if not desc:
        return {"morphology": "", "size": "", "position": ""}
    parts = [p.strip() for p in desc.split(";")]
    result = {"morphology": "", "size": "", "position": ""}
    for part in parts:
        if part.lower().startswith("morphology:"):
            result["morphology"] = part.split(":", 1)[1].strip()
        elif part.lower().startswith("size:"):
            result["size"] = part.split(":", 1)[1].strip()
        elif part.lower().startswith("position:"):
            result["position"] = part.split(":", 1)[1].strip()
    return result


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_outputs(heuristic: Dict[str, object], vlm: Dict[str, object]) -> Dict[str, object]:
    h_hard = heuristic.get("hard_classes", [])
    v_hard = vlm.get("hard_classes", [])
    h_desc = heuristic.get("descriptions", {})
    v_desc = vlm.get("descriptions", {})

    hard_overlap = len(set(h_hard) & set(v_hard))
    hard_precision = hard_overlap / 5.0 if v_hard else 0.0
    hard_recall = hard_overlap / 5.0 if h_hard else 0.0

    per_organ = {}
    for organ_name, h_text in h_desc.items():
        v_text = v_desc.get(organ_name, "")
        h_parts = parse_description(h_text)
        v_parts = parse_description(v_text)
        per_organ[organ_name] = {
            "morphology_match": h_parts["morphology"] == v_parts["morphology"],
            "size_match": h_parts["size"] == v_parts["size"],
            "position_match": h_parts["position"] == v_parts["position"],
        }
    return {
        "hard_precision": hard_precision,
        "hard_recall": hard_recall,
        "hard_overlap": hard_overlap,
        "per_organ": per_organ,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VLM outputs to heuristic outputs")
    parser.add_argument("--heuristic_dir", required=True, help="Heuristic output folder")
    parser.add_argument("--vlm_dir", required=True, help="VLM output folder")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    heuristic_files = [f for f in os.listdir(args.heuristic_dir) if f.endswith(".json")]
    results = []
    for fname in heuristic_files:
        h_path = os.path.join(args.heuristic_dir, fname)
        v_path = os.path.join(args.vlm_dir, fname)
        if not os.path.exists(v_path):
            continue
        h_data = load_json(h_path)
        v_data = load_json(v_path)
        res = compare_outputs(h_data, v_data)
        res["file"] = fname
        results.append(res)

    # aggregate
    hard_p = [r["hard_precision"] for r in results]
    hard_r = [r["hard_recall"] for r in results]
    avg_hard_p = sum(hard_p) / len(hard_p) if hard_p else 0.0
    avg_hard_r = sum(hard_r) / len(hard_r) if hard_r else 0.0

    organ_stats = {}
    for r in results:
        for organ, stat in r["per_organ"].items():
            organ_stats.setdefault(organ, {"morph": 0, "size": 0, "pos": 0, "count": 0})
            organ_stats[organ]["morph"] += 1 if stat["morphology_match"] else 0
            organ_stats[organ]["size"] += 1 if stat["size_match"] else 0
            organ_stats[organ]["pos"] += 1 if stat["position_match"] else 0
            organ_stats[organ]["count"] += 1

    for organ, stat in organ_stats.items():
        count = stat["count"] or 1
        stat["morphology_match_rate"] = stat["morph"] / count
        stat["size_match_rate"] = stat["size"] / count
        stat["position_match_rate"] = stat["pos"] / count

    output = {
        "files_compared": len(results),
        "avg_hard_precision": avg_hard_p,
        "avg_hard_recall": avg_hard_r,
        "organ_stats": organ_stats,
        "details": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
