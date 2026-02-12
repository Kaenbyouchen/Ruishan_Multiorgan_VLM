import argparse
import json
import os
from typing import Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_description(text: str) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""

    # Expected format:
    # "Morphology: irregular; Size: small; Position: left"
    segments = [seg.strip() for seg in raw.split(";") if seg.strip()]
    values = []
    for seg in segments:
        if ":" in seg:
            values.append(seg.split(":", 1)[1].strip())
        else:
            values.append(seg)

    if values:
        return "; ".join(values[:3])
    return raw


def convert_file(input_path: str, output_path: str) -> bool:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = data.get("descriptions")
    if not isinstance(descriptions, dict):
        return False

    new_descriptions: Dict[str, str] = {}
    for organ_name, text in descriptions.items():
        new_descriptions[organ_name] = normalize_description(text)
    data["descriptions"] = new_descriptions

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def resolve_each_class_dir(output_dir: str, each_class_dir: str) -> str:
    if each_class_dir:
        return each_class_dir
    if not output_dir:
        raise ValueError("必须提供 --output_dir 或 --each_class_dir")
    return os.path.join(output_dir, "each_class")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 each_class 下 descriptions 转为三段词组，并输出到 each_class_3words"
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="可选：output 目录（其下需有 each_class）",
    )
    parser.add_argument(
        "--each_class_dir",
        default="",
        help="可选：直接指定 each_class 目录",
    )
    args = parser.parse_args()

    src_dir = resolve_each_class_dir(args.output_dir, args.each_class_dir)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"each_class 目录不存在: {src_dir}")

    dst_dir = os.path.join(os.path.dirname(src_dir), "each_class_3words")
    ensure_dir(dst_dir)

    names = sorted([name for name in os.listdir(src_dir) if name.endswith(".json")])
    converted = 0
    skipped = 0
    for name in names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        ok = convert_file(src, dst)
        if ok:
            converted += 1
        else:
            skipped += 1

    print(f"[完成] 输入目录: {src_dir}")
    print(f"[完成] 输出目录: {dst_dir}")
    print(f"[完成] 转换文件数: {converted}, 跳过文件数: {skipped}")


if __name__ == "__main__":
    main()
