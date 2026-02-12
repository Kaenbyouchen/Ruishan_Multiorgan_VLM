import json
import base64
import os
import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import requests
import yaml
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


@dataclass
class PipelineConfig:
    image_dir: str
    label_dir: str
    list_file: str
    ids: List[str]
    use_all_cases: bool
    single_id: str
    output_dir: str
    keep_output_dir: bool
    plane: str
    positions_percent: List[float]
    positions_from_json: str
    positions_json_key: str
    auto_compute_positions: bool
    positions_compute_gt_dir: str
    positions_compute_output_dir: str
    positions_compute_window_size: int
    run_per_bin: bool
    use_all_slices: bool
    save_vlm_inputs_all: bool
    auto_generate_each_class_3words: bool
    auto_fill_missing_cases_after_run: bool
    auto_fill_max_rounds: int
    auto_fill_sleep_seconds: float
    overlay_alpha: float
    vlm_name: str
    vlm_mode: str
    prompt_template: str
    api_key_env: str
    api_key: str
    api_key_env_map: Dict[str, str]
    provider: str
    model: str
    api_base: str
    max_images: int
    request_retries: int
    per_class: bool
    prompt_template_each_class: str
    prompt_template_hard_class: str
    require_nonempty_hard_classes: bool
    save_each_class_prompt_preview_first_bin: bool
    force_nonempty_descriptions_for_hard_classes: bool
    descriptions_only_for_hard_classes: bool


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_nifti(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata())


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    low, high = np.percentile(volume, [1, 99])
    if high <= low:
        high = low + 1.0
    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low)
    return (volume * 255).astype(np.uint8)


def parse_ids(list_file: str, ids: List[str]) -> List[str]:
    if ids:
        return [str(v) for v in ids]
    data = read_yaml(list_file)
    return [str(v) for v in data]


def resolve_case_ids(cfg: PipelineConfig) -> List[str]:
    if cfg.ids:
        return [str(v) for v in cfg.ids]
    if not cfg.use_all_cases and cfg.single_id:
        return [str(cfg.single_id)]
    return parse_ids(cfg.list_file, [])


def to_fraction(value: float) -> float:
    if value > 1.0:
        return value / 100.0
    return value


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


def pick_slice_numbers(total: int, positions_percent: List[float]) -> List[int]:
    numbers = []
    for value in positions_percent:
        frac = min(max(to_fraction(value), 0.0), 1.0)
        number = int(round(frac * total))
        number = max(1, min(total, number))
        numbers.append(number)
    return sorted(set(numbers))


def load_slice_numbers_from_json(path: str, key: str, total_slices: int) -> List[int]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = data.get("positions", [])
    numbers = []
    for item in positions:
        value = item.get(key)
        if value is None:
            continue
        number = int(round(float(value)))
        number = max(1, min(total_slices, number))
        numbers.append(number)
    return sorted(set(numbers))


def load_bins_from_json(path: str, key: str, total_slices: int) -> List[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
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


def transform_output(img: Image.Image) -> Image.Image:
    rotated = img.rotate(-90, expand=True)
    return rotated.transpose(Image.FLIP_LEFT_RIGHT)


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


def is_first_decile(range_str: str) -> bool:
    if not range_str or "-" not in range_str or "%" not in range_str:
        return False
    try:
        start = float(range_str.split("-", 1)[0].replace("%", "").strip())
    except ValueError:
        return False
    return abs(start - 0.0) < 1e-6


def ensure_positions_json(cfg: PipelineConfig, case_id: str) -> None:
    if not cfg.auto_compute_positions or not cfg.positions_from_json:
        return
    json_path = cfg.positions_from_json.format(case_id=case_id)
    if os.path.exists(json_path):
        return
    # Auto-compute representative positions via organ_square_distribution.py
    cmd = [
        sys.executable,
        os.path.join("src", "organ_square_distribution.py"),
        "--gt_dir",
        cfg.positions_compute_gt_dir,
        "--pred_dir",
        cfg.label_dir,
        "--case_id",
        str(case_id),
        "--plane",
        cfg.plane,
        "--output_dir",
        cfg.positions_compute_output_dir,
        "--window_size",
        str(cfg.positions_compute_window_size),
    ]
    subprocess.run(cmd, check=True)


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    for _ in range(max(1, iterations)):
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        shifts = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        mask = np.logical_or.reduce(shifts)
    return mask


def mask_overlay(image_slice: np.ndarray, mask_slice: np.ndarray, alpha: float) -> Image.Image:
    image_rgb = np.stack([image_slice] * 3, axis=-1)
    overlay = image_rgb.copy()
    mask = mask_slice > 0
    if mask.any():
        dilated = dilate_mask(mask, iterations=1)
        outline = np.logical_and(dilated, ~mask)
        overlay[outline] = np.array([255, 255, 0], dtype=np.uint8)
        overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
    blended = (image_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return transform_output(Image.fromarray(blended))


def describe_position(center: Tuple[float, float], shape: Tuple[int, int]) -> str:
    row, col = center
    h, w = shape
    vertical = "上"
    if row > h * 0.66:
        vertical = "下"
    elif row > h * 0.33:
        vertical = "中"
    horizontal = "左"
    if col > w * 0.66:
        horizontal = "右"
    elif col > w * 0.33:
        horizontal = "中"
    if vertical == "中" and horizontal == "中":
        return "图像中央"
    return f"图像{vertical}{horizontal}"


def compute_mask_stats(mask: np.ndarray) -> Dict[str, float]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return {"volume": 0.0, "center_z": 0.0, "center_y": 0.0, "center_x": 0.0}
    center = coords.mean(axis=0)
    return {
        "volume": float(coords.shape[0]),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
    }


def build_prompt(
    template: str,
    plane: str,
    organ_map: Dict[int, str],
    image_labels: str = "",
    target_organ: str = "",
    slice_number: str = "",
) -> str:
    organ_list = ", ".join([f"{k}:{v}" for k, v in organ_map.items()])
    prompt = template
    prompt = prompt.replace("{plane}", plane)
    prompt = prompt.replace("{organ_list}", organ_list)
    prompt = prompt.replace("{image_labels}", image_labels)
    if "{target_organ}" in prompt:
        prompt = prompt.replace("{target_organ}", target_organ)
    if "{slice_number}" in prompt:
        prompt = prompt.replace("{slice_number}", slice_number)
    return prompt


def build_image_labels(images: List[dict]) -> str:
    if not images:
        return ""
    lines = []
    for item in images:
        image_type = item.get("image_type", "mask_overlay")
        lines.append(
            f"- Organ: {item['organ_name']}; Slice: {item['slice_number']}; Type: {image_type}"
        )
    return "\n".join(lines)


def build_image_labels_each_class(images: List[dict], target_organ: str) -> str:
    if not images:
        return ""
    lines = []
    for item in images:
        image_type = item.get("image_type", "mask_overlay")
        if image_type == "raw":
            lines.append(f"- Image: raw; Slice: {item['slice_number']}")
        else:
            lines.append(
                f"- Image: mask_overlay; Organ: {target_organ}; Slice: {item['slice_number']}"
            )
    return "\n".join(lines)


def save_prompt_preview(case_dir: str, prompt: str, images: List[dict], name: str = "prompt_preview") -> None:
    preview_path = os.path.join(case_dir, f"{name}.html")
    parts = [
        "<html><head><meta charset='utf-8'><title>VLM Prompt Preview</title></head><body>",
        "<h2>Prompt</h2>",
        "<pre style='white-space: pre-wrap; font-family: monospace;'>",
        prompt,
        "</pre>",
        "<h2>Images</h2>",
    ]
    for item in images:
        rel_path = os.path.relpath(item["path"], case_dir)
        image_type = item.get("image_type", "mask_overlay")
        label = f"Organ: {item['organ_name']}; Slice: {item['slice_number']}; Type: {image_type}"
        parts.append(f"<div style='margin-bottom:16px;'><div>{label}</div>")
        parts.append(f"<img src='{rel_path}' style='max-width:480px; border:1px solid #ccc;' /></div>")
    parts.append("</body></html>")
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def load_config(path: str) -> PipelineConfig:
    data = read_yaml(path)
    vlm_data = data["vlm"]
    per_class = bool(vlm_data.get("per_class", False))
    fallback_prompt = str(
        vlm_data.get("prompt_template")
        or vlm_data.get("prompt_template_each_class")
        or vlm_data.get("prompt_template_hard_class")
        or ""
    )
    prompt_template_each_class = str(
        vlm_data.get("prompt_template_each_class", fallback_prompt)
    )
    prompt_template_hard_class = str(
        vlm_data.get("prompt_template_hard_class", fallback_prompt)
    )
    require_nonempty_hard_classes = bool(vlm_data.get("require_nonempty_hard_classes", True))
    save_each_class_prompt_preview_first_bin = bool(
        vlm_data.get("save_each_class_prompt_preview_first_bin", False)
    )
    force_nonempty_descriptions_for_hard_classes = bool(
        vlm_data.get("force_nonempty_descriptions_for_hard_classes", False)
    )
    descriptions_only_for_hard_classes = bool(
        vlm_data.get("descriptions_only_for_hard_classes", False)
    )
    auto_generate_each_class_3words = bool(
        data["pipeline"].get("auto_generate_each_class_3words", False)
    )
    auto_fill_missing_cases_after_run = bool(
        data["pipeline"].get("auto_fill_missing_cases_after_run", False)
    )
    auto_fill_max_rounds = int(data["pipeline"].get("auto_fill_max_rounds", 1))
    auto_fill_sleep_seconds = float(data["pipeline"].get("auto_fill_sleep_seconds", 0.0))
    save_vlm_inputs_all = bool(data["pipeline"].get("save_vlm_inputs_all", True))
    return PipelineConfig(
        image_dir=data["dataset"]["image_dir"],
        label_dir=data["dataset"]["label_dir"],
        list_file=data["dataset"]["list_file"],
        ids=data["dataset"].get("ids", []),
        use_all_cases=bool(data["dataset"].get("use_all_cases", True)),
        single_id=str(data["dataset"].get("single_id", "")),
        output_dir=data["pipeline"]["output_dir"],
        keep_output_dir=bool(data["pipeline"].get("keep_output_dir", False)),
        plane=data["pipeline"]["plane"],
        positions_percent=data["pipeline"].get("positions_percent", []),
        positions_from_json=str(data["pipeline"].get("positions_from_json", "")),
        positions_json_key=str(data["pipeline"].get("positions_json_key", "average_slice")),
        auto_compute_positions=bool(data["pipeline"].get("auto_compute_positions", False)),
        positions_compute_gt_dir=str(
            data["pipeline"].get("positions_compute_gt_dir", "data/gt_amos_mr")
        ),
        positions_compute_output_dir=str(
            data["pipeline"].get("positions_compute_output_dir", "outputs/organ_square_distribution")
        ),
        positions_compute_window_size=int(data["pipeline"].get("positions_compute_window_size", 10)),
        run_per_bin=bool(data["pipeline"].get("run_per_bin", False)),
        use_all_slices=data["pipeline"].get("use_all_slices", False),
        save_vlm_inputs_all=save_vlm_inputs_all,
        auto_generate_each_class_3words=auto_generate_each_class_3words,
        auto_fill_missing_cases_after_run=auto_fill_missing_cases_after_run,
        auto_fill_max_rounds=auto_fill_max_rounds,
        auto_fill_sleep_seconds=auto_fill_sleep_seconds,
        overlay_alpha=float(data["pipeline"].get("overlay_alpha", 0.4)),
        vlm_name=vlm_data["name"],
        vlm_mode=vlm_data.get("mode", "export"),
        prompt_template=fallback_prompt,
        api_key_env=vlm_data.get("api_key_env", "VLM_API_KEY"),
        api_key=str(vlm_data.get("api_key", "")),
        api_key_env_map=vlm_data.get("api_key_env_map", {}),
        provider=vlm_data.get("provider", "openai"),
        model=vlm_data.get("model", vlm_data["name"]),
        api_base=vlm_data.get("api_base", ""),
        max_images=int(vlm_data.get("max_images", 0)),
        request_retries=int(vlm_data.get("request_retries", 3)),
        per_class=per_class,
        prompt_template_each_class=prompt_template_each_class,
        prompt_template_hard_class=prompt_template_hard_class,
        require_nonempty_hard_classes=require_nonempty_hard_classes,
        save_each_class_prompt_preview_first_bin=save_each_class_prompt_preview_first_bin,
        force_nonempty_descriptions_for_hard_classes=force_nonempty_descriptions_for_hard_classes,
        descriptions_only_for_hard_classes=descriptions_only_for_hard_classes,
    )


def center_for_plane(info: Dict[str, float], plane: str) -> Tuple[float, float]:
    if plane == "axial":
        return info["center_x"], info["center_y"]
    if plane == "coronal":
        return info["center_x"], info["center_z"]
    if plane == "sagittal":
        return info["center_y"], info["center_z"]
    return info["center_x"], info["center_y"]


def mock_vlm_output(
    stats: Dict[int, Dict[str, float]], plane_shape: Tuple[int, int], plane: str
) -> Dict[str, object]:
    volumes = [stats[k]["volume"] for k in stats]
    if not volumes:
        return {"hard_classes": [], "descriptions": {}}
    threshold = np.percentile(volumes, 30)
    hard = [ORGAN_MAP[k] for k in stats if stats[k]["volume"] <= threshold and stats[k]["volume"] > 0]
    descriptions = {}
    for organ_id, info in stats.items():
        name = ORGAN_MAP[organ_id]
        if name not in hard:
            continue
        size = "小"
        if info["volume"] > threshold * 1.8:
            size = "中"
        pos = describe_position(center_for_plane(info, plane), plane_shape)
        descriptions[name] = f"形态轮廓较细碎，整体大小偏{size}，主要位于{pos}。"
    return {"hard_classes": hard, "descriptions": descriptions}


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def downsample_images(items: List[dict], max_images: int) -> List[dict]:
    if max_images <= 0 or len(items) <= max_images:
        return items
    step = max(1, len(items) / max_images)
    selected = []
    index = 0.0
    while int(round(index)) < len(items) and len(selected) < max_images:
        selected.append(items[int(round(index))])
        index += step
    return selected


def normalize_description_three_words(text: str) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
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


def auto_generate_each_class_3words(each_class_dir: str) -> Tuple[int, int, str]:
    dst_dir = os.path.join(os.path.dirname(each_class_dir), "each_class_3words")
    ensure_dir(dst_dir)
    names = sorted([n for n in os.listdir(each_class_dir) if n.endswith(".json")])
    converted = 0
    skipped = 0
    for name in names:
        src_path = os.path.join(each_class_dir, name)
        dst_path = os.path.join(dst_dir, name)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue
        descriptions = data.get("descriptions")
        if not isinstance(descriptions, dict):
            skipped += 1
            continue
        data["descriptions"] = {
            organ_name: normalize_description_three_words(text)
            for organ_name, text in descriptions.items()
        }
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        converted += 1
    return converted, skipped, dst_dir


def build_retry_session(retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def format_http_error(resp: requests.Response, provider: str) -> str:
    status = resp.status_code
    url = resp.url
    body = ""
    error_type = ""
    error_code = ""
    error_message = ""

    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error", {})
            if isinstance(err, dict):
                error_type = str(err.get("type", "") or "")
                error_code = str(err.get("code", "") or "")
                error_message = str(err.get("message", "") or "")
        body = json.dumps(data, ensure_ascii=False)
    except Exception:
        body = (resp.text or "").strip()

    if len(body) > 800:
        body = body[:800] + "...(truncated)"

    details = []
    if error_type:
        details.append(f"type={error_type}")
    if error_code:
        details.append(f"code={error_code}")
    if error_message:
        details.append(f"message={error_message}")
    detail_text = ", ".join(details)
    if detail_text:
        detail_text = f"; {detail_text}"

    return (
        f"{provider} API error: status={status}, url={url}{detail_text}; "
        f"body={body}"
    )


def progress_iter(items: List[str], enabled: bool):
    if not enabled:
        for idx, item in enumerate(items, start=1):
            yield idx, item
        return
    total = max(1, len(items))
    bar_width = 30
    for idx, item in enumerate(items, start=1):
        filled = int(round(bar_width * idx / total))
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\rProgress: [{bar}] {idx}/{total} (case {item})")
        sys.stdout.flush()
        yield idx, item
    sys.stdout.write("\n")


def call_openai(
    prompt: str,
    images: List[dict],
    api_key: str,
    model: str,
    api_base: str,
    request_retries: int,
) -> Dict[str, object]:
    base = api_base or "https://api.openai.com"
    url = f"{base.rstrip('/')}/v1/responses"
    content = [{"type": "input_text", "text": prompt}]
    for item in images:
        b64 = image_to_base64(item["path"])
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
    payload = {"model": model, "input": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {api_key}"}
    session = build_retry_session(request_retries)
    resp = session.post(url, headers=headers, json=payload, timeout=300)
    if not resp.ok:
        raise RuntimeError(format_http_error(resp, "openai"))
    data = resp.json()
    text = data.get("output_text", "")
    if not text:
        outputs = data.get("output", [])
        if outputs:
            parts = outputs[0].get("content", [])
            text = "".join([p.get("text", "") for p in parts if p.get("type") == "output_text"])
    return {"raw": data, "text": text}


def call_gemini(
    prompt: str,
    images: List[dict],
    api_key: str,
    model: str,
    api_base: str,
    request_retries: int,
) -> Dict[str, object]:
    base = api_base or "https://generativelanguage.googleapis.com"
    url = f"{base.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
    parts = [{"text": prompt}]
    for item in images:
        b64 = image_to_base64(item["path"])
        parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
    payload = {"contents": [{"role": "user", "parts": parts}]}
    session = build_retry_session(request_retries)
    resp = session.post(url, json=payload, timeout=300)
    if not resp.ok:
        raise RuntimeError(format_http_error(resp, "gemini"))
    data = resp.json()
    text = ""
    candidates = data.get("candidates", [])
    if candidates:
        candidate_parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join([p.get("text", "") for p in candidate_parts if "text" in p])
    return {"raw": data, "text": text}


def parse_json_text(text: str) -> Tuple[bool, Dict[str, object]]:
    if not text:
        return False, {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False, {}
    if not isinstance(parsed, dict):
        return False, {}
    return True, parsed


def extract_description_from_text(text: str, organ_name: str) -> str:
    ok, parsed = parse_json_text(text)
    if ok:
        if "description" in parsed:
            return str(parsed.get("description", "")).strip()
        if organ_name in parsed:
            return str(parsed.get(organ_name, "")).strip()
    cleaned = text.strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    # Sometimes the model returns escaped quotes like "\"Morphology: ...\""
    if cleaned.startswith('\\"') and cleaned.endswith('\\"') and len(cleaned) >= 4:
        cleaned = cleaned[2:-2].strip()
    return cleaned


def force_nonempty_description_prompt(prompt: str) -> str:
    return (
        prompt
        + "\nIMPORTANT OVERRIDE: You MUST output a NON-EMPTY description in the required template. "
        + "Do NOT output an empty string.\n"
    )


def normalize_hard_classes(
    hard_classes: List[str], descriptions: Dict[str, str], organ_map: Dict[int, str]
) -> List[str]:
    # Prefer classes that have non-empty descriptions, but always return 5 items.
    keep: List[str] = []
    for name in hard_classes:
        if name in keep:
            continue
        if descriptions.get(name):
            keep.append(name)
        if len(keep) >= 5:
            return keep[:5]

    # Fill with other organs that have non-empty descriptions.
    for organ_id in organ_map:
        name = organ_map[organ_id]
        if name in keep:
            continue
        if descriptions.get(name):
            keep.append(name)
        if len(keep) >= 5:
            return keep[:5]

    # If still not enough, fall back to original hard_classes order (even if description empty).
    for name in hard_classes:
        if name not in keep:
            keep.append(name)
        if len(keep) >= 5:
            return keep[:5]

    # Absolute fallback: fill from organ_map order.
    for organ_id in organ_map:
        name = organ_map[organ_id]
        if name not in keep:
            keep.append(name)
        if len(keep) >= 5:
            return keep[:5]

    return keep


def filter_descriptions_by_hard_classes(
    hard_classes: List[str], descriptions: Dict[str, str], enabled: bool
) -> Dict[str, str]:
    if not enabled:
        return descriptions
    return {name: descriptions.get(name, "") for name in hard_classes}


def get_organ_input_overlay_path(case_entry: dict, organ_id: int, slice_number: int) -> str:
    for organ_entry in case_entry.get("organs", []):
        if int(organ_entry.get("id", -1)) != int(organ_id):
            continue
        slices = organ_entry.get("slice_indices", [])
        input_paths = organ_entry.get("input_image_paths", [])
        if slice_number in slices:
            idx = slices.index(slice_number)
            if 0 <= idx < len(input_paths):
                return str(input_paths[idx])
    return ""


def infer_bin(
    cfg: PipelineConfig,
    api_key_value: str,
    case_id: str,
    bin_entry: dict,
    case_dir: str,
    slice_number: int,
    slice_start: int,
    slice_end: int,
    total_slices: int,
    all_images_full: List[dict],
    case_entry: dict,
    raw_slice_paths: Dict[int, Dict[str, str]],
    all_slices_dir: str,
    outputs_dir: str,
) -> Tuple[str, str]:
    images_for_bin = [img for img in all_images_full if img["slice_number"] == slice_number]
    all_images = downsample_images(images_for_bin, cfg.max_images)
    hard_prompt = build_prompt(
        cfg.prompt_template_hard_class if cfg.per_class else cfg.prompt_template,
        cfg.plane,
        ORGAN_MAP,
        image_labels=build_image_labels(all_images),
    )
    if cfg.provider == "gemini":
        hard_result = call_gemini(
            hard_prompt, all_images, api_key_value, cfg.model, cfg.api_base, cfg.request_retries
        )
    elif cfg.provider == "openai":
        hard_result = call_openai(
            hard_prompt, all_images, api_key_value, cfg.model, cfg.api_base, cfg.request_retries
        )
    else:
        raise ValueError(f"Unsupported provider: {cfg.provider}")

    hard_ok, hard_parsed = parse_json_text(hard_result["text"])
    hard_classes = hard_parsed.get("hard_classes", []) if hard_ok else []
    output_path = os.path.join(outputs_dir, build_slice_range_name(case_id, slice_start, slice_end))

    if cfg.per_class:
        descriptions = {}
        range_str = str(bin_entry.get("range", ""))
        safe_label = (range_str or f"slice_{slice_number}").replace("%", "").replace(" ", "").replace("-", "_")
        save_each_class_preview = (
            cfg.save_each_class_prompt_preview_first_bin and is_first_decile(range_str)
        )
        for organ_entry in case_entry["organs"]:
            organ_name = organ_entry["name"]
            organ_id = organ_entry["id"]
            raw_path = raw_slice_paths[slice_number]["input_path"]
            mask_path = get_organ_input_overlay_path(case_entry, organ_id, slice_number)
            organ_images = [
                {
                    "path": raw_path,
                    "organ_name": "Raw",
                    "slice_number": slice_number,
                    "image_type": "raw",
                },
                {
                    "path": mask_path,
                    "organ_name": organ_name,
                    "slice_number": slice_number,
                    "image_type": "mask_overlay",
                },
            ]
            organ_prompt = build_prompt(
                cfg.prompt_template_each_class,
                cfg.plane,
                ORGAN_MAP,
                image_labels=build_image_labels_each_class(organ_images, target_organ=organ_name),
                target_organ=organ_name,
                slice_number=str(slice_number),
            )
            if save_each_class_preview:
                save_prompt_preview(
                    case_dir,
                    organ_prompt,
                    organ_images,
                    name=f"prompt_preview_{safe_label}_desc_organ{organ_id:02d}",
                )
            if cfg.provider == "gemini":
                organ_result = call_gemini(
                    organ_prompt,
                    organ_images,
                    api_key_value,
                    cfg.model,
                    cfg.api_base,
                    cfg.request_retries,
                )
            elif cfg.provider == "openai":
                organ_result = call_openai(
                    organ_prompt,
                    organ_images,
                    api_key_value,
                    cfg.model,
                    cfg.api_base,
                    cfg.request_retries,
                )
            else:
                raise ValueError(f"Unsupported provider: {cfg.provider}")
            descriptions[organ_name] = extract_description_from_text(organ_result["text"], organ_name)

        # Ensure exactly 5 hard classes, and (optionally) ensure their descriptions are non-empty.
        desired = 5
        final_hard: List[str] = []
        # Preserve model order for primary candidates.
        primary_candidates: List[str] = []
        for name in hard_classes:
            if name not in primary_candidates:
                primary_candidates.append(name)

        def ensure_desc(name: str) -> str:
            current = descriptions.get(name, "")
            if current:
                return current
            if not cfg.force_nonempty_descriptions_for_hard_classes:
                return ""
            # Re-call VLM with a forced-nonempty prompt.
            # Reuse the same two images (raw + this organ overlay) for this slice.
            organ_id = next((oid for oid, oname in ORGAN_MAP.items() if oname == name), None)
            if organ_id is None:
                return ""
            raw_path = raw_slice_paths[slice_number]["input_path"]
            mask_path = get_organ_input_overlay_path(case_entry, organ_id, slice_number)
            organ_images = [
                {"path": raw_path, "organ_name": "Raw", "slice_number": slice_number, "image_type": "raw"},
                {"path": mask_path, "organ_name": name, "slice_number": slice_number, "image_type": "mask_overlay"},
            ]
            base_prompt = build_prompt(
                cfg.prompt_template_each_class,
                cfg.plane,
                ORGAN_MAP,
                image_labels=build_image_labels_each_class(organ_images, target_organ=name),
                target_organ=name,
                slice_number=str(slice_number),
            )
            forced_prompt = force_nonempty_description_prompt(base_prompt)
            if cfg.provider == "gemini":
                organ_result = call_gemini(
                    forced_prompt,
                    organ_images,
                    api_key_value,
                    cfg.model,
                    cfg.api_base,
                    cfg.request_retries,
                )
            elif cfg.provider == "openai":
                organ_result = call_openai(
                    forced_prompt,
                    organ_images,
                    api_key_value,
                    cfg.model,
                    cfg.api_base,
                    cfg.request_retries,
                )
            else:
                raise ValueError(f"Unsupported provider: {cfg.provider}")
            forced_desc = extract_description_from_text(organ_result["text"], name)
            descriptions[name] = forced_desc
            return forced_desc

        for name in primary_candidates:
            if len(final_hard) >= desired:
                break
            if cfg.require_nonempty_hard_classes:
                desc = ensure_desc(name)
                if not desc:
                    continue
            if name not in final_hard:
                final_hard.append(name)

        # Secondary fill: use other organs ONLY if they already have non-empty descriptions
        # (avoid inventing new organs just to fill the list).
        if len(final_hard) < desired:
            for organ_id in ORGAN_MAP:
                name = ORGAN_MAP[organ_id]
                if name in final_hard:
                    continue
                if cfg.require_nonempty_hard_classes and not descriptions.get(name):
                    continue
                final_hard.append(name)
                if len(final_hard) >= desired:
                    break

        hard_classes = final_hard[:desired]
        output_descriptions = filter_descriptions_by_hard_classes(
            hard_classes, descriptions, cfg.descriptions_only_for_hard_classes
        )
        combined = {"hard_classes": hard_classes, "descriptions": output_descriptions}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hard_result["raw"], f, ensure_ascii=False, indent=2)
    return output_path, hard_result["text"]


def run_pipeline(config_path: str) -> None:
    cfg = load_config(config_path)
    if cfg.use_all_cases and cfg.per_class:
        cfg.output_dir = os.path.join(cfg.output_dir, "all_case")
    if (
        cfg.use_all_slices
        and not cfg.keep_output_dir
        and not os.path.basename(cfg.output_dir).startswith("full_sequence_")
    ):
        cfg.output_dir = os.path.join(
            os.path.dirname(cfg.output_dir),
            f"full_sequence_{os.path.basename(cfg.output_dir)}",
        )
    if not cfg.use_all_cases:
        case_ids = resolve_case_ids(cfg)
        if len(case_ids) == 1:
            case_suffix = f"case_{case_ids[0]}"
            if not cfg.keep_output_dir and not os.path.basename(cfg.output_dir).startswith(
                "selected_slices_case_"
            ):
                cfg.output_dir = os.path.join(
                    os.path.dirname(cfg.output_dir),
                    f"selected_slices_case_{case_ids[0]}",
                )
    if (
        cfg.use_all_cases
        and not cfg.keep_output_dir
        and not os.path.basename(cfg.output_dir).startswith("all_cases_")
    ):
        cfg.output_dir = os.path.join(
            os.path.dirname(cfg.output_dir),
            f"all_cases_{os.path.basename(cfg.output_dir)}",
        )
    ensure_dir(cfg.output_dir)
    all_slices_dir = os.path.join(cfg.output_dir, "vlm_inputs_all")
    if cfg.save_vlm_inputs_all:
        ensure_dir(all_slices_dir)
    ids = resolve_case_ids(cfg)
    env_name = cfg.api_key_env_map.get(cfg.vlm_name, cfg.api_key_env)
    api_key_value = cfg.api_key or os.getenv(env_name, "")
    key_source = "config" if cfg.api_key else ("env" if api_key_value else "missing")
    outputs_dir = os.path.join(cfg.output_dir, "each_class" if cfg.per_class else "vlm_outputs")
    ensure_dir(outputs_dir)
    manifest = {
        "vlm_name": cfg.vlm_name,
        "vlm_provider": cfg.provider,
        "vlm_model": cfg.model,
        "api_key_source": key_source,
        "api_key_env": env_name,
        "cases": [],
    }

    for _, case_id in progress_iter(ids, cfg.use_all_cases):
        image_path = os.path.join(cfg.image_dir, f"{case_id}.nii.gz")
        label_path = os.path.join(cfg.label_dir, f"{case_id}.nii.gz")
        if not (os.path.exists(image_path) and os.path.exists(label_path)):
            continue
        image = normalize_to_uint8(load_nifti(image_path))
        label = load_nifti(label_path).astype(np.int32)

        total_slices = slice_count_for_plane(image, cfg.plane)
        bins_from_json = []
        if cfg.use_all_slices:
            slice_numbers = list(range(1, total_slices + 1))
        else:
            slice_numbers = []
            if cfg.positions_from_json:
                ensure_positions_json(cfg, case_id)
                json_path = cfg.positions_from_json.format(case_id=case_id)
                if cfg.run_per_bin:
                    bins_from_json = load_bins_from_json(
                        json_path, cfg.positions_json_key, total_slices
                    )
                    slice_numbers = sorted({b["slice_number"] for b in bins_from_json})
                else:
                    slice_numbers = load_slice_numbers_from_json(
                        json_path, cfg.positions_json_key, total_slices
                    )
            if not slice_numbers:
                slice_numbers = pick_slice_numbers(total_slices, cfg.positions_percent)
        if not slice_numbers:
            slice_numbers = [max(1, total_slices // 2)]

        case_dir = os.path.join(cfg.output_dir, f"case_{case_id}")
        ensure_dir(case_dir)
        case_entry = {"case_id": case_id, "plane": cfg.plane, "organs": []}

        organ_stats = {}
        plane_shape = None
        raw_dir = os.path.join(case_dir, "raw")
        ensure_dir(raw_dir)
        raw_slice_paths = {}
        raw_slice_entries = []
        for slice_number in slice_numbers:
            slice_index = slice_number - 1
            image_slice = extract_slice(image, cfg.plane, slice_index)
            if plane_shape is None:
                plane_shape = image_slice.shape
            raw_image = transform_output(Image.fromarray(image_slice))
            raw_filename = f"slice_{slice_number:04d}_raw.png"
            raw_out_path = os.path.join(raw_dir, raw_filename)
            raw_image.save(raw_out_path)
            raw_input_path = raw_out_path
            if cfg.save_vlm_inputs_all:
                raw_all_name = f"{case_id}_slice_{slice_number:04d}_raw.png"
                raw_all_path = os.path.join(all_slices_dir, raw_all_name)
                raw_image.save(raw_all_path)
                raw_input_path = raw_all_path
            raw_slice_paths[slice_number] = {
                "case_path": raw_out_path,
                "input_path": raw_input_path,
            }
            raw_slice_entries.append({"slice_number": slice_number, "path": raw_out_path})
        case_entry["raw_slices"] = raw_slice_entries

        for organ_id, organ_name in ORGAN_MAP.items():
            organ_mask = (label == organ_id).astype(np.uint8)
            organ_dir = os.path.join(case_dir, f"{organ_id}_{organ_name.replace(' ', '_')}")
            ensure_dir(organ_dir)
            image_paths = []
            input_image_paths = []
            raw_image_paths = []

            for slice_number in slice_numbers:
                slice_index = slice_number - 1
                image_slice = extract_slice(image, cfg.plane, slice_index)
                mask_slice = extract_slice(organ_mask, cfg.plane, slice_index)
                if plane_shape is None:
                    plane_shape = image_slice.shape
                overlay = mask_overlay(image_slice, mask_slice, cfg.overlay_alpha)
                filename = f"slice_{slice_number:04d}.png"
                out_path = os.path.join(organ_dir, filename)
                overlay.save(out_path)
                input_path = out_path
                if cfg.save_vlm_inputs_all:
                    all_name = f"{case_id}_organ{organ_id:02d}_slice_{slice_number:04d}.png"
                    input_path = os.path.join(all_slices_dir, all_name)
                    overlay.save(input_path)
                image_paths.append(out_path)
                input_image_paths.append(input_path)
                raw_image_paths.append(raw_slice_paths[slice_number]["case_path"])

            organ_stats[organ_id] = compute_mask_stats(organ_mask)
            case_entry["organs"].append(
                {
                    "id": organ_id,
                    "name": organ_name,
                    "slice_indices": slice_numbers,
                    "image_paths": image_paths,
                    "input_image_paths": input_image_paths,
                    "raw_image_paths": raw_image_paths,
                }
            )

        all_images_full = []
        for slice_number in slice_numbers:
            all_images_full.append(
                {
                    "path": raw_slice_paths[slice_number]["input_path"],
                    "organ_name": "Raw",
                    "slice_number": slice_number,
                    "image_type": "raw",
                }
            )
        for organ_entry in case_entry["organs"]:
            for idx, path in enumerate(organ_entry["input_image_paths"]):
                all_images_full.append(
                    {
                        "path": path,
                        "organ_name": organ_entry["name"],
                        "slice_number": organ_entry["slice_indices"][idx],
                        "image_type": "mask_overlay",
                    }
                )
        if cfg.run_per_bin and bins_from_json:
            case_entry["bins"] = []
            for bin_entry in bins_from_json:
                slice_number = bin_entry["slice_number"]
                slice_start, slice_end = parse_percent_range(bin_entry.get("range", ""), total_slices)
                if slice_start == 0 or slice_end == 0:
                    slice_start = slice_number
                    slice_end = slice_number
                images_for_bin = [img for img in all_images_full if img["slice_number"] == slice_number]
                all_images = downsample_images(images_for_bin, cfg.max_images)
                hard_prompt = build_prompt(
                    cfg.prompt_template_hard_class if cfg.per_class else cfg.prompt_template,
                    cfg.plane,
                    ORGAN_MAP,
                    image_labels=build_image_labels(all_images),
                )
                bin_label = bin_entry["range"] or f"slice_{slice_number}"
                safe_label = bin_label.replace("%", "").replace(" ", "").replace("-", "_")
                save_prompt_preview(case_dir, hard_prompt, all_images, name=f"prompt_preview_{safe_label}")
                bin_result = {
                    "range": bin_entry["range"],
                    "slice_number": slice_number,
                    "prompt": hard_prompt,
                }
                if cfg.vlm_mode == "mock":
                    bin_result["mock_output"] = mock_vlm_output(organ_stats, plane_shape, cfg.plane)
                elif cfg.vlm_mode == "infer":
                    if not api_key_value:
                        bin_result["vlm_status"] = "missing_api_key"
                    else:
                        try:
                            output_path, output_text = infer_bin(
                                cfg,
                                api_key_value,
                                case_id,
                                bin_entry,
                                case_dir,
                                slice_number,
                                slice_start,
                                slice_end,
                                total_slices,
                                all_images_full,
                                case_entry,
                                raw_slice_paths,
                                all_slices_dir,
                                outputs_dir,
                            )
                            bin_result["vlm_status"] = "ok"
                            bin_result["vlm_output_path"] = output_path
                            if not cfg.per_class:
                                bin_result["vlm_output_text"] = output_text
                        except Exception as e:
                            # Do not abort the whole all-cases run; missing-bin validation below will retry.
                            bin_result["vlm_status"] = "error"
                            bin_result["vlm_error"] = str(e)
                case_entry["bins"].append(bin_result)
            if cfg.vlm_mode == "infer" and api_key_value:
                missing_bins = []
                for bin_entry in bins_from_json:
                    slice_number = bin_entry["slice_number"]
                    slice_start, slice_end = parse_percent_range(
                        bin_entry.get("range", ""), total_slices
                    )
                    if slice_start == 0 or slice_end == 0:
                        slice_start = slice_number
                        slice_end = slice_number
                    output_path = os.path.join(
                        outputs_dir, build_slice_range_name(case_id, slice_start, slice_end)
                    )
                    if not os.path.exists(output_path):
                        missing_bins.append((bin_entry, slice_number, slice_start, slice_end))
                if missing_bins:
                    for bin_entry, slice_number, slice_start, slice_end in missing_bins:
                        try:
                            infer_bin(
                                cfg,
                                api_key_value,
                                case_id,
                                bin_entry,
                                case_dir,
                                slice_number,
                                slice_start,
                                slice_end,
                                total_slices,
                                all_images_full,
                                case_entry,
                                raw_slice_paths,
                                all_slices_dir,
                                outputs_dir,
                            )
                        except Exception:
                            # Leave missing output for a later rerun; continue other bins/cases.
                            pass
        else:
            all_images = downsample_images(all_images_full, cfg.max_images)
            prompt = build_prompt(
                cfg.prompt_template_hard_class if cfg.per_class else cfg.prompt_template,
                cfg.plane,
                ORGAN_MAP,
                image_labels=build_image_labels(all_images),
            )
            case_entry["prompt"] = prompt
            save_prompt_preview(case_dir, prompt, all_images)
            if cfg.vlm_mode == "mock":
                case_entry["mock_output"] = mock_vlm_output(organ_stats, plane_shape, cfg.plane)
            elif cfg.vlm_mode == "infer":
                if not api_key_value:
                    case_entry["vlm_status"] = "missing_api_key"
                else:
                    if cfg.provider == "gemini":
                        hard_result = call_gemini(
                            prompt, all_images, api_key_value, cfg.model, cfg.api_base, cfg.request_retries
                        )
                    elif cfg.provider == "openai":
                        hard_result = call_openai(
                            prompt, all_images, api_key_value, cfg.model, cfg.api_base, cfg.request_retries
                        )
                    else:
                        raise ValueError(f"Unsupported provider: {cfg.provider}")

                    hard_ok, hard_parsed = parse_json_text(hard_result["text"])
                    hard_classes = hard_parsed.get("hard_classes", []) if hard_ok else []

                    if cfg.per_class:
                        descriptions = {}
                        for organ_entry in case_entry["organs"]:
                            organ_name = organ_entry["name"]
                            organ_id = organ_entry["id"]
                            raw_path = raw_slice_paths[slice_numbers[0]]["input_path"]
                            mask_path = get_organ_input_overlay_path(
                                case_entry, organ_id, slice_numbers[0]
                            )
                            organ_images = [
                                {
                                    "path": raw_path,
                                    "organ_name": "Raw",
                                    "slice_number": slice_numbers[0],
                                    "image_type": "raw",
                                },
                                {
                                    "path": mask_path,
                                    "organ_name": organ_name,
                                    "slice_number": slice_numbers[0],
                                    "image_type": "mask_overlay",
                                },
                            ]
                            organ_prompt = build_prompt(
                                cfg.prompt_template_each_class,
                                cfg.plane,
                                ORGAN_MAP,
                                image_labels=build_image_labels_each_class(
                                    organ_images, target_organ=organ_name
                                ),
                                target_organ=organ_name,
                                slice_number=str(slice_numbers[0]),
                            )
                            if cfg.provider == "gemini":
                                organ_result = call_gemini(
                                    organ_prompt,
                                    organ_images,
                                    api_key_value,
                                    cfg.model,
                                    cfg.api_base,
                                    cfg.request_retries,
                                )
                            elif cfg.provider == "openai":
                                organ_result = call_openai(
                                    organ_prompt,
                                    organ_images,
                                    api_key_value,
                                    cfg.model,
                                    cfg.api_base,
                                    cfg.request_retries,
                                )
                            else:
                                raise ValueError(f"Unsupported provider: {cfg.provider}")
                            descriptions[organ_name] = extract_description_from_text(
                                organ_result["text"], organ_name
                            )
                        hard_classes = normalize_hard_classes(hard_classes, descriptions, ORGAN_MAP)
                        output_descriptions = filter_descriptions_by_hard_classes(
                            hard_classes, descriptions, cfg.descriptions_only_for_hard_classes
                        )
                        combined = {
                            "hard_classes": hard_classes,
                            "descriptions": output_descriptions,
                        }
                        output_path = os.path.join(outputs_dir, f"{case_id}.json")
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(combined, f, ensure_ascii=False, indent=2)
                        case_entry["vlm_status"] = "ok"
                        case_entry["vlm_output_path"] = output_path
                    else:
                        output_path = os.path.join(outputs_dir, f"{case_id}.json")
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(hard_result["raw"], f, ensure_ascii=False, indent=2)
                        case_entry["vlm_status"] = "ok"
                        case_entry["vlm_output_path"] = output_path
                        case_entry["vlm_output_text"] = hard_result["text"]
        manifest["cases"].append(case_entry)

    manifest_path = os.path.join(cfg.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if cfg.per_class and cfg.auto_generate_each_class_3words and os.path.isdir(outputs_dir):
        converted, skipped, dst_dir = auto_generate_each_class_3words(outputs_dir)
        print(f"[3words] 输出目录: {dst_dir}; 转换={converted}; 跳过={skipped}")

    # Optional one-click补漏: trigger fill_missing_cases automatically after main run.
    # Use env guard to avoid recursive auto-fill when run_pipeline is launched by fill script.
    skip_auto_fill = os.getenv("VLM_SKIP_AUTO_FILL", "0") == "1"
    if (
        cfg.vlm_mode == "infer"
        and cfg.auto_fill_missing_cases_after_run
        and not skip_auto_fill
    ):
        cmd = [
            sys.executable,
            os.path.join("src", "fill_missing_cases.py"),
            "--config",
            config_path,
            "--run",
            "--max-rounds",
            str(max(1, int(cfg.auto_fill_max_rounds))),
            "--sleep-seconds",
            str(max(0.0, float(cfg.auto_fill_sleep_seconds))),
        ]
        print("[auto-fill] 主流程结束，开始自动补漏:", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if int(proc.returncode) != 0:
            print(f"[auto-fill] 补漏脚本返回非0: {proc.returncode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VLM MRI pipeline")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()
    run_pipeline(args.config)
