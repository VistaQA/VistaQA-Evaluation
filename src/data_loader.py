from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SAESample:
    image_id: int
    question: str
    gt_answer: str
    task_type: str
    domain: str
    is_hallucination: bool
    gt_masks: list[dict[str, Any]]
    pred_text: str
    pred_masks: list[dict[str, Any]]
    image_width: Optional[int] = None
    image_height: Optional[int] = None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_int(value: Any, field_name: str, context: str) -> int:
    parsed = _to_int(value)
    if parsed is None:
        raise ValueError(f"{context}: '{field_name}' must be an integer")
    return parsed


def _optional_positive_int(value: Any, field_name: str, context: str) -> Optional[int]:
    if value is None:
        return None
    parsed = _to_int(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{context}: '{field_name}' must be a positive integer when provided")
    return parsed


def _require_str(value: Any, field_name: str, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context}: '{field_name}' must be a string")
    return value


def _require_bool(value: Any, field_name: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no"}:
            return False
        if lowered in {"1", "true", "yes"}:
            return True
    raise ValueError(f"{context}: '{field_name}' must be bool-compatible (0/1/true/false)")


def _normalize_rle_item(rle_item: Any, context: str) -> dict[str, Any]:
    if not isinstance(rle_item, dict):
        raise ValueError(f"{context}: RLE mask must be an object")

    if "size" not in rle_item or "counts" not in rle_item:
        raise ValueError(f"{context}: RLE mask must include 'size' and 'counts'")

    size = rle_item.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"{context}: RLE 'size' must be [height, width]")

    height = _require_int(size[0], "size[0]", context)
    width = _require_int(size[1], "size[1]", context)
    if height <= 0 or width <= 0:
        raise ValueError(f"{context}: RLE 'size' values must be positive")

    counts = rle_item.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8", errors="ignore")
    if not isinstance(counts, str):
        raise ValueError(f"{context}: RLE 'counts' must be a string or bytes")

    return {
        "size": [height, width],
        "counts": counts,
    }


def _load_json_file(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        raise ValueError(f"Unsupported JSON root in {file_path}: expected object or array")

    invalid = [idx + 1 for idx, item in enumerate(records) if not isinstance(item, dict)]
    if invalid:
        raise ValueError(f"{file_path}: record(s) {invalid[:5]} are not JSON objects")

    return records


def _load_jsonl_file(file_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{file_path}: line {line_idx} is not a JSON object")
            records.append(obj)
    return records


def load_records_from_path(path_str: str, logger: Any = None) -> list[dict[str, Any]]:
    _ = logger
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".json":
            return _load_json_file(path)
        if suffix == ".jsonl":
            return _load_jsonl_file(path)
        raise ValueError(f"Unsupported file type: {path}")

    json_files = sorted(path.glob("*.json"))
    jsonl_files = sorted(path.glob("*.jsonl"))
    all_files = json_files + jsonl_files
    if not all_files:
        raise ValueError(f"No .json or .jsonl files found under {path}")

    records: list[dict[str, Any]] = []
    for file_path in all_files:
        if file_path.suffix.lower() == ".json":
            records.extend(_load_json_file(file_path))
        else:
            records.extend(_load_jsonl_file(file_path))
    return records


def _build_gt_index(gt_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    gt_index: dict[int, dict[str, Any]] = {}

    for rec_idx, rec in enumerate(gt_records, start=1):
        context = f"GT record #{rec_idx}"

        image = rec.get("image")
        if not isinstance(image, dict):
            raise ValueError(f"{context}: 'image' must be an object")

        image_id = _require_int(image.get("image_id"), "image.image_id", context)
        image_width = _optional_positive_int(image.get("width"), "image.width", context)
        image_height = _optional_positive_int(image.get("height"), "image.height", context)

        question = _require_str(rec.get("question"), "question", context)
        gt_answer = _require_str(rec.get("answer"), "answer", context)
        task_type = _require_str(rec.get("task_type"), "task_type", context)
        domain = _require_str(rec.get("task_domain"), "task_domain", context)
        is_hallucination = _require_bool(rec.get("hallucination"), "hallucination", context)

        annotations = rec.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"{context}: 'annotations' must be a list")

        gt_masks: list[dict[str, Any]] = []
        for ann_idx, ann in enumerate(annotations, start=1):
            ann_context = f"{context} annotations[{ann_idx}]"
            if not isinstance(ann, dict):
                raise ValueError(f"{ann_context}: annotation must be an object")
            if "segmentation" not in ann:
                raise ValueError(f"{ann_context}: missing 'segmentation'")
            gt_masks.append(_normalize_rle_item(ann.get("segmentation"), f"{ann_context}.segmentation"))

        if image_id in gt_index:
            raise ValueError(f"Duplicate GT image.image_id: {image_id}")

        gt_index[image_id] = {
            "image_id": image_id,
            "question": question,
            "gt_answer": gt_answer,
            "task_type": task_type,
            "domain": domain,
            "is_hallucination": is_hallucination,
            "gt_masks": gt_masks,
            "image_width": image_width,
            "image_height": image_height,
        }

    return gt_index


def _build_pred_index(pred_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    pred_index: dict[int, dict[str, Any]] = {}

    for rec_idx, rec in enumerate(pred_records, start=1):
        context = f"Pred record #{rec_idx}"

        image = rec.get("image")
        if not isinstance(image, dict):
            raise ValueError(f"{context}: 'image' must be an object")

        image_id = _require_int(image.get("image_id"), "image.image_id", context)
        image_width = _optional_positive_int(image.get("width"), "image.width", context)
        image_height = _optional_positive_int(image.get("height"), "image.height", context)

        prediction = rec.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"{context}: 'prediction' must be an object")

        pred_text = _require_str(prediction.get("text"), "prediction.text", context)
        masks = prediction.get("masks")
        if not isinstance(masks, list):
            raise ValueError(f"{context}: 'prediction.masks' must be a list")

        pred_masks: list[dict[str, Any]] = []
        for mask_idx, mask in enumerate(masks, start=1):
            pred_masks.append(_normalize_rle_item(mask, f"{context} prediction.masks[{mask_idx}]"))

        if image_id in pred_index:
            raise ValueError(f"Duplicate Pred image.image_id: {image_id}")

        pred_index[image_id] = {
            "image_id": image_id,
            "pred_text": pred_text,
            "pred_masks": pred_masks,
            "image_width": image_width,
            "image_height": image_height,
        }

    return pred_index


def _preview_ids(values: list[int], limit: int = 10) -> str:
    if len(values) <= limit:
        return ", ".join(str(x) for x in values)
    prefix = ", ".join(str(x) for x in values[:limit])
    return f"{prefix}, ... ({len(values)} total)"


def load_and_merge_samples(
    gt_path: str,
    pred_path: Optional[str],
    logger: Any = None,
) -> list[SAESample]:
    _ = logger
    if not pred_path:
        raise ValueError("pred_path is required")

    gt_records = load_records_from_path(gt_path)
    pred_records = load_records_from_path(pred_path)

    gt_index = _build_gt_index(gt_records)
    pred_index = _build_pred_index(pred_records)

    if not gt_index:
        raise ValueError("No GT records were loaded")

    gt_ids = set(gt_index.keys())
    pred_ids = set(pred_index.keys())

    missing_pred_ids = sorted(gt_ids - pred_ids)
    if missing_pred_ids:
        raise ValueError(f"Missing prediction records for image_id(s): {_preview_ids(missing_pred_ids)}")

    extra_pred_ids = sorted(pred_ids - gt_ids)
    if extra_pred_ids:
        raise ValueError(f"Prediction has unknown image_id(s) not in GT: {_preview_ids(extra_pred_ids)}")

    samples: list[SAESample] = []
    for image_id in sorted(gt_ids):
        gt_item = gt_index[image_id]
        pred_item = pred_index[image_id]

        samples.append(
            SAESample(
                image_id=image_id,
                question=gt_item["question"],
                gt_answer=gt_item["gt_answer"],
                task_type=gt_item["task_type"],
                domain=gt_item["domain"],
                is_hallucination=bool(gt_item["is_hallucination"]),
                gt_masks=gt_item["gt_masks"],
                pred_text=pred_item["pred_text"],
                pred_masks=pred_item["pred_masks"],
                image_width=gt_item["image_width"] or pred_item["image_width"],
                image_height=gt_item["image_height"] or pred_item["image_height"],
            )
        )

    return samples
