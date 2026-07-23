"""Оценка pipeline (детекция + классификация) на test-наборе."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import DEVICE, OUTPUT_ROOT, SEED, TARGET_CLASSES
from pipeline import DiatomPipeline

DETECTOR_PATH = OUTPUT_ROOT / "best_diatom.pt"
CLASSIFIER_PATH = OUTPUT_ROOT / "best_diatomnet.pth"
DATA_YAML = OUTPUT_ROOT / "data.yaml"
TEST_IMAGES_DIR = OUTPUT_ROOT / "test" / "images"
TEST_LABELS_DIR = OUTPUT_ROOT / "test" / "labels"
DEMO_OUTPUT_DIR = OUTPUT_ROOT / "demo_test"
NUM_DEMO_IMAGES = 5
IOU_THRESHOLD = 0.5

random.seed(SEED)


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict[str, Any]]:
    """Читает YOLO-разметку и возвращает bbox в абсолютных координатах."""
    objects = []
    if not label_path.exists():
        return objects

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            objects.append({
                "class_id": class_id,
                "class_name": TARGET_CLASSES[class_id],
                "box": [x1, y1, x2, y2],
            })
    return objects


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _match_predictions_to_gt(
    predictions: dict[str, Any],
    gt_objects: list[dict[str, Any]],
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, int]:
    """Сопоставляет предсказания pipeline с GT по IoU."""
    pred_boxes = predictions["boxes"]
    pred_classes = predictions["class_ids"]
    matched_gt = set()
    stats = {"det_tp": 0, "cls_correct": 0, "matched": 0}

    for pred_box, pred_cls in zip(pred_boxes, pred_classes):
        best_iou = 0.0
        best_gt_idx = None

        for idx, gt in enumerate(gt_objects):
            if idx in matched_gt:
                continue
            iou = _box_iou(pred_box, gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            stats["matched"] += 1
            stats["det_tp"] += 1
            if pred_cls == gt_objects[best_gt_idx]["class_id"]:
                stats["cls_correct"] += 1

    stats["gt_total"] = len(gt_objects)
    stats["pred_total"] = len(pred_boxes)
    stats["det_fp"] = stats["pred_total"] - stats["det_tp"]
    stats["det_fn"] = stats["gt_total"] - stats["det_tp"]
    return stats


def _draw_box(
    image: np.ndarray,
    box: list[float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image, label, (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def _save_demo_image(
    image_path: Path,
    gt_objects: list[dict[str, Any]],
    pipeline_result: dict[str, Any],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return

    gt_vis = image.copy()
    pred_vis = image.copy()

    for obj in gt_objects:
        _draw_box(gt_vis, obj["box"], f"GT: {obj['class_name']}", (0, 200, 0))

    for box, name, conf in zip(
        pipeline_result["boxes"],
        pipeline_result["class_names"],
        pipeline_result["confidences"],
    ):
        _draw_box(pred_vis, box, f"{name} ({conf:.2f})", (0, 120, 255))

    combined = np.hstack([gt_vis, pred_vis])
    cv2.putText(
        combined, "Ground Truth", (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        combined, "Pipeline", (gt_vis.shape[1] + 10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2, cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:>24}: {value:.4f}")
        else:
            print(f"{key:>24}: {value}")


def evaluate_detection(pipeline: DiatomPipeline) -> dict[str, Any]:
    return pipeline.validate_detection(
        data=DATA_YAML,
        val_cfg={"split": "test", "verbose": False},
    )


def evaluate_classification(pipeline: DiatomPipeline) -> dict[str, float]:
    return pipeline.validate_classification(dataset_root=OUTPUT_ROOT, split="test")


def evaluate_pipeline_end_to_end(pipeline: DiatomPipeline) -> dict[str, float]:
    """End-to-end оценка pipeline на всех test-изображениях с разметкой."""
    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Test images not found: {TEST_IMAGES_DIR}")

    total_stats = {"det_tp": 0, "det_fp": 0, "det_fn": 0, "cls_correct": 0, "gt_total": 0}

    image_files = sorted(TEST_IMAGES_DIR.iterdir())
    for image_path in image_files:
        if not image_path.is_file():
            continue

        label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        gt_objects = _parse_yolo_labels(label_path, img_w, img_h)
        if not gt_objects:
            continue

        predictions = pipeline.predict(image_path)
        stats = _match_predictions_to_gt(predictions, gt_objects)

        for key in total_stats:
            total_stats[key] += stats[key]

    det_precision = (
        total_stats["det_tp"] / (total_stats["det_tp"] + total_stats["det_fp"])
        if (total_stats["det_tp"] + total_stats["det_fp"]) > 0 else 0.0
    )
    det_recall = (
        total_stats["det_tp"] / (total_stats["det_tp"] + total_stats["det_fn"])
        if (total_stats["det_tp"] + total_stats["det_fn"]) > 0 else 0.0
    )
    cls_accuracy = (
        total_stats["cls_correct"] / total_stats["det_tp"]
        if total_stats["det_tp"] > 0 else 0.0
    )

    return {
        "gt_objects": total_stats["gt_total"],
        "det_tp": total_stats["det_tp"],
        "det_fp": total_stats["det_fp"],
        "det_fn": total_stats["det_fn"],
        "det_precision": det_precision,
        "det_recall": det_recall,
        "cls_accuracy_on_matched": cls_accuracy,
    }


def run_demo(pipeline: DiatomPipeline, num_images: int = NUM_DEMO_IMAGES) -> None:
    """Визуальная демонстрация pipeline на нескольких test-изображениях."""
    if not TEST_IMAGES_DIR.exists():
        print(f"Demo skipped: {TEST_IMAGES_DIR} not found")
        return

    candidates = [
        p for p in sorted(TEST_IMAGES_DIR.iterdir())
        if p.is_file() and (TEST_LABELS_DIR / f"{p.stem}.txt").exists()
    ]
    if not candidates:
        print("Demo skipped: no labeled test images found")
        return

    demo_images = random.sample(candidates, min(num_images, len(candidates)))
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Демонстрация pipeline на {len(demo_images)} test-изображениях")
    print(f"Результаты сохраняются в: {DEMO_OUTPUT_DIR}")
    print("=" * 60)

    for image_path in demo_images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        gt_objects = _parse_yolo_labels(TEST_LABELS_DIR / f"{image_path.stem}.txt", img_w, img_h)
        predictions = pipeline.predict(image_path)
        stats = _match_predictions_to_gt(predictions, gt_objects)

        print(f"\nИзображение: {image_path.name}")
        print(f"  GT объектов: {stats['gt_total']}, предсказано: {stats['pred_total']}")
        print(f"  Совпадений (IoU>={IOU_THRESHOLD}): {stats['matched']}")
        print(f"  Классификация верна: {stats['cls_correct']}/{stats['det_tp']}")

        for i, (box, name, conf) in enumerate(zip(
            predictions["boxes"], predictions["class_names"], predictions["confidences"],
        )):
            print(f"    [{i + 1}] {name} ({conf:.2f}) bbox={box}")

        output_path = DEMO_OUTPUT_DIR / f"{image_path.stem}_demo.jpg"
        _save_demo_image(image_path, gt_objects, predictions, output_path)
        print(f"  Сохранено: {output_path}")


def main() -> None:
    print("Загрузка pipeline...")
    pipeline = DiatomPipeline(device=DEVICE)
    pipeline.load(
        detector_path=DETECTOR_PATH,
        classifier_path=CLASSIFIER_PATH,
    )

    try:
        det_metrics = evaluate_detection(pipeline)
        print_metrics("Метрики детектора (YOLOv11) на TEST", det_metrics)
    except Exception as e:
        print(f"Ошибка оценки детектора: {e}")

    try:
        cls_metrics = evaluate_classification(pipeline)
        print_metrics("Метрики классификатора (DiatomNet) на TEST", cls_metrics)
    except Exception as e:
        print(f"Ошибка оценки классификатора: {e}")

    try:
        e2e_metrics = evaluate_pipeline_end_to_end(pipeline)
        print_metrics("End-to-end pipeline на TEST", e2e_metrics)
    except Exception as e:
        print(f"Ошибка end-to-end оценки: {e}")

    run_demo(pipeline)


if __name__ == "__main__":
    main()
