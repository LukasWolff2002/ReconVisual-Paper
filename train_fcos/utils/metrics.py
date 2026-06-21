"""
utils/metrics.py
Cálculo de mAP@IoU=0.5 por clase y global.
"""

import numpy as np
import torch
from collections import defaultdict
from .box_utils import box_iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP usando interpolación de 11 puntos (VOC-style)."""
    ap = 0.0
    for thr in np.arange(0.0, 1.1, 0.1):
        p = precision[recall >= thr]
        ap += p.max() if p.size > 0 else 0.0
    return ap / 11.0


def compute_map(
    predictions: list,   # lista de dicts por imagen: {boxes, scores, labels}
    targets: list,       # lista de dicts por imagen: {boxes, labels}
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Calcula mAP@iou_threshold por clase y global.

    Returns: dict con keys:
        'mAP'     → float, mAP global
        'AP_{c}'  → float, AP de clase c
        'per_class' → dict {class_id: ap}
    """
    # Acumular TP/FP por clase
    class_tp  = defaultdict(list)
    class_fp  = defaultdict(list)
    class_scores = defaultdict(list)
    class_n_gt   = defaultdict(int)

    for pred, tgt in zip(predictions, targets):
        pred_boxes  = pred["boxes"]   # [N, 4]
        pred_scores = pred["scores"]  # [N]
        pred_labels = pred["labels"]  # [N]
        gt_boxes    = tgt["boxes"]    # [M, 4]
        gt_labels   = tgt["labels"]  # [M]

        # Ordenar por score descendente
        order = pred_scores.argsort(descending=True)
        pred_boxes  = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]

        # Contar gt por clase
        for lbl in gt_labels:
            class_n_gt[lbl.item()] += 1

        # Matched gt para evitar doble conteo
        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for i in range(len(pred_boxes)):
            cls = pred_labels[i].item()
            class_scores[cls].append(pred_scores[i].item())

            # Filtrar gt de misma clase
            gt_cls_mask = (gt_labels == cls)
            if gt_cls_mask.sum() == 0 or len(gt_boxes) == 0:
                class_fp[cls].append(1)
                class_tp[cls].append(0)
                continue

            gt_cls_idx = gt_cls_mask.nonzero(as_tuple=False).squeeze(1)
            iou = box_iou(pred_boxes[i:i+1], gt_boxes[gt_cls_idx])  # [1, K]
            iou = iou.squeeze(0)  # [K]

            best_iou, best_j = iou.max(0)
            best_gt_idx = gt_cls_idx[best_j]

            if best_iou >= iou_threshold and not matched[best_gt_idx]:
                class_tp[cls].append(1)
                class_fp[cls].append(0)
                matched[best_gt_idx] = True
            else:
                class_tp[cls].append(0)
                class_fp[cls].append(1)

    # Calcular AP por clase
    per_class_ap = {}
    for cls in range(num_classes):
        n_gt = class_n_gt.get(cls, 0)
        if n_gt == 0:
            continue

        scores = np.array(class_scores.get(cls, []))
        tp     = np.array(class_tp.get(cls, []))
        fp     = np.array(class_fp.get(cls, []))

        if len(scores) == 0:
            per_class_ap[cls] = 0.0
            continue

        # Ordenar por score
        order  = np.argsort(-scores)
        tp     = tp[order]
        fp     = fp[order]

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recall    = cum_tp / n_gt
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)

        per_class_ap[cls] = compute_ap(recall, precision)

    mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0

    result = {"mAP": float(mAP), "per_class": per_class_ap}
    for cls, ap in per_class_ap.items():
        result[f"AP_{cls}"] = float(ap)

    return result


class DetectionEvaluator:
    """
    Acumula predicciones y ground-truths a lo largo de un epoch
    y calcula mAP al final.
    """
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes   = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets     = []

    def update(self, predictions: list, targets: list):
        """
        predictions: lista de dicts {boxes [N,4], scores [N], labels [N]}
        targets:     lista de dicts {boxes [M,4], labels [M]}
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute(self) -> dict:
        return compute_map(
            self.predictions,
            self.targets,
            self.num_classes,
            self.iou_threshold,
        )
