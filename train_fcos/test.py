"""
test.py — Evaluación completa del modelo FCOS-iDoc

Genera:
  - mAP global y por clase (con ranking)
  - mAP por tamaño de bbox (small, medium, large)
  - Curvas Precision-Recall por clase
  - Análisis de errores: FP y FN por clase y por tamaño
  - Análisis GIoU: distribución, mean por clase, heatmap clase×tamaño, score vs GIoU
  - Visualización de detecciones sobre imágenes del test set (GT vs Predicciones)
  - Reporte en consola y guardado en JSON

Uso:
    python3 -m train_fcos.test \
    --checkpoint /home/rvdl_2/train_fcos/run_01/best_model.pth \
    --image_root /home/rvdl_2/ \
    --dataset_json /home/rvdl_2/detection_dataset_sketches.json \
    --output_dir /home/rvdl_2/train_fcos/test_results/ \
    --score_thr 0.05 \
    --iou_thr 0.5 \
    --vis_n 10
"""

import os, sys, json, argparse, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageOps, ImageFont
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.abspath(os.path.dirname(__file__))
ROOT  = os.path.abspath(os.path.join(_HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_fcos          import config as C
from train_fcos.models.detector   import FCOSDetector
from train_fcos.datasets import build_datasets, collate_fn


# ══════════════════════════════════════════════════════════════════════════════
# 1. ARGUMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Test FCOS iDoc")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--image_root",   type=str, required=True)
    p.add_argument("--dataset_json", type=str, default=C.DATASET_JSON)
    p.add_argument("--output_dir",   type=str, default="test_results")
    p.add_argument("--score_thr",    type=float, default=0.05)
    p.add_argument("--iou_thr",      type=float, default=0.5)
    p.add_argument("--nms_iou",      type=float, default=0.5)
    p.add_argument("--vis_n",        type=int,   default=10)
    p.add_argument("--split",        type=str,   default="test",
                   choices=["test", "val"])
    p.add_argument("--device",       type=str,   default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: torch.device) -> FCOSDetector:
    cfg = {
        "BACKBONE":      C.BACKBONE,
        "FPN":           C.FPN,
        "FCOS_HEAD":     C.FCOS_HEAD,
        "QUERY_ENCODER": C.QUERY_ENCODER,
        "EVAL":          C.EVAL,
        "PRETRAINED_PTH": None,
    }
    model = FCOSDetector(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    epoch    = ckpt.get("epoch", "?")
    best_map = ckpt.get("best_map", None)
    print(f"  Checkpoint: epoch={epoch}"
          + (f"  best_mAP={best_map:.4f}" if best_map else ""))
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÉTRICAS  (IoU + GIoU)
# ══════════════════════════════════════════════════════════════════════════════

def box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU entre [N,4] y [M,4]. Retorna [N,M]."""
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)
    ix1 = torch.max(boxes1[:,None,0], boxes2[None,:,0])
    iy1 = torch.max(boxes1[:,None,1], boxes2[None,:,1])
    ix2 = torch.min(boxes1[:,None,2], boxes2[None,:,2])
    iy2 = torch.min(boxes1[:,None,3], boxes2[None,:,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    union = area1[:,None] + area2[None,:] - inter
    return inter / union.clamp(1e-6)


# ── NUEVO ─────────────────────────────────────────────────────────────────────
def box_giou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    GIoU [N,4] × [M,4] → [N,M].   Rango: [-1, 1]
      1  → cajas idénticas
      0  → sin solapamiento, tangentes
     -1  → máxima separación posible
    """
    a1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    a2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)
    ix1 = torch.max(boxes1[:,None,0], boxes2[None,:,0])
    iy1 = torch.max(boxes1[:,None,1], boxes2[None,:,1])
    ix2 = torch.min(boxes1[:,None,2], boxes2[None,:,2])
    iy2 = torch.min(boxes1[:,None,3], boxes2[None,:,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    union = a1[:,None] + a2[None,:] - inter
    iou   = inter / union.clamp(1e-6)
    ex1 = torch.min(boxes1[:,None,0], boxes2[None,:,0])
    ey1 = torch.min(boxes1[:,None,1], boxes2[None,:,1])
    ex2 = torch.max(boxes1[:,None,2], boxes2[None,:,2])
    ey2 = torch.max(boxes1[:,None,3], boxes2[None,:,3])
    enc  = (ex2-ex1).clamp(0) * (ey2-ey1).clamp(0)
    return iou - (enc - union) / enc.clamp(1e-6)
# ─────────────────────────────────────────────────────────────────────────────


def classify_box_size(box: torch.Tensor, img_area: float) -> str:
    w    = (box[2] - box[0]).clamp(0)
    h    = (box[3] - box[1]).clamp(0)
    area = (w * h).item()
    if area < 1024:  return 'small'
    if area < 9216:  return 'medium'
    return 'large'


class MetricsAccumulator:
    """
    Acumula predicciones y GT para calcular métricas completas al final.
    Incluye métricas por tamaño de bbox y análisis GIoU.
    """
    def __init__(self, num_classes: int, iou_thr: float = 0.5):
        self.num_classes = num_classes
        self.iou_thr     = iou_thr

        self.cls_preds   = defaultdict(list)
        self.cls_n_gt    = defaultdict(int)
        self.fn_per_cls  = defaultdict(int)
        self.fp_per_cls  = defaultdict(int)

        self.size_metrics = {
            s: defaultdict(lambda: {'preds': [], 'n_gt': 0, 'fn': 0, 'fp': 0})
            for s in ['small', 'medium', 'large']
        }

        # ── NUEVO: almacenamiento GIoU ────────────────────────────────────────
        # giou_tp[cls]  → GIoU de cada TP (predicción correcta)
        # giou_fp[cls]  → GIoU del mejor match para cada FP (aunque no pase iou_thr)
        self.giou_tp      = defaultdict(list)
        self.giou_fp      = defaultdict(list)
        self.giou_tp_size = {s: defaultdict(list) for s in ['small','medium','large']}
        # ─────────────────────────────────────────────────────────────────────

    def update(self, predictions: list, targets: list, img_shapes: list):
        for pred, tgt, shape in zip(predictions, targets, img_shapes):
            pred_boxes  = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]
            gt_boxes    = tgt["boxes"]
            gt_labels   = tgt["labels"]

            img_h, img_w = shape
            img_area = img_h * img_w

            gt_sizes = []
            for box, lbl in zip(gt_boxes, gt_labels):
                size_cat = classify_box_size(box, img_area)
                gt_sizes.append(size_cat)
                self.cls_n_gt[lbl.item()] += 1
                self.size_metrics[size_cat][lbl.item()]['n_gt'] += 1

            if len(pred_scores) > 0:
                order       = pred_scores.argsort(descending=True)
                pred_boxes  = pred_boxes[order]
                pred_scores = pred_scores[order]
                pred_labels = pred_labels[order]

            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

            # ── NUEVO: calcular IoU y GIoU de una sola vez ────────────────────
            all_iou  = box_iou_pairwise(pred_boxes, gt_boxes)  \
                       if len(gt_boxes) > 0 else None
            all_giou = box_giou_pairwise(pred_boxes, gt_boxes) \
                       if len(gt_boxes) > 0 else None
            # ─────────────────────────────────────────────────────────────────

            for i in range(len(pred_boxes)):
                cls       = pred_labels[i].item()
                score     = pred_scores[i].item()
                pred_size = classify_box_size(pred_boxes[i], img_area)

                gt_cls_mask = (gt_labels == cls)
                if gt_cls_mask.sum() == 0 or len(gt_boxes) == 0:
                    self.cls_preds[cls].append((score, False))
                    self.fp_per_cls[cls] += 1
                    self.size_metrics[pred_size][cls]['preds'].append((score, False))
                    self.size_metrics[pred_size][cls]['fp'] += 1
                    self.giou_fp[cls].append(-1.0)   # sin GT → peor caso
                    continue

                gt_cls_idx = gt_cls_mask.nonzero(as_tuple=False).squeeze(1)

                iou_row  = all_iou[i,  gt_cls_idx]
                giou_row = all_giou[i, gt_cls_idx]   # ← NUEVO

                best_iou, best_j = iou_row.max(0)
                best_gt  = gt_cls_idx[best_j.item()].item()
                giou_val = giou_row[best_j].item()    # ← NUEVO: GIoU del mismo GT

                if best_iou >= self.iou_thr and not matched_gt[best_gt]:
                    matched_gt[best_gt] = True
                    self.cls_preds[cls].append((score, True))
                    self.giou_tp[cls].append(giou_val)                    # ← NUEVO
                    gt_size = gt_sizes[best_gt]
                    self.size_metrics[gt_size][cls]['preds'].append((score, True))
                    self.giou_tp_size[gt_size][cls].append(giou_val)      # ← NUEVO
                else:
                    self.cls_preds[cls].append((score, False))
                    self.fp_per_cls[cls] += 1
                    self.giou_fp[cls].append(giou_val)                    # ← NUEVO
                    self.size_metrics[pred_size][cls]['preds'].append((score, False))
                    self.size_metrics[pred_size][cls]['fp'] += 1

            for idx, lbl in enumerate(gt_labels.tolist()):
                if not matched_gt[idx]:
                    self.fn_per_cls[lbl] += 1
                    self.size_metrics[gt_sizes[idx]][lbl]['fn'] += 1

    def compute_ap(self, scores_tp: list, n_gt: int):
        if n_gt == 0 or len(scores_tp) == 0:
            return 0.0, np.array([0.]), np.array([0.])
        scores = np.array([s for s, _ in scores_tp])
        tps    = np.array([int(tp) for _, tp in scores_tp])
        order  = np.argsort(-scores)
        tps    = tps[order]
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(1 - tps)
        rec    = cum_tp / n_gt
        prec   = cum_tp / (cum_tp + cum_fp + 1e-9)
        ap = sum(prec[rec >= t].max() if (rec >= t).any() else 0.
                 for t in np.arange(0., 1.1, 0.1)) / 11.
        return ap, rec, prec

    def compute(self):
        per_class = {}
        pr_curves = {}

        for cls in range(self.num_classes):
            n_gt  = self.cls_n_gt.get(cls, 0)
            preds = self.cls_preds.get(cls, [])
            ap, rec, prec = self.compute_ap(preds, n_gt)
            n_tp = sum(1 for _, tp in preds if tp)
            n_fp = self.fp_per_cls.get(cls, 0)
            n_fn = self.fn_per_cls.get(cls, 0)
            per_class[cls] = {
                "ap":        round(float(ap), 4),
                "n_gt":      n_gt,
                "n_pred":    len(preds),
                "n_tp":      n_tp,
                "n_fp":      n_fp,
                "n_fn":      n_fn,
                "recall":    round(float(rec[-1]), 4) if len(rec) else 0.,
                "precision": round(float(prec[-1]), 4) if len(prec) else 0.,
            }
            pr_curves[cls] = {"recall": rec.tolist(), "precision": prec.tolist()}

        aps = [v["ap"] for v in per_class.values() if v["n_gt"] > 0]
        mAP = float(np.mean(aps)) if aps else 0.0

        per_size = {}
        for size_name in ['small', 'medium', 'large']:
            size_data = self.size_metrics[size_name]
            per_size[size_name] = {}
            for cls in range(self.num_classes):
                d = size_data[cls]
                ap_s, _, _ = self.compute_ap(d['preds'], d['n_gt'])
                n_tp = sum(1 for _, tp in d['preds'] if tp)
                per_size[size_name][cls] = {
                    "ap": round(float(ap_s), 4), "n_gt": d['n_gt'],
                    "n_pred": len(d['preds']), "n_tp": n_tp,
                    "n_fp": d['fp'], "n_fn": d['fn'],
                }
            size_aps = [v["ap"] for v in per_size[size_name].values() if v["n_gt"] > 0]
            per_size[size_name]["mAP"] = round(float(np.mean(size_aps)), 4) \
                                         if size_aps else 0.0

        # ── NUEVO: estadísticas GIoU ──────────────────────────────────────────
        def _stats(vals):
            if not vals:
                return {"mean": None, "median": None,
                        "std":  None, "min":    None, "max": None}
            a = np.array(vals)
            return {"mean":   round(float(a.mean()),    4),
                    "median": round(float(np.median(a)),4),
                    "std":    round(float(a.std()),     4),
                    "min":    round(float(a.min()),     4),
                    "max":    round(float(a.max()),     4)}

        giou_stats = {
            cls: {
                "tp":        _stats(self.giou_tp.get(cls, [])),
                "fp":        _stats(self.giou_fp.get(cls, [])),
                "tp_small":  _stats(self.giou_tp_size["small"].get(cls,  [])),
                "tp_medium": _stats(self.giou_tp_size["medium"].get(cls, [])),
                "tp_large":  _stats(self.giou_tp_size["large"].get(cls,  [])),
                "tp_values": self.giou_tp.get(cls, []),
                "fp_values": self.giou_fp.get(cls, []),
            }
            for cls in range(self.num_classes)
        }

        all_tp_giou = [g for c in range(self.num_classes)
                       for g in self.giou_tp.get(c, [])]
        all_fp_giou = [g for c in range(self.num_classes)
                       for g in self.giou_fp.get(c, [])]
        # ─────────────────────────────────────────────────────────────────────

        return {
            "mAP":      round(mAP, 4),
            "per_class": per_class,
            "pr_curves": pr_curves,
            "per_size":  per_size,
            # ── NUEVO ─────────────────────────────────────────────────────────
            "giou_stats": giou_stats,
            "giou_global": {
                "tp_mean": round(float(np.mean(all_tp_giou)), 4) if all_tp_giou else None,
                "fp_mean": round(float(np.mean(all_fp_giou)), 4) if all_fp_giou else None,
                "n_tp":    len(all_tp_giou),
                "n_fp":    len(all_fp_giou),
                "tp_values": all_tp_giou,
                "fp_values": all_fp_giou,
            },
            # ──────────────────────────────────────────────────────────────────
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#8338EC", "#06D6A0", "#FB5607", "#3A86FF", "#FFBE0B",
    "#8D99AE", "#EF233C", "#4CC9F0", "#7209B7", "#560BAD",
    "#480CA8", "#3A0CA3", "#3F37C9", "#4361EE", "#4895EF",
    "#B5179E", "#F72585",
]


def draw_boxes_single(img, boxes, scores, labels, class_names,
                      score_thr=0.0, title="", giou_vals=None):
    img  = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font_t = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_l = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_t = font_l = ImageFont.load_default()

    if title:
        tb = draw.textbbox((0, 0), title, font=font_t)
        th = tb[3] - tb[1] + 10
        draw.rectangle([0, 0, img.width, th], fill=(0, 0, 0))
        draw.text((10, 5), title, fill="white", font=font_t)

    for i, (box, lbl) in enumerate(zip(boxes.tolist(), labels.tolist())):
        if scores is not None and scores[i].item() < score_thr:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS[lbl % len(COLORS)]
        name  = class_names[lbl] if lbl < len(class_names) else str(lbl)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if scores is not None:
            # ── NUEVO: mostrar GIoU en el label si está disponible ────────────
            if giou_vals is not None and i < len(giou_vals):
                txt = f"{name}: {scores[i].item():.2f} G={giou_vals[i]:.2f}"
            else:
                txt = f"{name}: {scores[i].item():.2f}"
            # ──────────────────────────────────────────────────────────────────
        else:
            txt = name

        lb = draw.textbbox((x1, y1), txt, font=font_l)
        lw = lb[2] - lb[0] + 4
        lh = lb[3] - lb[1] + 4
        draw.rectangle([x1, max(0, y1-lh), x1+lw, y1], fill=color)
        draw.text((x1+2, max(0, y1-lh+2)), txt, fill="white", font=font_l)

    return img


def create_comparison_visualization(img, pred_boxes, pred_scores, pred_labels,
                                    gt_boxes, gt_labels, class_names,
                                    score_thr, giou_vals=None):
    gt_img   = draw_boxes_single(img, gt_boxes, None, gt_labels,
                                  class_names, title="Ground Truth")
    pred_img = draw_boxes_single(img, pred_boxes, pred_scores, pred_labels,
                                  class_names, score_thr, "Predictions",
                                  giou_vals=giou_vals)  # ← NUEVO
    out = Image.new("RGB", (gt_img.width + pred_img.width,
                            max(gt_img.height, pred_img.height)), (255, 255, 255))
    out.paste(gt_img,   (0, 0))
    out.paste(pred_img, (gt_img.width, 0))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTS ESTÁNDAR  (sin cambios respecto a tu código original)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves(pr_curves, per_class, class_names, output_dir):
    valid_cls = [c for c, v in per_class.items() if v["n_gt"] > 0]
    n = len(valid_cls)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()
    for ax_idx, cls in enumerate(sorted(valid_cls)):
        rec  = np.array(pr_curves[cls]["recall"])
        prec = np.array(pr_curves[cls]["precision"])
        ap   = per_class[cls]["ap"]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        color = COLORS[cls % len(COLORS)]
        axes[ax_idx].plot(rec, prec, color=color, linewidth=2)
        axes[ax_idx].fill_between(rec, prec, alpha=.15, color=color)
        axes[ax_idx].set_xlim(0, 1); axes[ax_idx].set_ylim(0, 1.05)
        axes[ax_idx].set_title(f"{name}\nAP={ap:.3f}", fontsize=9)
        axes[ax_idx].set_xlabel("Recall", fontsize=7)
        axes[ax_idx].set_ylabel("Precision", fontsize=7)
        axes[ax_idx].tick_params(labelsize=6)
        axes[ax_idx].grid(True, alpha=.3)
    for i in range(len(valid_cls), len(axes)):
        axes[i].set_visible(False)
    plt.suptitle("Curvas Precision-Recall por clase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → pr_curves.png guardado")


def plot_ap_ranking(per_class, class_names, mAP, output_dir):
    valid = {c: v for c, v in per_class.items() if v["n_gt"] > 0}
    sorted_cls = sorted(valid.keys(), key=lambda c: valid[c]["ap"], reverse=True)
    names  = [class_names[c] if c < len(class_names) else str(c) for c in sorted_cls]
    aps    = [valid[c]["ap"] for c in sorted_cls]
    colors = [COLORS[c % len(COLORS)] for c in sorted_cls]
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * .45)))
    bars = ax.barh(range(len(names)), aps, color=colors, edgecolor="white", height=.7)
    for bar, ap_val in zip(bars, aps):
        ax.text(bar.get_width() + .005, bar.get_y() + bar.get_height()/2,
                f"{ap_val:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AP@0.5", fontsize=10); ax.set_xlim(0, 1.12)
    ax.axvline(mAP, color="red", linestyle="--", linewidth=1.5,
               label=f"mAP={mAP:.3f}")
    ax.legend(fontsize=9)
    ax.set_title("Average Precision por clase", fontsize=12, fontweight="bold")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ap_ranking.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → ap_ranking.png guardado")


def plot_error_analysis(per_class, class_names, output_dir):
    valid = {c: v for c, v in per_class.items() if v["n_gt"] > 0}
    sorted_cls = sorted(valid.keys(), key=lambda c: valid[c]["n_gt"], reverse=True)
    names = [class_names[c] if c < len(class_names) else str(c) for c in sorted_cls]
    x = np.arange(len(names)); w = .25
    fig, ax = plt.subplots(figsize=(max(10, len(names) * .7), 5))
    ax.bar(x-w, [valid[c]["n_tp"] for c in sorted_cls], w,
           label="TP (correcto)", color="#2A9D8F")
    ax.bar(x,   [valid[c]["n_fp"] for c in sorted_cls], w,
           label="FP (falso positivo)", color="#E63946")
    ax.bar(x+w, [valid[c]["n_fn"] for c in sorted_cls], w,
           label="FN (falso negativo)", color="#E9C46A")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cantidad")
    ax.set_title("Análisis de errores por clase (test set)", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  → error_analysis.png guardado")


def plot_size_analysis(per_size, class_names, output_dir):
    sizes     = ['small', 'medium', 'large']
    size_maps = [per_size[s]['mAP'] for s in sizes]
    colors    = {'small': '#3A86FF', 'medium': '#06D6A0', 'large': '#FB5607'}
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(3), size_maps,
                  color=[colors[s] for s in sizes], edgecolor='white', width=.6)
    for bar, val in zip(bars, size_maps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Small\n(< 32²px)', 'Medium\n(32² - 96²px)',
                        'Large\n(≥ 96²px)'], fontsize=10)
    ax.set_ylabel('mAP@0.5', fontsize=11)
    ax.set_ylim(0, max(size_maps) * 1.15 if size_maps else 1)
    ax.set_title('mAP por tamaño de Bounding Box', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "size_analysis.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  → size_analysis.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS GIOU  (completamente nuevos)
# ══════════════════════════════════════════════════════════════════════════════
def plot_giou_distribution(giou_global: dict, out_dir: str):
    """
    Histograma comparativo de GIoU para TP vs FP.
    """
    tp_vals = np.array(giou_global["tp_values"])
    fp_vals = np.array(giou_global["fp_values"])
    bins    = np.linspace(-1, 1, 41)

    # Creamos un solo eje (ax) en lugar de una lista de ejes (axes)
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    if len(tp_vals):
        ax.hist(tp_vals, bins=bins, alpha=.75, color="#2A9D8F", density=True,
                     label=f"TP  n={len(tp_vals)}  μ={tp_vals.mean():.3f}")
    if len(fp_vals):
        ax.hist(fp_vals, bins=bins, alpha=.75, color="#E63946", density=True,
                     label=f"FP  n={len(fp_vals)}  μ={fp_vals.mean():.3f}")
    
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="IoU thr=0.5")
    ax.set(xlabel="GIoU", ylabel="Densidad")
               # title="Distribución GIoU: TP vs FP", xlim=(-1, 1))
    ax.legend(fontsize=12)
    ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "giou_distribution.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  → giou_distribution.png guardado")

def plot_giou_per_class(giou_stats: dict, per_class: dict,
                        class_names: list, out_dir: str):
    """
    Bar chart horizontal: mean GIoU de TP por clase con barras de error (std).
    Color codificado por cantidad de TP.

    Qué buscar:
      - GIoU alto (>0.7)  → el modelo ajusta bien las cajas, no solo las detecta
      - GIoU bajo (<0.5)  → muchos TP "barely passing" el umbral IoU=0.5
      - Std alta          → ajuste inconsistente para esa clase
    """
    valid = [c for c in range(len(class_names))
             if per_class[c]["n_gt"] > 0 and per_class[c]["n_tp"] > 0]
    if not valid:
        print("  (sin TPs para graficar GIoU por clase)")
        return

    valid  = sorted(valid,
                    key=lambda c: giou_stats[c]["tp"]["mean"] or -2,
                    reverse=True)
    names  = [class_names[c] if c < len(class_names) else str(c) for c in valid]
    means  = [giou_stats[c]["tp"]["mean"] or 0 for c in valid]
    stds   = [giou_stats[c]["tp"]["std"]  or 0 for c in valid]
    n_tps  = [per_class[c]["n_tp"]            for c in valid]

    norm   = plt.Normalize(min(n_tps), max(n_tps) + 1)
    colors = [plt.cm.YlOrRd(norm(n)) for n in n_tps]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * .5)))
    bars = ax.barh(range(len(names)), means, xerr=stds,
                   color=colors, edgecolor="white", height=.7,
                   capsize=3, error_kw={"elinewidth": 1})
    for bar, n in zip(bars, n_tps):
        ax.text(bar.get_width() + .01, bar.get_y() + bar.get_height()/2,
                f"n={n}", va="center", fontsize=7, color="#444")

    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set(xlabel="Mean GIoU (solo TP)", xlim=(-0.05, 1.15),
           title="GIoU Medio por Clase (TP) ± std\ncolor = cantidad de TP")
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="IoU thr=0.5")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.3); ax.legend(fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="n_tp", fraction=0.025)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "giou_per_class.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  → giou_per_class.png guardado")


def plot_giou_by_size_heatmap(giou_stats: dict, per_class: dict,
                               class_names: list, out_dir: str):
    """
    Heatmap: clases (filas) × tamaño de bbox (columnas) → mean GIoU de TP.
    Celdas grises = sin TPs de esa combinación.

    Qué buscar:
      - Rojo en 'small'  → detección correcta pero cajas mal ajustadas en pequeños
      - Verde en 'large' → buen ajuste en objetos grandes (más fácil)
      - Diagonal de mejora → el modelo mejora con el tamaño del objeto
    """
    valid     = [c for c in range(len(class_names)) if per_class[c]["n_gt"] > 0]
    size_keys = ["tp_small", "tp_medium", "tp_large"]
    size_lbl  = ["Small\n<32²px", "Medium\n32²–96²px", "Large\n≥96²px"]

    matrix = np.full((len(valid), 3), np.nan)
    for row, cls in enumerate(valid):
        for col, sk in enumerate(size_keys):
            m = giou_stats[cls][sk]["mean"]
            if m is not None:
                matrix[row, col] = m

    if np.all(np.isnan(matrix)):
        print("  (sin datos GIoU por tamaño para el heatmap)")
        return

    names = [class_names[c] if c < len(class_names) else str(c) for c in valid]
    fig, ax = plt.subplots(figsize=(5, max(4, len(names) * .42)))
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#cccccc")
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                   vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(size_lbl, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Mean GIoU (TP) por Clase y Tamaño\ngris = sin TPs",
                 fontsize=11, fontweight="bold")
    for row in range(len(valid)):
        for col in range(3):
            val = matrix[row, col]
            if not np.isnan(val):
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.55 else "black")
    plt.colorbar(im, ax=ax, label="Mean GIoU", fraction=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "giou_by_size_heatmap.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  → giou_by_size_heatmap.png guardado")


def plot_giou_vs_score(acc, out_dir: str):
    """
    Scatter: score de confianza (X) vs GIoU (Y) para TP y FP.

    Qué buscar:
      - TP en zona superior-derecha (score alto, GIoU alto) → bien calibrado
      - FP con score alto y GIoU bajo → problema grave de calibración
      - Correlación de Pearson r>0.3 para TP indica que el score predice
        la calidad geométrica de la caja
    """
    tp_scores, tp_gious = [], []
    fp_scores, fp_gious = [], []

    for cls in range(acc.num_classes):
        preds     = acc.cls_preds.get(cls, [])
        giou_tp_v = acc.giou_tp.get(cls, [])
        giou_fp_v = acc.giou_fp.get(cls, [])
        tp_idx = fp_idx = 0
        for score, is_tp in preds:
            if is_tp and tp_idx < len(giou_tp_v):
                tp_scores.append(score); tp_gious.append(giou_tp_v[tp_idx])
                tp_idx += 1
            elif not is_tp and fp_idx < len(giou_fp_v):
                fp_scores.append(score); fp_gious.append(giou_fp_v[fp_idx])
                fp_idx += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    if fp_scores:
        ax.scatter(fp_scores, fp_gious, s=8, alpha=.3, color="#E63946",
                   label=f"FP (n={len(fp_scores)})")
    if tp_scores:
        ax.scatter(tp_scores, tp_gious, s=8, alpha=.5, color="#2A9D8F",
                   label=f"TP (n={len(tp_scores)})")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="IoU thr=0.5")
    ax.set(xlabel="Score de confianza", ylabel="GIoU",
           title="Score vs GIoU  (TP y FP)\nTP bien calibrado → zona superior derecha",
           xlim=(0, 1), ylim=(-1.05, 1.05))
    ax.legend(fontsize=9, markerscale=3); ax.grid(alpha=.3)

    if len(tp_scores) > 2:
        r = float(np.corrcoef(tp_scores, tp_gious)[0, 1])
        ax.text(0.02, 0.96, f"Correlación TP: r = {r:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=.85))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "giou_vs_score.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  → giou_vs_score.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# 7. REPORTE EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict, class_names: list):
    per      = metrics["per_class"]
    mAP      = metrics["mAP"]
    per_size = metrics["per_size"]
    giou_g   = metrics["giou_global"]   # ← NUEVO
    giou_s   = metrics["giou_stats"]    # ← NUEVO

    sorted_cls = sorted([c for c, v in per.items() if v["n_gt"] > 0],
                        key=lambda c: per[c]["ap"], reverse=True)

    print("\n" + "═"*82)
    print(f"  RESULTADOS TEST SET   mAP@0.5 = {mAP:.4f}")
    print("═"*82)
    # ── NUEVO: columna GIoU_TP añadida al header ──────────────────────────────
    print(f"  {'Clase':<16} {'AP':>6}  {'GT':>4}  {'TP':>4}  {'FP':>4}  "
          f"{'FN':>4}  {'Rec':>6}  {'Prec':>6}  {'GIoU_TP':>8}")
    print("  " + "-"*80)

    for cls in sorted_cls:
        v    = per[cls]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        # ── NUEVO: extraer mean GIoU TP ───────────────────────────────────────
        g_tp = (f"{giou_s[cls]['tp']['mean']:.3f}"
                if giou_s[cls]["tp"]["mean"] is not None else "   —   ")
        # ─────────────────────────────────────────────────────────────────────
        bar  = "█" * int(v["ap"] * 15)
        print(f"  {name:<16} {v['ap']:>6.3f}  {v['n_gt']:>4}  {v['n_tp']:>4}  "
              f"{v['n_fp']:>4}  {v['n_fn']:>4}  {v['recall']:>6.3f}  "
              f"{v['precision']:>6.3f}  {g_tp:>8}  {bar}")

    print("  " + "-"*80)

    # ── NUEVO: resumen GIoU global ────────────────────────────────────────────
    print(f"\n  GIoU global:")
    tp_m = f"{giou_g['tp_mean']:.4f}" if giou_g["tp_mean"] is not None else "—"
    fp_m = f"{giou_g['fp_mean']:.4f}" if giou_g["fp_mean"] is not None else "—"
    print(f"    TP (n={giou_g['n_tp']:4d}) → mean GIoU = {tp_m}")
    print(f"    FP (n={giou_g['n_fp']:4d}) → mean GIoU = {fp_m}")
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  mAP por tamaño:")
    for s in ['small', 'medium', 'large']:
        print(f"    {s.upper():<7} = {per_size[s]['mAP']:.4f}")

    no_gt = [c for c, v in per.items() if v["n_gt"] == 0]
    if no_gt:
        names_no_gt = [class_names[c] if c < len(class_names) else str(c)
                       for c in no_gt]
        print(f"\n  Clases sin GT: {', '.join(names_no_gt)}")

    if len(sorted_cls) >= 3:
        print(f"\n  Mejores → {', '.join(class_names[c] for c in sorted_cls[:3])}")
        print(f"  Peores  → {', '.join(class_names[c] for c in sorted_cls[-3:])}")

    print("═"*82 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"; vis_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  iDoc-FCOS — Test Script")
    print(f"  Device:    {device}")
    print(f"  Ckpt:      {args.checkpoint}")
    print(f"  Split:     {args.split}")
    print(f"  Score thr: {args.score_thr}  |  IoU thr: {args.iou_thr}")
    print(f"{'='*60}\n")

    with open(args.dataset_json) as f:
        data = json.load(f)
    sorted_ids  = sorted(c["class_id"] for c in data["classes"])
    id_to_name  = {c["class_id"]: c["class_name"] for c in data["classes"]}
    class_names = [id_to_name[cid] for cid in sorted_ids]
    num_classes = len(class_names)
    print(f"  Clases ({num_classes}): {', '.join(class_names)}\n")

    print("Cargando modelo...")
    model = load_model(args.checkpoint, device)

    print("Cargando dataset...")
    cfg = {
        "BACKBONE": C.BACKBONE, "FPN": C.FPN, "FCOS_HEAD": C.FCOS_HEAD,
        "QUERY_ENCODER": C.QUERY_ENCODER, "DATASET": C.DATASET,
        "AUGMENTATION": C.AUGMENTATION, "EVAL": C.EVAL,
    }
    train_ds, val_ds, test_ds = build_datasets(args.dataset_json, args.image_root, cfg)
    ds = test_ds if args.split == "test" else val_ds
    print(f"  {args.split} set: {len(ds)} muestras\n")

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    C.EVAL["score_threshold"] = args.score_thr
    C.EVAL["nms_iou_thresh"]  = args.nms_iou

    print("Ejecutando inferencia...")
    acc       = MetricsAccumulator(num_classes=num_classes, iou_thr=args.iou_thr)
    vis_saved = 0
    t0        = time.time()
    mean_t    = torch.tensor(C.DATASET["pixel_mean"]).view(3, 1, 1)
    std_t     = torch.tensor(C.DATASET["pixel_std"]).view(3, 1, 1)

    for batch_idx, batch in enumerate(loader):
        pages   = batch["page_imgs"].to(device)
        queries = batch["query_imgs"].to(device)
        targets = batch["targets"]
        shapes  = batch["img_shapes"]

        dets = model.predict(pages, queries, shapes[0])
        cpu_dets = [{"boxes":  d["boxes"].cpu(),
                     "scores": d["scores"].cpu(),
                     "labels": d["labels"].cpu()} for d in dets]

        acc.update(cpu_dets, targets, [shapes[0]])

        if vis_saved < args.vis_n:
            img_t   = (pages[0].cpu() * std_t + mean_t).clamp(0, 1)
            img_pil = TF.to_pil_image(img_t)
            det     = cpu_dets[0]
            tgt     = targets[0]

            # ── NUEVO: calcular GIoU de cada pred con su mejor GT ─────────────
            giou_vals = None
            if len(det["boxes"]) > 0 and len(tgt["boxes"]) > 0:
                giou_mat  = box_giou_pairwise(det["boxes"], tgt["boxes"])
                giou_vals = giou_mat.max(dim=1).values.tolist()
            # ──────────────────────────────────────────────────────────────────

            img_vis = create_comparison_visualization(
                img_pil,
                det["boxes"], det["scores"], det["labels"],
                tgt["boxes"], tgt["labels"],
                class_names, args.score_thr,
                giou_vals=giou_vals,     # ← NUEVO
            )
            sid = batch["sample_ids"][0]
            img_vis.save(str(vis_dir / f"sample_{sid:04d}_comparison.jpg"))
            vis_saved += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]", end="\r")

    elapsed = time.time() - t0
    print(f"\n  Completado en {elapsed:.1f}s "
          f"({elapsed/max(len(ds),1)*1000:.1f}ms/img)")

    print("\nCalculando métricas...")
    metrics = acc.compute()
    print_report(metrics, class_names)

    # JSON
    g = metrics["giou_global"]
    metrics_out = {
        "mAP": metrics["mAP"],
        "iou_threshold":   args.iou_thr,
        "score_threshold": args.score_thr,
        "split": args.split,
        # ── NUEVO: GIoU global en JSON ─────────────────────────────────────────
        "giou_global": {k: v for k, v in g.items()
                        if k not in ("tp_values", "fp_values")},
        # ─────────────────────────────────────────────────────────────────────
        "per_class": {
            class_names[c] if c < len(class_names) else str(c): {
                **v,
                # ── NUEVO: GIoU stats por clase en JSON ───────────────────────
                "giou_tp": metrics["giou_stats"][c]["tp"],
                "giou_fp": metrics["giou_stats"][c]["fp"],
                # ─────────────────────────────────────────────────────────────
            }
            for c, v in metrics["per_class"].items()
        },
        "per_size": {
            sn: {
                "mAP": sd["mAP"],
                "classes": {
                    class_names[c] if c < len(class_names) else str(c): v
                    for c, v in sd.items()
                    if c != "mAP" and isinstance(v, dict)
                }
            }
            for sn, sd in metrics["per_size"].items()
        }
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print("  → metrics.json guardado")

    print("\nGenerando gráficos...")
    # Plots originales
    plot_ap_ranking(metrics["per_class"], class_names, metrics["mAP"], str(out_dir))
    plot_pr_curves(metrics["pr_curves"], metrics["per_class"], class_names, str(out_dir))
    plot_error_analysis(metrics["per_class"], class_names, str(out_dir))
    plot_size_analysis(metrics["per_size"], class_names, str(out_dir))
    # ── NUEVO: plots GIoU ─────────────────────────────────────────────────────
    plot_giou_distribution(metrics["giou_global"], str(out_dir))
    plot_giou_per_class(metrics["giou_stats"], metrics["per_class"],
                        class_names, str(out_dir))
    plot_giou_by_size_heatmap(metrics["giou_stats"], metrics["per_class"],
                               class_names, str(out_dir))
    plot_giou_vs_score(acc, str(out_dir))
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n✓ Resultados en: {out_dir}/")
    print(f"  metrics.json             ← métricas + GIoU por clase")
    print(f"  ap_ranking.png           ← AP por clase")
    print(f"  pr_curves.png            ← curvas PR")
    print(f"  error_analysis.png       ← TP / FP / FN")
    print(f"  size_analysis.png        ← mAP por tamaño")
    print(f"  giou_distribution.png    ← histograma GIoU TP vs FP")
    print(f"  giou_per_class.png       ← mean GIoU por clase")
    print(f"  giou_by_size_heatmap.png ← GIoU clase × tamaño")
    print(f"  giou_vs_score.png        ← score vs GIoU scatter")
    print(f"  visualizations/          ← {vis_saved} imgs (GIoU en label)")

    print(f"\n  mAP@{args.iou_thr}  = {metrics['mAP']:.4f}")
    tp_m = f"{g['tp_mean']:.4f}" if g["tp_mean"] is not None else "—"
    fp_m = f"{g['fp_mean']:.4f}" if g["fp_mean"] is not None else "—"
    print(f"  GIoU TP   = {tp_m}  (n={g['n_tp']})")
    print(f"  GIoU FP   = {fp_m}  (n={g['n_fp']})")
    for s in ['small', 'medium', 'large']:
        print(f"  mAP {s:<7}= {metrics['per_size'][s]['mAP']:.4f}")


if __name__ == "__main__":
    main()