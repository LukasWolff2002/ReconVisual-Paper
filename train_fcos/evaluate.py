"""
evaluate.py

Evaluación con guardado de resultados visuales, reporte de métricas y gráficos.

Modos de uso:

  # Evaluar val set → imágenes anotadas + report.txt + gráficos:
  python -m train_fcos.evaluate \
      --checkpoint train_fcos/run_03/best_model.pth \
      --image_root /home/rvdl_2/ \
      --output_dir eval_results/ \
      --split val

  # Evaluar test set:
  python -m train_fcos.evaluate \
      --checkpoint train_fcos/run_03/best_model.pth \
      --image_root /home/rvdl_2/ \
      --output_dir eval_results/ \
      --split test

  # Inferencia single:
  python -m train_fcos.evaluate \
      --checkpoint train_fcos/run_03/best_model.pth \
      --page_img page.jpg --query_img sketch.jpg \
      --output_img result.jpg
"""

import os
import sys
import argparse
import json
import datetime
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

_HERE = os.path.abspath(os.path.dirname(__file__))
ROOT  = os.path.abspath(os.path.join(_HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_fcos.config as C
from train_fcos.models.detector  import FCOSDetector
from train_fcos.datasets         import build_datasets, collate_fn
from train_fcos.utils.metrics    import DetectionEvaluator, compute_map
from train_fcos.utils.box_utils  import box_iou

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[evaluate] matplotlib no disponible — se omiten gráficos.")


# ─── Paleta de colores ────────────────────────────────────────────────────────
COL_TP_PRED  = "#4C9BE8"   # azul    — predicción verdadero positivo
COL_FP_PRED  = "#E85C4C"   # rojo    — predicción falso positivo
COL_TP_GT    = "#2ECC71"   # verde   — GT matcheado (TP)
COL_FN_GT    = "#F39C12"   # naranja — GT perdido (FN)


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Evaluación FCOS iDoc")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--image_root",   type=str, default=None)
    p.add_argument("--dataset_json", type=str, default=C.DATASET_JSON)
    p.add_argument("--output_dir",   type=str, default="eval_output",
                   help="Directorio donde se guardan imágenes, reporte y gráficos.")
    p.add_argument("--split",        type=str, default="val",
                   choices=["val", "test"],
                   help="Qué split evaluar (val o test).")
    p.add_argument("--score_thr",    type=float, default=0.25,
                   help="Score mínimo para mostrar detecciones en imágenes guardadas.")
    p.add_argument("--iou_thr",      type=float, default=0.5,
                   help="IoU threshold para TP/FP.")
    p.add_argument("--save_images",  action="store_true", default=True,
                   help="Guardar imágenes anotadas por muestra.")
    p.add_argument("--no_images",    action="store_true",
                   help="Saltar guardado de imágenes (solo métricas y plots).")
    # Single inference
    p.add_argument("--page_img",     type=str, default=None)
    p.add_argument("--query_img",    type=str, default=None)
    p.add_argument("--output_img",   type=str, default="result.jpg")
    return p.parse_args()


# ─── Fuente ───────────────────────────────────────────────────────────────────

def _load_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

FONT_MD = _load_font(14)
FONT_SM = _load_font(11)


# ─── Carga del modelo ─────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> FCOSDetector:
    cfg = {
        "BACKBONE":       C.BACKBONE,
        "FPN":            C.FPN,
        "FCOS_HEAD":      C.FCOS_HEAD,
        "QUERY_ENCODER":  C.QUERY_ENCODER,
        "EVAL":           C.EVAL,
        "PRETRAINED_PTH": None,
    }
    model = FCOSDetector(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"  Checkpoint : {checkpoint_path}")
    if "best_map" in ckpt:
        print(f"  Mejor mAP  : {ckpt['best_map']:.4f}  (epoch {ckpt.get('epoch','?')})")
    model.eval()
    return model


# ─── Pre-procesado ────────────────────────────────────────────────────────────

NORMALIZE = T.Normalize(mean=C.DATASET["pixel_mean"], std=C.DATASET["pixel_std"])


def preprocess_page(img_path: str, min_size: int, max_size: int):
    """Carga y escala la imagen; retorna (tensor [1,3,H,W], PIL resizeada, (H,W))."""
    img    = Image.open(img_path).convert("RGB")
    W0, H0 = img.size
    scale  = min_size / min(H0, W0)
    if scale * max(H0, W0) > max_size:
        scale = max_size / max(H0, W0)
    new_H, new_W = int(round(H0 * scale)), int(round(W0 * scale))
    img    = img.resize((new_W, new_H), Image.BILINEAR)
    tensor = NORMALIZE(TF.to_tensor(img)).unsqueeze(0)
    return tensor, img, (new_H, new_W)


def preprocess_query(img_path: str, size: int = 224):
    img    = Image.open(img_path).convert("RGB")
    img    = ImageOps.pad(img, (size, size))
    tensor = NORMALIZE(TF.to_tensor(img)).unsqueeze(0)
    return tensor, img


# ─── Match predicciones con GT ────────────────────────────────────────────────

def match_preds_to_gt(
    pred_boxes:  torch.Tensor,   # [N, 4] xyxy
    pred_scores: torch.Tensor,   # [N]
    gt_boxes:    torch.Tensor,   # [M, 4] xyxy
    iou_thr:     float = 0.5,
):
    """
    Asigna cada predicción a TP o FP y cada GT a matched o FN.
    Retorna:
        pred_is_tp [N] bool
        gt_matched [M] bool
    """
    N, M = len(pred_boxes), len(gt_boxes)
    pred_is_tp = [False] * N
    gt_matched = [False] * M

    if N == 0 or M == 0:
        return pred_is_tp, gt_matched

    # Ordenar predicciones por score descendente
    order = pred_scores.argsort(descending=True).tolist()

    for i in order:
        iou = box_iou(pred_boxes[i:i+1], gt_boxes).squeeze(0)  # [M]
        best_iou, best_j = float(iou.max()), int(iou.argmax())
        if best_iou >= iou_thr and not gt_matched[best_j]:
            pred_is_tp[i] = True
            gt_matched[best_j] = True

    return pred_is_tp, gt_matched


# ─── Dibujar imagen anotada ───────────────────────────────────────────────────

def _draw_box(draw: ImageDraw.Draw, box, label: str, color: str,
              dashed: bool = False, font=None):
    """Dibuja un box con etiqueta. dashed=True para GT perdido (FN)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    font = font or FONT_MD

    if dashed:
        # Simular línea discontinua con segmentos
        dash, gap = 8, 4
        for x in range(x1, x2, dash + gap):
            draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=2)
            draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=2)
        for y in range(y1, y2, dash + gap):
            draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=2)
            draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=2)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    if label:
        try:
            bbox_text = font.getbbox(label)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
        except AttributeError:
            tw, th = len(label) * 7, 12
        tx = x1
        ty = max(0, y1 - th - 4)
        draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=color)
        draw.text((tx + 2, ty + 2), label, fill="white", font=font)


def draw_evaluation_image(
    page_pil:     Image.Image,
    query_pil:    Image.Image,
    pred_boxes:   torch.Tensor,
    pred_scores:  torch.Tensor,
    pred_is_tp:   list,          # [N] bool
    gt_boxes:     torch.Tensor,
    gt_matched:   list,          # [M] bool
    class_name:   str,
    score_thr:    float = 0.25,
) -> Image.Image:
    """
    Genera imagen anotada con:
        Azul  sólido  = predicción TP (correcta)
        Rojo  sólido  = predicción FP (falso positivo)
        Verde sólido  = GT matcheado
        Naranja dashed= GT perdido (FN)

    Sketch de query como thumbnail (80×80) en esquina superior derecha.
    """
    img  = page_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Dibujar GT
    for j, box in enumerate(gt_boxes):
        if gt_matched[j]:
            _draw_box(draw, box, "GT", COL_TP_GT, dashed=False, font=FONT_SM)
        else:
            _draw_box(draw, box, "GT-FN", COL_FN_GT, dashed=True, font=FONT_SM)

    # Dibujar predicciones
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        if float(score) < score_thr:
            continue
        color = COL_TP_PRED if pred_is_tp[i] else COL_FP_PRED
        label = f"{float(score):.2f}"
        _draw_box(draw, box, label, color, font=FONT_SM)

    # Thumbnail del sketch (80×80) en esquina superior derecha
    thumb_size = 80
    thumb = query_pil.copy().convert("RGB").resize(
        (thumb_size, thumb_size), Image.LANCZOS
    )
    border = Image.new("RGB", (thumb_size + 4, thumb_size + 4), "#333333")
    border.paste(thumb, (2, 2))
    margin = 8
    img.paste(border, (W - thumb_size - 4 - margin, margin))

    # Header con nombre de clase
    hdr_h = 24
    hdr   = Image.new("RGB", (W, hdr_h), "#1A1A2E")
    hdr_d = ImageDraw.Draw(hdr)
    hdr_d.text((8, 4), f"  {class_name}  |  GT={len(gt_boxes)}  "
               f"TP={sum(gt_matched)}  FN={sum(not m for m in gt_matched)}  "
               f"Pred={sum(1 for s in pred_scores if s >= score_thr)}",
               fill="white", font=FONT_SM)

    canvas = Image.new("RGB", (W, H + hdr_h))
    canvas.paste(hdr, (0, 0))
    canvas.paste(img, (0, hdr_h))
    return canvas


# ─── Evaluación completa del split ───────────────────────────────────────────

@torch.no_grad()
def evaluate_and_save(
    model:       FCOSDetector,
    image_root:  str,
    checkpoint:  str,
    output_dir:  str,
    split:       str   = "val",
    score_thr:   float = 0.25,
    iou_thr:     float = 0.5,
    save_images: bool  = True,
    device:      torch.device = None,
):
    if device is None:
        device = next(model.parameters()).device

    out_path     = Path(output_dir)
    det_path     = out_path / "detections"
    plot_path    = out_path / "plots"
    out_path.mkdir(parents=True, exist_ok=True)
    if save_images:
        det_path.mkdir(exist_ok=True)
    plot_path.mkdir(exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    cfg = {
        "BACKBONE":      C.BACKBONE, "FPN":           C.FPN,
        "FCOS_HEAD":     C.FCOS_HEAD, "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET":       C.DATASET,   "AUGMENTATION":  C.AUGMENTATION,
        "EVAL":          C.EVAL,
    }
    train_ds, val_ds, test_ds = build_datasets(C.DATASET_JSON, image_root, cfg)
    ds = val_ds if split == "val" else test_ds
    print(f"  Split '{split}': {len(ds)} muestras")

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn,
                        pin_memory=False)

    # Metadata del JSON para nombres de clase y rutas de query
    with open(C.DATASET_JSON) as f:
        json_data = json.load(f)
    sorted_ids  = sorted(c["class_id"] for c in json_data["classes"])
    id_to_name  = {c["class_id"]: c["class_name"] for c in json_data["classes"]}
    id_to_idx   = {cid: i for i, cid in enumerate(sorted_ids)}
    idx_to_name = [id_to_name[cid] for cid in sorted_ids]

    # Mapa sample_id → metadata
    sample_meta = {s["sample_id"]: s for s in json_data["samples"]}

    # ── Acumuladores de métricas ───────────────────────────────────────────────
    all_preds   = []   # dicts {boxes, scores, labels}
    all_targets = []   # dicts {boxes, labels}

    # Para estadísticas por clase: total GT, TP, FP, FN
    per_cls_gt  = defaultdict(int)
    per_cls_tp  = defaultdict(int)
    per_cls_fp  = defaultdict(int)
    per_cls_fn  = defaultdict(int)

    model.eval()
    total = len(loader)

    print("  Ejecutando inferencia...")
    for i_batch, batch in enumerate(loader, 1):
        pages   = batch["page_imgs"].to(device)
        queries = batch["query_imgs"].to(device)
        shapes  = batch["img_shapes"]
        targets = batch["targets"]
        sid     = batch["sample_ids"][0]
        meta    = sample_meta.get(sid, {})

        # Inferencia
        dets = model.predict(pages, queries, shapes[0])
        det  = dets[0]  # batch_size=1

        pred_boxes  = det["boxes"].cpu()
        pred_scores = det["scores"].cpu()
        pred_labels = det["labels"].cpu()
        gt_boxes    = targets[0]["boxes"]
        gt_labels   = targets[0]["labels"]

        all_preds.append({
            "boxes":  pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels,
        })
        all_targets.append({
            "boxes":  gt_boxes,
            "labels": gt_labels,
        })

        # ── Por clase: stats TP/FP/FN ─────────────────────────────────────────
        for cls_idx in range(C.FCOS_HEAD["num_classes"]):
            gt_mask   = (gt_labels   == cls_idx)
            pred_mask = (pred_labels == cls_idx)

            gt_cls   = gt_boxes[gt_mask]
            pred_cls = pred_boxes[pred_mask]
            scr_cls  = pred_scores[pred_mask]

            # Filtrar por score threshold
            keep = scr_cls >= score_thr
            pred_cls = pred_cls[keep]
            scr_cls  = scr_cls[keep]

            n_gt   = int(gt_mask.sum())
            n_pred = len(pred_cls)

            per_cls_gt[cls_idx] += n_gt

            if n_gt == 0 and n_pred == 0:
                continue

            pred_is_tp, gt_matched = match_preds_to_gt(
                pred_cls, scr_cls, gt_cls, iou_thr
            )
            tp = sum(pred_is_tp)
            fp = n_pred - tp
            fn = n_gt - sum(gt_matched)

            per_cls_tp[cls_idx] += tp
            per_cls_fp[cls_idx] += fp
            per_cls_fn[cls_idx] += fn

        # ── Guardar imagen anotada ────────────────────────────────────────────
        if save_images:
            class_id = meta.get("class_id")
            cls_idx  = id_to_idx.get(class_id, 0) if class_id is not None else 0
            cls_name = id_to_name.get(class_id, "unknown")

            # Obtener predicciones solo de esta clase
            cls_mask    = (pred_labels == cls_idx)
            cls_boxes   = pred_boxes[cls_mask]
            cls_scores  = pred_scores[cls_mask]

            # Match para colorear
            pred_is_tp, gt_matched = match_preds_to_gt(
                cls_boxes, cls_scores, gt_boxes, iou_thr
            )

            # Cargar imagen original (resizeada) y query
            page_path  = os.path.join(image_root, meta.get("page_path", ""))
            query_paths = ds.query_pool.get(class_id, [])
            query_path  = (os.path.join(image_root, query_paths[0])
                           if query_paths else None)

            try:
                _, page_pil, img_shape = preprocess_page(
                    page_path,
                    C.DATASET["min_size"],
                    C.DATASET["max_size"],
                )
                if query_path:
                    _, query_pil = preprocess_query(query_path,
                                                    C.QUERY_ENCODER.get("size", 224))
                else:
                    query_pil = Image.new("RGB", (224, 224), "#888888")

                out_img = draw_evaluation_image(
                    page_pil, query_pil,
                    cls_boxes, cls_scores, pred_is_tp,
                    gt_boxes, gt_matched,
                    cls_name, score_thr,
                )

                page_stem = Path(meta.get("page_path", f"sample_{sid}")).stem
                fname     = f"{i_batch:04d}_{cls_name}_{page_stem}.jpg"
                out_img.save(det_path / fname, quality=88)

            except Exception as e:
                print(f"    [!] No se pudo guardar imagen {sid}: {e}")

        if i_batch % 10 == 0 or i_batch == total:
            print(f"    {i_batch}/{total} procesadas...", end="\r")

    print()

    # ── Calcular mAP ──────────────────────────────────────────────────────────
    metrics = compute_map(
        all_preds, all_targets,
        num_classes   = C.FCOS_HEAD["num_classes"],
        iou_threshold = iou_thr,
    )

    # ── Generar reporte ────────────────────────────────────────────────────────
    report_path = generate_text_report(
        metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
        idx_to_name, checkpoint, split, len(ds), iou_thr, score_thr,
        out_path,
    )
    print(f"  Reporte guardado: {report_path}")

    # ── Generar plots ──────────────────────────────────────────────────────────
    if HAS_MPL:
        n_plots = generate_plots(
            metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
            idx_to_name, plot_path,
        )
        print(f"  Gráficos guardados en: {plot_path}  ({n_plots} archivos)")
    else:
        print("  (matplotlib no disponible, sin gráficos)")

    print(f"\n  mAP@{iou_thr:.2f} = {metrics['mAP']:.4f}")
    return metrics


# ─── Reporte de texto ─────────────────────────────────────────────────────────

def generate_text_report(
    metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
    idx_to_name, checkpoint, split, n_samples,
    iou_thr, score_thr, out_path,
) -> Path:
    lines = []
    W = 72

    def hline(char="─"):
        lines.append(char * W)

    def section(title):
        lines.append("")
        lines.append(f"{'─── ' + title + ' ':─<{W}}")

    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    lines.append("╔" + "═" * (W - 2) + "╗")
    lines.append("║" + "  iDoc-FCOS — Evaluation Report".center(W - 2) + "║")
    lines.append("╚" + "═" * (W - 2) + "╝")
    lines.append("")
    lines.append(f"  Fecha       : {now}")
    lines.append(f"  Checkpoint  : {checkpoint}")
    lines.append(f"  Split       : {split}  ({n_samples} muestras)")
    lines.append(f"  IoU thresh  : {iou_thr:.2f}")
    lines.append(f"  Score thresh: {score_thr:.2f}  (solo para stats TP/FP/FN)")

    section("Overall")
    lines.append(f"  mAP@{iou_thr:.2f}  :  {metrics['mAP']:.4f}")

    section("Per-class — AP y estadísticas")
    # Cabecera tabla
    h = (f"  {'Rank':>4}  {'Clase':<18}  {'AP':>6}  {'GT':>5}  "
         f"{'TP':>5}  {'FP':>5}  {'FN':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    lines.append(h)
    lines.append("  " + "─" * (len(h) - 2))

    # Ordenar clases por AP descendente
    class_aps = [(cls_idx, metrics["per_class"].get(cls_idx, 0.0))
                 for cls_idx in range(len(idx_to_name))]
    class_aps.sort(key=lambda x: -x[1])

    total_gt, total_tp, total_fp, total_fn = 0, 0, 0, 0

    for rank, (cls_idx, ap) in enumerate(class_aps, 1):
        name = idx_to_name[cls_idx] if cls_idx < len(idx_to_name) else str(cls_idx)
        gt   = per_cls_gt.get(cls_idx, 0)
        tp   = per_cls_tp.get(cls_idx, 0)
        fp   = per_cls_fp.get(cls_idx, 0)
        fn   = per_cls_fn.get(cls_idx, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        total_gt += gt
        total_tp += tp
        total_fp += fp
        total_fn += fn

        marker = "  " if gt > 0 else " ·"   # · = clase sin GT en este split
        lines.append(
            f"{marker}{rank:>4}  {name:<18}  {ap:>6.3f}  {gt:>5}  "
            f"{tp:>5}  {fp:>5}  {fn:>5}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}"
        )

    section("Resumen aggregate")
    prec_agg = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec_agg  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_agg   = 2*prec_agg*rec_agg / (prec_agg+rec_agg) if (prec_agg+rec_agg) > 0 else 0
    lines.append(f"  Total GT boxes    : {total_gt:>6,}")
    lines.append(f"  Total TP          : {total_tp:>6,}  ({100*total_tp/max(total_gt,1):.1f}% recall)")
    lines.append(f"  Total FP          : {total_fp:>6,}  ({100*total_fp/max(total_tp+total_fp,1):.1f}% FP rate)")
    lines.append(f"  Total FN (missed) : {total_fn:>6,}")
    lines.append(f"  Precision global  : {prec_agg:.4f}")
    lines.append(f"  Recall global     : {rec_agg:.4f}")
    lines.append(f"  F1 global         : {f1_agg:.4f}")

    best_cls_idx, best_ap = class_aps[0]
    worst_with_gt = [(idx, ap) for idx, ap in class_aps
                     if per_cls_gt.get(idx, 0) > 0]
    worst_cls_idx, worst_ap = worst_with_gt[-1] if worst_with_gt else (0, 0.0)

    section("Highlights")
    lines.append(f"  Mejor clase   : {idx_to_name[best_cls_idx]}  (AP={best_ap:.4f})")
    lines.append(f"  Peor clase    : {idx_to_name[worst_cls_idx]}  (AP={worst_ap:.4f})")
    n_above_50 = sum(1 for _, ap in class_aps if ap >= 0.5)
    n_zero     = sum(1 for _, ap in class_aps
                     if ap == 0.0 and per_cls_gt.get(_, 0) > 0)
    lines.append(f"  Clases AP≥0.5 : {n_above_50} / {len(class_aps)}")
    lines.append(f"  Clases AP=0   : {n_zero}  (con GT en este split)")

    lines.append("")
    lines.append("─" * W)

    report_path = out_path / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ─── Gráficos ─────────────────────────────────────────────────────────────────

def generate_plots(
    metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
    idx_to_name, plot_path,
) -> int:
    """Genera y guarda todos los gráficos. Retorna número de archivos guardados."""
    n_saved = 0
    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "font.size":    10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor":   "#F8F9FA",
        "axes.grid":        True,
        "grid.alpha":       0.4,
    })

    # ── 1. AP por clase (barras horizontales) ─────────────────────────────────
    class_aps = [(idx, metrics["per_class"].get(idx, 0.0), idx_to_name[idx])
                 for idx in range(len(idx_to_name))
                 if per_cls_gt.get(idx, 0) > 0]
    class_aps.sort(key=lambda x: x[1])  # ascendente para barh

    fig, ax = plt.subplots(figsize=(10, max(5, len(class_aps) * 0.38)))
    aps     = [x[1] for x in class_aps]
    names   = [x[2] for x in class_aps]
    colors  = [plt.cm.RdYlGn(ap) for ap in aps]

    bars = ax.barh(names, aps, color=colors, edgecolor="white", height=0.7)
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{ap:.3f}", va="center", ha="left", fontsize=9)

    ax.axvline(x=metrics["mAP"], color="#2C3E50", linestyle="--",
               linewidth=1.5, label=f"mAP = {metrics['mAP']:.3f}")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Average Precision @ IoU=0.5")
    ax.set_title("AP por clase")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(plot_path / "01_ap_per_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_saved += 1

    # ── 2. Precision / Recall / F1 por clase (grouped bars) ──────────────────
    cls_with_gt = [(idx, idx_to_name[idx]) for idx in range(len(idx_to_name))
                   if per_cls_gt.get(idx, 0) > 0]
    cls_with_gt.sort(key=lambda x: -metrics["per_class"].get(x[0], 0.0))

    idxs   = [x[0] for x in cls_with_gt]
    names2 = [x[1] for x in cls_with_gt]
    precs, recs, f1s = [], [], []
    for idx in idxs:
        tp = per_cls_tp.get(idx, 0)
        fp = per_cls_fp.get(idx, 0)
        fn = per_cls_fn.get(idx, 0)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.0
        precs.append(pr); recs.append(rc); f1s.append(f1)

    x   = np.arange(len(names2))
    w   = 0.28
    fig, ax = plt.subplots(figsize=(max(10, len(names2) * 0.55), 5))
    ax.bar(x - w, precs, w, label="Precision", color="#3498DB", alpha=0.85)
    ax.bar(x,     recs,  w, label="Recall",    color="#2ECC71", alpha=0.85)
    ax.bar(x + w, f1s,   w, label="F1",        color="#E74C3C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names2, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 por clase (threshold=score_thr)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path / "02_precision_recall_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_saved += 1

    # ── 3. TP / FP / FN por clase (stacked bars) ─────────────────────────────
    tps = [per_cls_tp.get(i, 0) for i in idxs]
    fps = [per_cls_fp.get(i, 0) for i in idxs]
    fns = [per_cls_fn.get(i, 0) for i in idxs]

    fig, ax = plt.subplots(figsize=(max(10, len(names2) * 0.55), 5))
    x = np.arange(len(names2))
    ax.bar(x, tps, label="TP", color="#27AE60", alpha=0.9)
    ax.bar(x, fps, bottom=tps, label="FP", color="#E74C3C", alpha=0.85)
    ax.bar(x, fns, bottom=[t + f for t, f in zip(tps, fps)],
           label="FN (missed)", color="#F39C12", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names2, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Número de boxes")
    ax.set_title("TP / FP / FN por clase")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path / "03_tp_fp_fn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_saved += 1

    # ── 4. Resumen global (texto en figura) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    total_gt  = sum(per_cls_gt.values())
    total_tp_ = sum(per_cls_tp.values())
    total_fp_ = sum(per_cls_fp.values())
    total_fn_ = sum(per_cls_fn.values())
    p_agg = total_tp_ / max(total_tp_ + total_fp_, 1)
    r_agg = total_tp_ / max(total_tp_ + total_fn_, 1)
    f_agg = 2*p_agg*r_agg / max(p_agg + r_agg, 1e-9)

    summary_text = (
        f"mAP @ IoU=0.5\n{metrics['mAP']:.4f}\n\n"
        f"Precision  {p_agg:.4f}\n"
        f"Recall     {r_agg:.4f}\n"
        f"F1         {f_agg:.4f}\n\n"
        f"GT boxes   {total_gt:,}\n"
        f"TP         {total_tp_:,}  ({100*total_tp_/max(total_gt,1):.1f}%)\n"
        f"FP         {total_fp_:,}\n"
        f"FN         {total_fn_:,}"
    )
    ax.text(0.5, 0.5, summary_text,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=13, family="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#EBF5FB",
                      edgecolor="#2980B9", linewidth=2))
    ax.set_title("Métricas globales", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(plot_path / "04_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_saved += 1

    return n_saved


# ─── Modo test set (backward compat) ─────────────────────────────────────────

@torch.no_grad()
def evaluate_test_set_simple(model, image_root, device):
    """Evaluación rápida sin guardar imágenes (backward compatible)."""
    cfg = {
        "BACKBONE":      C.BACKBONE, "FPN": C.FPN,
        "FCOS_HEAD":     C.FCOS_HEAD, "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET":       C.DATASET,   "AUGMENTATION": C.AUGMENTATION,
        "EVAL":          C.EVAL,
    }
    _, _, test_ds = build_datasets(C.DATASET_JSON, image_root, cfg)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)
    ev = DetectionEvaluator(num_classes=C.FCOS_HEAD["num_classes"],
                            iou_threshold=C.EVAL["iou_threshold"])
    for batch in loader:
        dets = model.predict(batch["page_imgs"].to(device),
                             batch["query_imgs"].to(device),
                             batch["img_shapes"][0])
        ev.update(
            [{"boxes": d["boxes"].cpu(), "scores": d["scores"].cpu(),
              "labels": d["labels"].cpu()} for d in dets],
            batch["targets"],
        )
    return ev.compute()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nCargando modelo...")
    model = load_model(args.checkpoint, device)

    # ── Inferencia single ─────────────────────────────────────────────────────
    if args.page_img and args.query_img:
        print(f"\nInferencia: {args.page_img}  +  query: {args.query_img}")

        page_t, page_pil, img_shape = preprocess_page(
            args.page_img, C.DATASET["min_size"], C.DATASET["max_size"]
        )
        query_t, query_pil = preprocess_query(
            args.query_img, C.QUERY_ENCODER.get("size", 224)
        )
        dets = model.predict(page_t.to(device), query_t.to(device), img_shape)
        det  = dets[0]

        with open(C.DATASET_JSON) as f:
            jdata = json.load(f)
        sorted_ids  = sorted(c["class_id"] for c in jdata["classes"])
        idx_to_name = [jdata["id_to_name"][cid] if "id_to_name" in jdata
                       else {c["class_id"]: c["class_name"] for c in jdata["classes"]}[cid]
                       for cid in sorted_ids]

        pred_is_tp = [False] * len(det["boxes"])
        gt_matched = []
        out_img = draw_evaluation_image(
            page_pil, query_pil,
            det["boxes"], det["scores"], pred_is_tp,
            torch.zeros((0, 4)), gt_matched,
            "query", args.score_thr,
        )
        out_img.save(args.output_img)
        print(f"  {len(det['boxes'])} detecciones → {args.output_img}")
        return

    # ── Evaluación completa ───────────────────────────────────────────────────
    if args.image_root:
        print(f"\nEvaluando split '{args.split}' → {args.output_dir}")
        save_imgs = args.save_images and not args.no_images
        evaluate_and_save(
            model        = model,
            image_root   = args.image_root,
            checkpoint   = args.checkpoint,
            output_dir   = args.output_dir,
            split        = args.split,
            score_thr    = args.score_thr,
            iou_thr      = args.iou_thr,
            save_images  = save_imgs,
            device       = device,
        )
    else:
        print("Especifica --image_root para evaluar, o --page_img + --query_img "
              "para inferencia single.")


if __name__ == "__main__":
    main()