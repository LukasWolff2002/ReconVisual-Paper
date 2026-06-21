"""
evaluate.py  —  iDoc-FCOS  |  Evaluación completa

Genera automáticamente:
  eval_output/
  ├── report.txt                  — reporte de texto completo con todas las métricas
  ├── detections/                 — imágenes anotadas por muestra (TP/FP/FN/GT)
  ├── plots/
  │   ├── 01_ap_per_class.png     — AP por clase (barras horizontales, color RdYlGn)
  │   ├── 02_precision_recall_f1.png
  │   ├── 03_tp_fp_fn.png
  │   ├── 04_summary_dashboard.png — dashboard 2×2 con métricas globales
  │   ├── 05_score_distribution.png
  │   ├── 06_pr_curves.png        — curvas P/R por clase
  │   ├── 07_confusion_matrix.png — heatmap GT vs predicho
  │   └── 08_worst_best_grid.png  — grid de las mejores/peores clases
  └── kit/
      ├── kit_im.txt              — solución image retrieval (formato oficial)
      └── kit_ps.txt              — solución pattern spotting (formato oficial)

Modos de uso:

  # Evaluar val set:
  python -m train_fcos.evaluate \\
      --checkpoint train_fcos/run_03/best_model.pth \\
      --image_root /home/rvdl_2/ \\
      --output_dir eval_results/ \\
      --split val

  # Evaluar test set (también genera archivos para el kit oficial):
  python -m train_fcos.evaluate \\
      --checkpoint train_fcos/run_03/best_model.pth \\
      --image_root /home/rvdl_2/ \\
      --output_dir eval_results/ \\
      --split test

  # Inferencia single (página + query sketch):
  python -m train_fcos.evaluate \\
      --checkpoint train_fcos/run_03/best_model.pth \\
      --page_img page.jpg --query_img sketch.jpg \\
      --output_img result.jpg
"""

import os
import sys
import json
import math
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torchvision.transforms.functional as TF
import torchvision.transforms as T
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
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MPL = True
except ImportError:
    print("[evaluate] matplotlib no disponible — intentando instalar...")
    import subprocess as _sp
    # Intentar instalar (funciona tanto en venv como en sistema)
    ret = _sp.run(
        [sys.executable, "-m", "pip", "install", "matplotlib", "--quiet"],
        capture_output=True
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        HAS_MPL = True
        print("[evaluate] matplotlib instalado correctamente.")
    except ImportError:
        HAS_MPL = False
        print("[evaluate] matplotlib sigue sin estar disponible — se omiten gráficos.")
        print(f"           Instala manualmente: pip install matplotlib")


# ══════════════════════════════════════════════════════════════════════════════
# PALETA Y CONSTANTES VISUALES
# ══════════════════════════════════════════════════════════════════════════════

# Colores para anotaciones en imágenes
COL_TP_PRED = "#4C9BE8"   # azul   — predicción correcta (TP)
COL_FP_PRED = "#E85C4C"   # rojo   — predicción falsa   (FP)
COL_TP_GT   = "#2ECC71"   # verde  — GT matcheado
COL_FN_GT   = "#F39C12"   # naranja— GT perdido         (FN)
COL_HDR_BG  = "#1A1A2E"   # fondo header imagen

# Paleta para gráficas matplotlib
PLT_BLUE    = "#3498DB"
PLT_GREEN   = "#27AE60"
PLT_RED     = "#E74C3C"
PLT_ORANGE  = "#E67E22"
PLT_PURPLE  = "#9B59B6"
PLT_BG      = "#F8F9FA"
PLT_GRID    = "#DEE2E6"

# rcParams globales para matplotlib
MPL_RC = {
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "figure.facecolor":   "white",
    "axes.facecolor":     PLT_BG,
    "axes.grid":          True,
    "grid.color":         PLT_GRID,
    "grid.alpha":         0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE FUENTES
# ══════════════════════════════════════════════════════════════════════════════

def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT_LG = _load_font(16)
FONT_MD = _load_font(13)
FONT_SM = _load_font(11)


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Evaluación completa iDoc-FCOS")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--image_root",    type=str, default=None)
    p.add_argument("--dataset_json",  type=str, default=C.DATASET_JSON)
    p.add_argument("--output_dir",    type=str, default="eval_output")
    p.add_argument("--split",         type=str, default="val",
                   choices=["val", "test"])
    p.add_argument("--score_thr",     type=float, default=0.25)
    p.add_argument("--iou_thr",       type=float, default=0.5)
    p.add_argument("--save_images",   action="store_true", default=True)
    p.add_argument("--no_images",     action="store_true")
    p.add_argument("--max_det_imgs",  type=int, default=200,
                   help="Número máximo de imágenes de detección guardadas.")
    p.add_argument("--kit_dir",       type=str,
                   default=None,
                   help="Ruta al directorio evaluation_kit_v2/ que contiene "
                        "el binario 'main'. Si se especifica, se ejecuta "
                        "automáticamente tras generar kit_im.txt y kit_ps.txt.")
    # Single inference
    p.add_argument("--page_img",      type=str, default=None)
    p.add_argument("--query_img",     type=str, default=None)
    p.add_argument("--output_img",    type=str, default="result.jpg")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# PRE-PROCESADO
# ══════════════════════════════════════════════════════════════════════════════

NORMALIZE = T.Normalize(mean=C.DATASET["pixel_mean"], std=C.DATASET["pixel_std"])


def preprocess_page(img_path: str, min_size: int, max_size: int):
    img = Image.open(img_path).convert("RGB")
    W0, H0 = img.size
    scale = min_size / min(H0, W0)
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


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING PRED ↔ GT
# ══════════════════════════════════════════════════════════════════════════════

def match_preds_to_gt(
    pred_boxes:  torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes:    torch.Tensor,
    iou_thr:     float = 0.5,
):
    """Asigna cada pred a TP/FP y cada GT a matched/FN."""
    N, M = len(pred_boxes), len(gt_boxes)
    pred_is_tp = [False] * N
    gt_matched = [False] * M

    if N == 0 or M == 0:
        return pred_is_tp, gt_matched

    order = pred_scores.argsort(descending=True).tolist()
    for i in order:
        iou = box_iou(pred_boxes[i:i+1], gt_boxes).squeeze(0)
        best_iou, best_j = float(iou.max()), int(iou.argmax())
        if best_iou >= iou_thr and not gt_matched[best_j]:
            pred_is_tp[i] = True
            gt_matched[best_j] = True

    return pred_is_tp, gt_matched


# ══════════════════════════════════════════════════════════════════════════════
# DIBUJO DE IMÁGENES ANOTADAS
# ══════════════════════════════════════════════════════════════════════════════

def _text_size(font: ImageFont.ImageFont, text: str):
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return len(text) * 7, 12


def _draw_box(draw: ImageDraw.Draw, box, label: str, color: str,
              dashed: bool = False, font=None, width: int = 2):
    x1, y1, x2, y2 = [int(v) for v in box]
    font = font or FONT_MD

    if dashed:
        dash, gap = 9, 4
        for x in range(x1, x2, dash + gap):
            draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
            draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
        for y in range(y1, y2, dash + gap):
            draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
            draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if label:
        tw, th = _text_size(font, label)
        tx, ty = x1, max(0, y1 - th - 4)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=color)
        draw.text((tx + 3, ty + 2), label, fill="white", font=font)


def _legend_strip(width: int) -> Image.Image:
    """Barra de leyenda de colores al pie de la imagen."""
    h    = 22
    strip = Image.new("RGB", (width, h), "#2C3E50")
    draw  = ImageDraw.Draw(strip)
    items = [
        (COL_TP_PRED, "Pred TP"),
        (COL_FP_PRED, "Pred FP"),
        (COL_TP_GT,   "GT matched"),
        (COL_FN_GT,   "GT missed (FN)"),
    ]
    x = 8
    for color, label in items:
        draw.rectangle([x, 5, x + 12, 17], fill=color)
        draw.text((x + 16, 4), label, fill="white", font=FONT_SM)
        tw, _ = _text_size(FONT_SM, label)
        x += 16 + tw + 20
    return strip


def draw_evaluation_image(
    page_pil:    Image.Image,
    query_pil:   Image.Image,
    pred_boxes:  torch.Tensor,
    pred_scores: torch.Tensor,
    pred_is_tp:  list,
    gt_boxes:    torch.Tensor,
    gt_matched:  list,
    class_name:  str,
    score_thr:   float = 0.25,
    sample_id:   str   = "",
) -> Image.Image:
    """
    Imagen anotada completa con:
      - Header con stats
      - Bounding boxes coloreados (TP azul, FP rojo, GT verde, FN naranja dashed)
      - Thumbnail del sketch de query
      - Barra de leyenda inferior
    """
    img  = page_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # ── GT boxes ──────────────────────────────────────────────────────────────
    for j, box in enumerate(gt_boxes):
        if gt_matched[j]:
            _draw_box(draw, box, "GT✓", COL_TP_GT,  dashed=False, font=FONT_SM)
        else:
            _draw_box(draw, box, "GT✗", COL_FN_GT,  dashed=True,  font=FONT_SM)

    # ── Pred boxes ────────────────────────────────────────────────────────────
    n_shown = 0
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        if float(score) < score_thr:
            continue
        color = COL_TP_PRED if pred_is_tp[i] else COL_FP_PRED
        _draw_box(draw, box, f"{score:.2f}", color, font=FONT_SM)
        n_shown += 1

    # ── Thumbnail query ───────────────────────────────────────────────────────
    thumb_size = 88
    thumb  = query_pil.copy().convert("RGB").resize(
        (thumb_size, thumb_size), Image.LANCZOS)
    border = Image.new("RGB", (thumb_size + 4, thumb_size + 4), "#2C3E50")
    border.paste(thumb, (2, 2))
    img.paste(border, (W - thumb_size - 4 - 8, 8))

    # Mini label bajo el thumbnail
    draw.rectangle([W - thumb_size - 4 - 8, thumb_size + 14,
                    W - 8, thumb_size + 28], fill="#2C3E50")
    draw.text((W - thumb_size - 4 - 4, thumb_size + 15),
              "query", fill="white", font=FONT_SM)

    # ── Header ────────────────────────────────────────────────────────────────
    n_gt   = len(gt_boxes)
    n_tp   = sum(gt_matched)
    n_fn   = n_gt - n_tp
    n_pred = n_shown
    prec   = n_tp / (n_tp + (n_pred - n_tp)) if n_pred > 0 else 0.0
    rec    = n_tp / n_gt if n_gt > 0 else 0.0

    hdr_h  = 28
    hdr    = Image.new("RGB", (W, hdr_h), COL_HDR_BG)
    hdr_d  = ImageDraw.Draw(hdr)
    info   = (f"  {class_name}  │  GT={n_gt}  TP={n_tp}  FN={n_fn}  "
              f"Pred={n_pred}  │  P={prec:.2f}  R={rec:.2f}"
              + (f"  │  id={sample_id}" if sample_id else ""))
    hdr_d.text((6, 6), info, fill="#ECF0F1", font=FONT_SM)

    # ── Leyenda inferior ──────────────────────────────────────────────────────
    legend = _legend_strip(W)

    # ── Composición final ─────────────────────────────────────────────────────
    canvas = Image.new("RGB", (W, H + hdr_h + legend.height))
    canvas.paste(hdr,    (0, 0))
    canvas.paste(img,    (0, hdr_h))
    canvas.paste(legend, (0, hdr_h + H))
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN AL KIT OFICIAL
# ══════════════════════════════════════════════════════════════════════════════

def _page_stem(page_path: str) -> str:
    return Path(page_path).stem   # 'page1042.jpg' → 'page1042'


def export_kit_files(
    all_preds:   list,
    all_targets: list,
    sample_ids:  list,
    sample_meta: dict,
    id_to_idx:   dict,
    image_root:  str,
    output_dir:  str,
    score_thr:   float = 0.25,
    img_shapes:  list  = None,
) -> tuple:
    """
    Genera kit/kit_im.txt y kit/kit_ps.txt compatibles con el binario `main`
    del evaluation kit de DocExplore.

    Formato IM:   query_id:\\tpage851\\tpage208\\t...\\r
    Formato PS:   query_id:\\tpage851_x1_y1_x2_y2\\t...\\r

    Las coordenadas PS se mantienen en el espacio de inferencia (el kit evalúa
    contra anotaciones en el espacio reescalado a 1024 px).
    """
    kit_dir = Path(output_dir) / "kit"
    kit_dir.mkdir(parents=True, exist_ok=True)

    # Acumuladores
    im_data: dict[int, dict[str, float]]           = defaultdict(dict)
    ps_data: dict[int, list]                        = defaultdict(list)
    all_qids: set[int]                              = set()

    for i, (preds, sid) in enumerate(zip(all_preds, sample_ids)):
        meta      = sample_meta.get(sid, {})
        class_id  = meta.get("class_id")
        page_path = meta.get("page_path", "")
        if class_id is None:
            continue

        query_id = int(class_id)
        all_qids.add(query_id)
        page_id  = _page_stem(page_path)
        cls_idx  = id_to_idx.get(class_id)
        if cls_idx is None:
            continue

        mask   = (preds["labels"] == cls_idx) & (preds["scores"] >= score_thr)
        boxes  = preds["boxes"][mask]
        scores = preds["scores"][mask]

        # Image Retrieval — keep best score per page
        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        if page_id not in im_data[query_id] or max_score > im_data[query_id][page_id]:
            im_data[query_id][page_id] = max_score

        # Pattern Spotting — all detections
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            ps_data[query_id].append(
                (float(score), page_id, x1, y1, x2, y2)
            )

    # El kit SIEMPRE exige exactamente 1447 queries (0..1446).
    # Las queries que no están en el split se rellenan con línea vacía.
    ALL_QUERY_IDS = list(range(1447))

    # ── Escribir IM ───────────────────────────────────────────────────────────
    im_lines = []
    for qid in ALL_QUERY_IDS:
        if qid in im_data and im_data[qid]:
            ranked = sorted(im_data[qid].items(), key=lambda x: -x[1])
            pages  = "\t".join(p for p, _ in ranked)
            im_lines.append(f"{qid}:\t{pages}\r")
        else:
            im_lines.append(f"{qid}:\r")   # query vacía — no penaliza
    im_path = kit_dir / "kit_im.txt"
    im_path.write_text("\n".join(im_lines), encoding="utf-8")

    # ── Escribir PS ───────────────────────────────────────────────────────────
    ps_lines = []
    for qid in ALL_QUERY_IDS:
        if qid in ps_data and ps_data[qid]:
            entries = sorted(ps_data[qid], key=lambda x: -x[0])
            dets    = "\t".join(f"{pid}_{x1}_{y1}_{x2}_{y2}"
                                for _, pid, x1, y1, x2, y2 in entries)
            ps_lines.append(f"{qid}:\t{dets}\r")
        else:
            ps_lines.append(f"{qid}:\r")   # query vacía
    ps_path = kit_dir / "kit_ps.txt"
    ps_path.write_text("\n".join(ps_lines), encoding="utf-8")

    n_active = len(all_qids)
    print(f"  [kit] kit_im.txt → {im_path}  "
          f"({n_active} activas + {1447-n_active} vacías = 1447 queries)")
    print(f"  [kit] kit_ps.txt → {ps_path}  "
          f"({n_active} activas + {1447-n_active} vacías = 1447 queries)")
    return im_path, ps_path


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN SOBRE EL GT DEL KIT OFICIAL
# ══════════════════════════════════════════════════════════════════════════════

def _parse_ps_file(filepath: str) -> dict:
    """
    Parsea un archivo PS (ground truth o predicciones).
    Formato de cada línea:
        query_id: pageID_x1_y1_x2_y2 pageID_x1_y1_x2_y2 ...

    Devuelve:
        {query_id (int): [(page_id (str), x1, y1, x2, y2), ...]}
    """
    result = {}
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip("\r").rstrip()
            if not line:
                continue
            if ":" not in line:
                continue
            qid_str, rest = line.split(":", 1)
            try:
                qid = int(qid_str.strip())
            except ValueError:
                continue
            boxes = []
            for token in rest.split():
                token = token.strip()
                if not token:
                    continue
                parts = token.rsplit("-", 4)   # pageID-x1-y1-x2-y2
                if len(parts) != 5:
                    continue
                page_id = parts[0]
                try:
                    x1, y1, x2, y2 = int(parts[1]), int(parts[2]), \
                                      int(parts[3]), int(parts[4])
                    boxes.append((page_id, x1, y1, x2, y2))
                except ValueError:
                    continue
            result[qid] = boxes
    return result


def _parse_im_file(filepath: str) -> dict:
    """
    Parsea un archivo IM (ground truth o predicciones).
    Formato de cada línea:
        query_id: pageID pageID pageID ...

    Devuelve:
        {query_id (int): [page_id (str), ...]}   (orden = ranking)
    """
    result = {}
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip("\r").rstrip()
            if not line or ":" not in line:
                continue
            qid_str, rest = line.split(":", 1)
            try:
                qid = int(qid_str.strip())
            except ValueError:
                continue
            pages = [t.strip() for t in rest.split() if t.strip()]
            result[qid] = pages
    return result


def _iou_box(b1, b2) -> float:
    """IoU entre dos boxes (x1,y1,x2,y2)."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _match_kit_boxes(pred_boxes: list, gt_boxes: list, iou_thr: float = 0.5):
    """
    pred_boxes: [(x1,y1,x2,y2), ...]  — ya en orden de confianza descendente
    gt_boxes:   [(x1,y1,x2,y2), ...]
    Devuelve:
        pred_is_tp [N] bool
        gt_matched [M] bool
    """
    N, M = len(pred_boxes), len(gt_boxes)
    pred_is_tp = [False] * N
    gt_matched = [False] * M
    for i in range(N):
        best_iou, best_j = 0.0, -1
        for j in range(M):
            if gt_matched[j]:
                continue
            iou = _iou_box(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            pred_is_tp[i] = True
            gt_matched[best_j] = True
    return pred_is_tp, gt_matched


def generate_kit_visual_report(
    gt_ps_file:   str,
    pred_ps_file: str,
    image_root:   str,
    output_dir:   str,
    iou_thr:      float = 0.5,
    max_images:   int   = 150,
    score_thr:    float = 0.0,   # PS no tiene scores, se ignora
):
    """
    Genera imágenes anotadas comparando el GT del kit (ps_example.txt / gt_ps_file)
    contra las predicciones del modelo (kit_ps.txt / pred_ps_file).

    Para cada query, agrupa las detecciones por página y dibuja:
        Verde sólido   — GT matcheado (TP)
        Naranja dashed — GT perdido   (FN)
        Azul sólido    — Pred TP
        Rojo sólido    — Pred FP

    Estructura de salida:
        output_dir/kit_visuals/
            QXXX_pageYYY_tp{N}_fn{M}_fp{K}.jpg
    """
    out_dir = Path(output_dir) / "kit_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [kit_visuals] Parseando GT:   {gt_ps_file}")
    gt_data   = _parse_ps_file(gt_ps_file)
    print(f"  [kit_visuals] Parseando pred: {pred_ps_file}")
    pred_data = _parse_ps_file(pred_ps_file)

    if not gt_data:
        print("  [kit_visuals] ✗ GT vacío — verifica la ruta del archivo.")
        return
    if not pred_data:
        print("  [kit_visuals] ✗ Predicciones vacías — ejecuta primero la evaluación.")
        return

    # Solo queries que tienen PREDICCIONES (evita iterar 1447 queries vacías)
    pred_query_ids = sorted(pred_data.keys())
    active_qids    = [qid for qid in pred_query_ids if pred_data.get(qid)]
    print(f"  [kit_visuals] {len(gt_data)} queries en GT, "
          f"{len(active_qids)} con predicciones activas.")

    if not active_qids:
        print("  [kit_visuals] Sin predicciones que visualizar.")
        return

    # Pre-construir índice de imágenes para búsqueda rápida (evita stat por cada box)
    print(f"  [kit_visuals] Indexando imágenes en {image_root} ...")
    img_index: dict[str, Path] = {}
    if image_root and Path(image_root).exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
                    "*.JPG", "*.JPEG", "*.PNG"]:
            for p in Path(image_root).rglob(ext):
                img_index[p.stem] = p
    print(f"  [kit_visuals] {len(img_index)} imágenes indexadas.")

    imgs_saved = 0
    total_tp_global = 0
    total_fn_global = 0
    total_fp_global = 0

    for qid in active_qids:
        if imgs_saved >= max_images:
            break

        gt_entries   = gt_data.get(qid, [])
        pred_entries = pred_data.get(qid, [])

        # Agrupar por página
        gt_by_page: dict[str, list]   = defaultdict(list)
        pred_by_page: dict[str, list] = defaultdict(list)

        for page, x1, y1, x2, y2 in gt_entries:
            gt_by_page[page].append((x1, y1, x2, y2))
        for page, x1, y1, x2, y2 in pred_entries:
            pred_by_page[page].append((x1, y1, x2, y2))

        # Páginas con actividad (GT o pred)
        pages_with_activity = sorted(
            set(gt_by_page.keys()) | set(pred_by_page.keys())
        )

        for page_id in pages_with_activity:
            if imgs_saved >= max_images:
                break

            gt_boxes   = gt_by_page.get(page_id, [])
            pred_boxes = pred_by_page.get(page_id, [])

            pred_is_tp, gt_matched = _match_kit_boxes(
                pred_boxes, gt_boxes, iou_thr)

            tp = sum(pred_is_tp)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - sum(gt_matched)

            total_tp_global += tp
            total_fn_global += fn
            total_fp_global += fp

            # Buscar imagen usando el índice pre-construido
            page_img_path = img_index.get(page_id)
            if page_img_path is None:
                continue   # no tenemos la imagen, omitir

            try:
                # Cargar y redimensionar igual que en inferencia
                page_pil_raw = Image.open(page_img_path).convert("RGB")
                W0, H0 = page_pil_raw.size
                min_s, max_s = C.DATASET["min_size"], C.DATASET["max_size"]
                scale = min_s / min(H0, W0)
                if scale * max(H0, W0) > max_s:
                    scale = max_s / max(H0, W0)
                new_H = int(round(H0 * scale))
                new_W = int(round(W0 * scale))
                page_pil = page_pil_raw.resize((new_W, new_H), Image.BILINEAR)

                # Escalar boxes al mismo espacio que la imagen redimensionada
                def _scale_boxes(boxes, sx, sy):
                    return [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                            for x1, y1, x2, y2 in boxes]

                sx, sy = new_W / W0, new_H / H0
                gt_scaled   = _scale_boxes(gt_boxes,   sx, sy)
                pred_scaled = _scale_boxes(pred_boxes, sx, sy)

                # Dibujar imagen anotada
                img  = page_pil.copy()
                draw = ImageDraw.Draw(img)
                W, H = img.size

                # GT boxes
                for j, box in enumerate(gt_scaled):
                    if gt_matched[j]:
                        _draw_box(draw, box, "GT✓", COL_TP_GT,
                                  dashed=False, font=FONT_SM)
                    else:
                        _draw_box(draw, box, "GT✗", COL_FN_GT,
                                  dashed=True, font=FONT_SM)

                # Pred boxes
                for i, box in enumerate(pred_scaled):
                    color = COL_TP_PRED if pred_is_tp[i] else COL_FP_PRED
                    label = "TP" if pred_is_tp[i] else "FP"
                    _draw_box(draw, box, label, color,
                              dashed=False, font=FONT_SM)

                # Header
                hdr_h = 28
                hdr   = Image.new("RGB", (W, hdr_h), COL_HDR_BG)
                hdr_d = ImageDraw.Draw(hdr)
                info  = (f"  Query={qid}  │  {page_id}  │  "
                         f"GT={len(gt_boxes)}  TP={tp}  FN={fn}  FP={fp}  "
                         f"│  IoU≥{iou_thr}")
                hdr_d.text((6, 6), info, fill="#ECF0F1", font=FONT_SM)

                # Leyenda
                legend = _legend_strip(W)

                canvas = Image.new("RGB", (W, H + hdr_h + legend.height))
                canvas.paste(hdr,    (0, 0))
                canvas.paste(img,    (0, hdr_h))
                canvas.paste(legend, (0, hdr_h + H))

                fname = f"Q{qid:04d}_{page_id}_tp{tp}_fn{fn}_fp{fp}.jpg"
                canvas.save(out_dir / fname, quality=88)
                imgs_saved += 1

            except Exception as e:
                print(f"    [!] Q{qid} {page_id}: {e}")
                continue

    # Resumen
    total_pred = total_tp_global + total_fp_global
    prec = total_tp_global / max(total_pred, 1)
    rec  = total_tp_global / max(total_tp_global + total_fn_global, 1)
    f1   = 2*prec*rec / max(prec+rec, 1e-9)

    print(f"\n  [kit_visuals] ✓  {imgs_saved} imágenes guardadas → {out_dir}")
    print(f"  [kit_visuals]    TP={total_tp_global}  FP={total_fp_global}  "
          f"FN={total_fn_global}  |  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # Guardar resumen txt
    summary_path = out_dir / "kit_visual_summary.txt"
    lines = [
        "Kit Visual Report — Comparación GT vs Predicciones PS",
        f"GT file  : {gt_ps_file}",
        f"Pred file: {pred_ps_file}",
        f"IoU thr  : {iou_thr}",
        f"Imágenes : {imgs_saved}",
        "",
        f"TP total : {total_tp_global}",
        f"FP total : {total_fp_global}",
        f"FN total : {total_fn_global}",
        f"Precision: {prec:.4f}",
        f"Recall   : {rec:.4f}",
        f"F1       : {f1:.4f}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return imgs_saved


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN AUTOMÁTICA DEL KIT OFICIAL
# ══════════════════════════════════════════════════════════════════════════════

def run_official_kit(
    kit_bin_dir: str,
    kit_files_dir: str,
    iou_thr: float = 0.5,
) -> dict:
    """
    Ejecuta el binario `main` del evaluation kit de DocExplore para ambas
    tareas (im y ps) y devuelve un dict con los mAP oficiales.

    Args:
        kit_bin_dir  : directorio que contiene el binario 'main'
                       (ej. '/home/rvdl_2/evaluation_kit_v2')
        kit_files_dir: directorio donde están kit_im.txt y kit_ps.txt
                       (normalmente output_dir/kit/)
        iou_thr      : IoU para la tarea ps (default 0.5)

    Returns:
        {
          "im_map":      float | None,
          "ps_map":      float | None,
          "im_out_path": Path,
          "ps_out_path": Path,
          "im_stdout":   str,
          "ps_stdout":   str,
        }
    """
    import subprocess

    kit_bin_dir   = Path(kit_bin_dir).expanduser().resolve()
    kit_files_dir = Path(kit_files_dir).expanduser().resolve()
    binary        = kit_bin_dir / "main"

    if not binary.exists():
        print(f"  [kit] ✗  Binario no encontrado: {binary}")
        return {}

    im_in   = kit_files_dir / "kit_im.txt"
    ps_in   = kit_files_dir / "kit_ps.txt"
    im_out  = kit_files_dir / "result_im.txt"
    ps_out  = kit_files_dir / "result_ps.txt"

    results = {
        "im_map": None, "ps_map": None,
        "im_out_path": im_out, "ps_out_path": ps_out,
        "im_stdout": "", "ps_stdout": "",
        "_image_root": "",   # se sobreescribe desde evaluate_and_save
        "_max_images": 150,
    }

    def _run(task: str, in_file: Path, out_file: Path, extra_args: list = None):
        if not in_file.exists():
            print(f"  [kit] ✗  Archivo no encontrado: {in_file}")
            return None, ""
        cmd = [str(binary), "--task", task,
               "--in_file",  str(in_file),
               "--out_file", str(out_file)]
        if extra_args:
            cmd += extra_args
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(kit_bin_dir),
                capture_output=True,
                text=True,
                timeout=600,
            )
            stdout = proc.stdout + proc.stderr
            if proc.returncode != 0:
                print(f"  [kit] ✗  Error ejecutando '{task}':\n{stdout}")
                return None, stdout

            # Leer mAP del archivo de salida
            map_val = None
            if out_file.exists():
                content = out_file.read_text(encoding="utf-8", errors="replace")
                # El kit escribe algo como: "mAP: 0.3421" o solo el número
                for line in content.splitlines():
                    line = line.strip()
                    # Intentar parsear "mAP: 0.3421" o "0.3421"
                    for token in line.replace(":", " ").replace("=", " ").split():
                        try:
                            val = float(token)
                            if 0.0 <= val <= 1.0:
                                map_val = val
                                break
                        except ValueError:
                            pass
                    if map_val is not None:
                        break
            return map_val, stdout

        except subprocess.TimeoutExpired:
            print(f"  [kit] ✗  Timeout ejecutando '{task}' (>600 s)")
            return None, ""
        except Exception as e:
            print(f"  [kit] ✗  Excepción ejecutando '{task}': {e}")
            return None, ""

    # ── Image Retrieval ────────────────────────────────────────────────────────
    print(f"\n  [kit] Ejecutando task=im ...")
    im_map, im_stdout = _run("im", im_in, im_out)
    results["im_map"]    = im_map
    results["im_stdout"] = im_stdout
    if im_map is not None:
        print(f"  [kit] ✓  IM  mAP (oficial) = {im_map:.4f}")
    print(f"  [kit]    resultado → {im_out}")

    # ── Pattern Spotting ───────────────────────────────────────────────────────
    print(f"  [kit] Ejecutando task=ps (IoU={iou_thr}) ...")
    ps_map, ps_stdout = _run("ps", ps_in, ps_out,
                              extra_args=["--iou", str(iou_thr)])
    results["ps_map"]    = ps_map
    results["ps_stdout"] = ps_stdout
    if ps_map is not None:
        print(f"  [kit] ✓  PS  mAP (oficial) = {ps_map:.4f}")
    print(f"  [kit]    resultado → {ps_out}")

    # ── Imágenes visuales GT vs Predicciones ──────────────────────────────────
    gt_ps_file = kit_bin_dir / "ps_example.txt"
    if gt_ps_file.exists() and ps_in.exists():
        generate_kit_visual_report(
            gt_ps_file   = str(gt_ps_file),
            pred_ps_file = str(ps_in),
            image_root   = results["_image_root"],
            output_dir   = str(kit_files_dir.parent),
            iou_thr      = iou_thr,
            max_images   = results["_max_images"],
        )
    else:
        if not gt_ps_file.exists():
            print(f"  [kit_visuals] ps_example.txt no encontrado en {kit_bin_dir}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

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
    max_det_imgs:int   = 200,
    kit_dir:     str   = None,
    device:      torch.device = None,
):
    if device is None:
        device = next(model.parameters()).device

    out_path  = Path(output_dir)
    det_path  = out_path / "detections"
    plot_path = out_path / "plots"
    out_path.mkdir(parents=True, exist_ok=True)
    if save_images:
        det_path.mkdir(exist_ok=True)
    plot_path.mkdir(exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    cfg = {
        "BACKBONE":     C.BACKBONE,  "FPN":           C.FPN,
        "FCOS_HEAD":    C.FCOS_HEAD, "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET":      C.DATASET,   "AUGMENTATION":  C.AUGMENTATION,
        "EVAL":         C.EVAL,
    }
    train_ds, val_ds, test_ds = build_datasets(C.DATASET_JSON, image_root, cfg)
    ds = val_ds if split == "val" else test_ds
    print(f"  Split '{split}': {len(ds)} muestras")

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=False)

    # Metadata del JSON
    with open(C.DATASET_JSON) as f:
        json_data = json.load(f)
    sorted_ids  = sorted(c["class_id"] for c in json_data["classes"])
    id_to_name  = {c["class_id"]: c["class_name"] for c in json_data["classes"]}
    id_to_idx   = {cid: i for i, cid in enumerate(sorted_ids)}
    idx_to_name = [id_to_name[cid] for cid in sorted_ids]
    sample_meta = {s["sample_id"]: s for s in json_data["samples"]}

    num_classes = C.FCOS_HEAD["num_classes"]

    # ── Acumuladores ───────────────────────────────────────────────────────────
    all_preds   = []
    all_targets = []
    all_sids    = []
    all_shapes  = []

    # Por clase: listas de (score, is_tp) para curvas P/R
    cls_score_tp: dict[int, list] = defaultdict(list)

    # Stats globales
    per_cls_gt  = defaultdict(int)
    per_cls_tp  = defaultdict(int)
    per_cls_fp  = defaultdict(int)
    per_cls_fn  = defaultdict(int)

    # Scores de todas las predicciones (para distribución)
    all_scores_tp = []
    all_scores_fp = []

    # Confusion: gt_cls → pred_cls counts
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    model.eval()
    total      = len(loader)
    imgs_saved = 0

    print("  Ejecutando inferencia...")
    for i_batch, batch in enumerate(loader, 1):
        pages   = batch["page_imgs"].to(device)
        queries = batch["query_imgs"].to(device)
        shapes  = batch["img_shapes"]
        targets = batch["targets"]
        sid     = batch["sample_ids"][0]
        meta    = sample_meta.get(sid, {})

        dets        = model.predict(pages, queries, shapes[0])
        det         = dets[0]
        pred_boxes  = det["boxes"].cpu()
        pred_scores = det["scores"].cpu()
        pred_labels = det["labels"].cpu()
        gt_boxes    = targets[0]["boxes"]
        gt_labels   = targets[0]["labels"]

        all_preds.append({"boxes": pred_boxes, "scores": pred_scores,
                          "labels": pred_labels})
        all_targets.append({"boxes": gt_boxes, "labels": gt_labels})
        all_sids.append(sid)
        all_shapes.append(shapes[0])

        # ── Stats por clase ────────────────────────────────────────────────────
        class_id = meta.get("class_id")
        main_cls = id_to_idx.get(class_id, 0) if class_id is not None else 0

        for cls_idx in range(num_classes):
            gt_mask   = (gt_labels   == cls_idx)
            pred_mask = (pred_labels == cls_idx)

            gt_cls   = gt_boxes[gt_mask]
            pred_cls = pred_boxes[pred_mask]
            scr_cls  = pred_scores[pred_mask]

            keep     = scr_cls >= score_thr
            pred_cls = pred_cls[keep]
            scr_cls  = scr_cls[keep]

            n_gt   = int(gt_mask.sum())
            n_pred = len(pred_cls)
            per_cls_gt[cls_idx] += n_gt

            if n_gt == 0 and n_pred == 0:
                continue

            pred_is_tp, gt_matched = match_preds_to_gt(
                pred_cls, scr_cls, gt_cls, iou_thr)
            tp = sum(pred_is_tp)
            fp = n_pred - tp
            fn = n_gt - sum(gt_matched)
            per_cls_tp[cls_idx] += tp
            per_cls_fp[cls_idx] += fp
            per_cls_fn[cls_idx] += fn

            # Score-TP para curvas P/R
            for score, is_tp in zip(scr_cls.tolist(), pred_is_tp):
                cls_score_tp[cls_idx].append((score, is_tp))
                if cls_idx == main_cls:
                    (all_scores_tp if is_tp else all_scores_fp).append(score)

        # ── Confusion: para cada pred, ¿coincide con algún GT? ────────────────
        for pi, (pb, ps) in enumerate(zip(pred_boxes, pred_scores)):
            if float(ps) < score_thr:
                continue
            pl = int(pred_labels[pi])
            if len(gt_boxes) > 0:
                iou_row = box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
                if float(iou_row.max()) >= iou_thr:
                    best_gt_cls = int(gt_labels[int(iou_row.argmax())])
                    if pl < num_classes and best_gt_cls < num_classes:
                        confusion[best_gt_cls, pl] += 1

        # ── Guardar imagen anotada ─────────────────────────────────────────────
        if save_images and imgs_saved < max_det_imgs and class_id is not None:
            cls_name = id_to_name.get(class_id, "unknown")
            cls_idx  = id_to_idx.get(class_id, 0)

            cls_mask    = (pred_labels == cls_idx)
            cls_boxes   = pred_boxes[cls_mask]
            cls_scores  = pred_scores[cls_mask]
            pred_is_tp, gt_matched = match_preds_to_gt(
                cls_boxes, cls_scores, gt_boxes, iou_thr)

            page_path   = os.path.join(image_root, meta.get("page_path", ""))
            query_paths = ds.query_pool.get(class_id, [])
            query_path  = (os.path.join(image_root, query_paths[0])
                           if query_paths else None)

            try:
                _, page_pil, _ = preprocess_page(
                    page_path, C.DATASET["min_size"], C.DATASET["max_size"])
                query_pil = (preprocess_query(query_path,
                             C.QUERY_ENCODER.get("size", 224))[1]
                             if query_path else Image.new("RGB", (224, 224), "#888"))

                out_img = draw_evaluation_image(
                    page_pil, query_pil,
                    cls_boxes, cls_scores, pred_is_tp,
                    gt_boxes, gt_matched,
                    cls_name, score_thr, str(sid),
                )
                page_stem = Path(meta.get("page_path", f"s{sid}")).stem
                fname     = f"{i_batch:04d}_{cls_name}_{page_stem}.jpg"
                out_img.save(det_path / fname, quality=88)
                imgs_saved += 1
            except Exception as e:
                print(f"    [!] Imagen {sid}: {e}")

        if i_batch % 20 == 0 or i_batch == total:
            print(f"    {i_batch}/{total} procesadas...", end="\r")

    print()

    # ── mAP ───────────────────────────────────────────────────────────────────
    metrics = compute_map(all_preds, all_targets,
                          num_classes=num_classes, iou_threshold=iou_thr)

    # ── Reporte de texto ───────────────────────────────────────────────────────
    report_path = generate_text_report(
        metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
        idx_to_name, checkpoint, split, len(ds),
        iou_thr, score_thr, out_path,
    )
    print(f"  Reporte    → {report_path}")

    # ── Gráficas ───────────────────────────────────────────────────────────────
    if HAS_MPL:
        n_plots = generate_all_plots(
            metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
            idx_to_name, cls_score_tp,
            all_scores_tp, all_scores_fp,
            confusion, plot_path,
        )
        print(f"  Gráficas   → {plot_path}  ({n_plots} archivos)")

    # ── Kit oficial ────────────────────────────────────────────────────────────
    im_path, ps_path = export_kit_files(
        all_preds, all_targets, all_sids,
        sample_meta, id_to_idx, image_root, output_dir,
        score_thr, all_shapes,
    )

    # ── Ejecutar binario oficial automáticamente ───────────────────────────────
    kit_results = {}
    if kit_dir:
        # Pre-cargar valores que run_official_kit necesita para las visualizaciones
        _pre = {"_image_root": image_root, "_max_images": max_det_imgs}
        kr = run_official_kit(
            kit_bin_dir   = kit_dir,
            kit_files_dir = Path(output_dir) / "kit",
            iou_thr       = iou_thr,
        )
        kit_results = {**_pre, **kr}
        # Añadir resultados oficiales al reporte
        _append_kit_results_to_report(
            out_path / "report.txt", kit_results, iou_thr
        )

    # ── Print final ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  ✓  mAP interno  @{iou_thr:.2f} = {metrics['mAP']:.4f}")
    if kit_results.get("im_map") is not None:
        print(f"  ✓  mAP oficial  IM           = {kit_results['im_map']:.4f}")
    if kit_results.get("ps_map") is not None:
        print(f"  ✓  mAP oficial  PS @{iou_thr:.2f}  = {kit_results['ps_map']:.4f}")
    print(f"  {'─'*50}\n")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# REPORTE DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════

def _append_kit_results_to_report(report_path: Path, kit_results: dict,
                                   iou_thr: float):
    """Añade los resultados oficiales del kit al final del report.txt."""
    if not kit_results or not report_path.exists():
        return

    W = 80
    lines = [
        "",
        f"{'── RESULTADOS OFICIALES DEL EVALUATION KIT ':─<{W}}",
        "",
    ]

    im_map = kit_results.get("im_map")
    ps_map = kit_results.get("ps_map")
    im_out = kit_results.get("im_out_path")
    ps_out = kit_results.get("ps_out_path")

    if im_map is not None:
        lines.append(f"  Image Retrieval  mAP (oficial)        :  {im_map:.4f}")
    else:
        lines.append("  Image Retrieval  mAP (oficial)        :  N/A  (error o no ejecutado)")

    if ps_map is not None:
        lines.append(f"  Pattern Spotting mAP (oficial, IoU={iou_thr:.2f}):  {ps_map:.4f}")
    else:
        lines.append(f"  Pattern Spotting mAP (oficial, IoU={iou_thr:.2f}):  N/A  (error o no ejecutado)")

    lines += [
        "",
        f"  Resultado IM → {im_out}",
        f"  Resultado PS → {ps_out}",
        "",
        "  Nota: el mAP oficial puede diferir del mAP interno porque el kit",
        "  evalúa contra el ground truth completo del dataset (incluyendo",
        "  imágenes fuera del split actual).",
        "",
        "─" * W,
    ]

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_text_report(
    metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
    idx_to_name, checkpoint, split, n_samples,
    iou_thr, score_thr, out_path,
) -> Path:
    W = 80
    lines = []

    def sep(char="─"):
        lines.append(char * W)

    def section(title):
        lines.append("")
        lines.append(f"{'── ' + title + ' ':─<{W}}")

    # Cabecera
    lines.append("╔" + "═" * (W - 2) + "╗")
    lines.append("║" + "  iDoc-FCOS — Reporte de Evaluación Completo".center(W - 2) + "║")
    lines.append("╚" + "═" * (W - 2) + "╝")
    lines.append("")
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    lines += [
        f"  Fecha         : {now}",
        f"  Checkpoint    : {checkpoint}",
        f"  Split         : {split}  ({n_samples} muestras)",
        f"  IoU threshold : {iou_thr:.2f}",
        f"  Score thresh  : {score_thr:.2f}  (para stats TP/FP/FN)",
    ]

    # ── Métricas globales ──────────────────────────────────────────────────────
    section("MÉTRICAS GLOBALES")
    total_gt = sum(per_cls_gt.values())
    total_tp = sum(per_cls_tp.values())
    total_fp = sum(per_cls_fp.values())
    total_fn = sum(per_cls_fn.values())
    p_agg    = total_tp / max(total_tp + total_fp, 1)
    r_agg    = total_tp / max(total_tp + total_fn, 1)
    f1_agg   = 2 * p_agg * r_agg / max(p_agg + r_agg, 1e-9)
    lines += [
        f"  mAP @ IoU={iou_thr:.2f}   :  {metrics['mAP']:.4f}",
        f"  Precision global :  {p_agg:.4f}",
        f"  Recall global    :  {r_agg:.4f}",
        f"  F1 global        :  {f1_agg:.4f}",
        "",
        f"  Total GT boxes   :  {total_gt:,}",
        f"  Total TP         :  {total_tp:,}  ({100*total_tp/max(total_gt,1):.1f}% recall)",
        f"  Total FP         :  {total_fp:,}  ({100*total_fp/max(total_tp+total_fp,1):.1f}% FP rate)",
        f"  Total FN (missed):  {total_fn:,}",
    ]

    # ── Tabla por clase ────────────────────────────────────────────────────────
    section("AP Y ESTADÍSTICAS POR CLASE")
    hdr = (f"  {'Rk':>3}  {'Clase':<20}  {'AP':>6}  {'GT':>5}  "
           f"{'TP':>5}  {'FP':>5}  {'FN':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    lines.append(hdr)
    sep_row = "  " + "─" * (len(hdr) - 2)
    lines.append(sep_row)

    class_aps = [(i, metrics["per_class"].get(i, 0.0))
                 for i in range(len(idx_to_name))]
    class_aps.sort(key=lambda x: -x[1])

    for rank, (idx, ap) in enumerate(class_aps, 1):
        name = idx_to_name[idx] if idx < len(idx_to_name) else str(idx)
        gt   = per_cls_gt.get(idx, 0)
        tp   = per_cls_tp.get(idx, 0)
        fp   = per_cls_fp.get(idx, 0)
        fn   = per_cls_fn.get(idx, 0)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        flag = "  " if gt > 0 else " ·"
        lines.append(
            f"{flag}{rank:>3}  {name:<20}  {ap:>6.3f}  {gt:>5}  "
            f"{tp:>5}  {fp:>5}  {fn:>5}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}"
        )

    # ── Highlights ────────────────────────────────────────────────────────────
    section("HIGHLIGHTS")
    cls_with_gt = [(i, ap) for i, ap in class_aps if per_cls_gt.get(i, 0) > 0]
    if cls_with_gt:
        best_i,  best_ap  = cls_with_gt[0]
        worst_i, worst_ap = cls_with_gt[-1]
        lines += [
            f"  Mejor clase    : {idx_to_name[best_i]}  (AP={best_ap:.4f})",
            f"  Peor clase     : {idx_to_name[worst_i]}  (AP={worst_ap:.4f})",
        ]
    n_above50 = sum(1 for _, ap in class_aps if ap >= 0.5)
    n_zero    = sum(1 for i, ap in class_aps
                    if ap == 0.0 and per_cls_gt.get(i, 0) > 0)
    n_novgt   = sum(1 for i, _ in class_aps if per_cls_gt.get(i, 0) == 0)
    lines += [
        f"  Clases AP≥0.50 : {n_above50} / {len(class_aps)}",
        f"  Clases AP=0.00 : {n_zero}  (con GT en el split)",
        f"  Clases sin GT  : {n_novgt}  (no evaluadas)",
    ]

    # ── Distribución de scores ─────────────────────────────────────────────────
    section("ANÁLISIS DE SCORES (score_thr aplicado)")
    lines += [
        f"  Predicciones mostradas (score≥{score_thr:.2f}) : {total_tp + total_fp:,}",
        f"    TP : {total_tp:,}  ({100*total_tp/max(total_tp+total_fp,1):.1f}%)",
        f"    FP : {total_fp:,}  ({100*total_fp/max(total_tp+total_fp,1):.1f}%)",
    ]

    # ── Instrucciones para el kit ──────────────────────────────────────────────
    section("USO DEL EVALUATION KIT OFICIAL")
    lines += [
        "  Archivos generados en  eval_output/kit/",
        "",
        "  # Image Retrieval:",
        "  ./main --task im \\",
        "      --in_file eval_output/kit/kit_im.txt \\",
        "      --out_file eval_output/kit/result_im.txt",
        "",
        "  # Pattern Spotting:",
        "  ./main --task ps --iou 0.5 \\",
        "      --in_file eval_output/kit/kit_ps.txt \\",
        "      --out_file eval_output/kit/result_ps.txt",
    ]

    lines.append("")
    lines.append("─" * W)

    report_path = out_path / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_style():
    plt.rcParams.update(MPL_RC)


def generate_all_plots(
    metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
    idx_to_name, cls_score_tp,
    all_scores_tp, all_scores_fp,
    confusion, plot_path,
) -> int:
    _apply_style()
    n = 0

    # Clases con GT en el split
    cls_with_gt = [(i, idx_to_name[i])
                   for i in range(len(idx_to_name))
                   if per_cls_gt.get(i, 0) > 0]

    print(f"  Generando gráficas en {plot_path} ...")

    fns = [
        ("01_ap_per_class",
         lambda: _plot_ap_per_class(metrics, per_cls_gt, idx_to_name, plot_path)),
        ("02_precision_recall_f1",
         lambda: _plot_prf_per_class(per_cls_tp, per_cls_fp, per_cls_fn,
                                     cls_with_gt, plot_path)),
        ("03_tp_fp_fn",
         lambda: _plot_tp_fp_fn(per_cls_tp, per_cls_fp, per_cls_fn,
                                cls_with_gt, plot_path)),
        ("04_summary_dashboard",
         lambda: _plot_summary_dashboard(metrics, per_cls_gt, per_cls_tp,
                                         per_cls_fp, per_cls_fn, idx_to_name,
                                         plot_path)),
        ("05_score_distribution",
         lambda: _plot_score_distribution(all_scores_tp, all_scores_fp,
                                          plot_path)),
        ("06_pr_curves",
         lambda: _plot_pr_curves(cls_score_tp, per_cls_gt, idx_to_name,
                                 metrics, plot_path)),
        ("07_confusion_matrix",
         lambda: _plot_confusion_matrix(confusion, idx_to_name, plot_path)),
        ("08_worst_best_grid",
         lambda: _plot_worst_best_grid(metrics, per_cls_gt, idx_to_name,
                                       plot_path)),
    ]

    for name, fn in fns:
        try:
            result = fn()
            n += result if result else 0
            print(f"    ✓ {name}.png")
        except Exception as e:
            print(f"    ✗ {name}: {e}")

    return n


# ── 01: AP por clase ──────────────────────────────────────────────────────────

def _plot_ap_per_class(metrics, per_cls_gt, idx_to_name, plot_path) -> int:
    aps = [(i, metrics["per_class"].get(i, 0.0), idx_to_name[i])
           for i in range(len(idx_to_name))
           if per_cls_gt.get(i, 0) > 0]
    aps.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(11, max(5, len(aps) * 0.38)))
    vals    = [x[1] for x in aps]
    names   = [x[2] for x in aps]
    colors  = [plt.cm.RdYlGn(v) for v in vals]

    bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.72)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=8.5)

    mAP = metrics["mAP"]
    ax.axvline(mAP, color="#2C3E50", ls="--", lw=1.5,
               label=f"mAP = {mAP:.3f}")
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("Average Precision @ IoU=0.5")
    ax.set_title("AP por clase")
    ax.legend(loc="lower right", fontsize=9)

    # Color bar
    sm = ScalarMappable(cmap="RdYlGn", norm=Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="AP", fraction=0.02, pad=0.01)

    plt.tight_layout()
    fig.savefig(plot_path / "01_ap_per_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 02: Precision / Recall / F1 por clase ────────────────────────────────────

def _plot_prf_per_class(per_cls_tp, per_cls_fp, per_cls_fn,
                         cls_with_gt, plot_path) -> int:
    cls_with_gt_s = sorted(cls_with_gt, key=lambda x: x[0])
    idxs, names   = zip(*cls_with_gt_s) if cls_with_gt_s else ([], [])

    precs, recs, f1s = [], [], []
    for idx in idxs:
        tp = per_cls_tp.get(idx, 0)
        fp = per_cls_fp.get(idx, 0)
        fn = per_cls_fn.get(idx, 0)
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        precs.append(pr); recs.append(rc); f1s.append(f1)

    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.54), 5))
    ax.bar(x - w, precs, w, label="Precision", color=PLT_BLUE,   alpha=0.88)
    ax.bar(x,     recs,  w, label="Recall",    color=PLT_GREEN,  alpha=0.88)
    ax.bar(x + w, f1s,   w, label="F1",        color=PLT_RED,    alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 por clase")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(plot_path / "02_precision_recall_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 03: TP / FP / FN stacked ──────────────────────────────────────────────────

def _plot_tp_fp_fn(per_cls_tp, per_cls_fp, per_cls_fn,
                   cls_with_gt, plot_path) -> int:
    cls_s = sorted(cls_with_gt, key=lambda x: -(per_cls_tp.get(x[0], 0) +
                                                  per_cls_fn.get(x[0], 0)))
    if not cls_s:
        return 0
    idxs, names = zip(*cls_s)

    tps = [per_cls_tp.get(i, 0) for i in idxs]
    fps = [per_cls_fp.get(i, 0) for i in idxs]
    fns = [per_cls_fn.get(i, 0) for i in idxs]

    x   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.54), 5))
    ax.bar(x, tps, label="TP",        color=PLT_GREEN,  alpha=0.90)
    ax.bar(x, fps, bottom=tps, label="FP", color=PLT_RED, alpha=0.85)
    ax.bar(x, fns, bottom=[t + f for t, f in zip(tps, fps)],
           label="FN (missed)", color=PLT_ORANGE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Número de detecciones")
    ax.set_title("TP / FP / FN por clase")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(plot_path / "03_tp_fp_fn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 04: Dashboard 2×2 ─────────────────────────────────────────────────────────

def _plot_summary_dashboard(metrics, per_cls_gt, per_cls_tp, per_cls_fp,
                             per_cls_fn, idx_to_name, plot_path) -> int:
    total_gt = sum(per_cls_gt.values())
    total_tp = sum(per_cls_tp.values())
    total_fp = sum(per_cls_fp.values())
    total_fn = sum(per_cls_fn.values())
    p_agg    = total_tp / max(total_tp + total_fp, 1)
    r_agg    = total_tp / max(total_tp + total_fn, 1)
    f1_agg   = 2 * p_agg * r_agg / max(p_agg + r_agg, 1e-9)
    mAP      = metrics["mAP"]

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ─ Panel 1: métricas globales como texto estilizado ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    kv = [
        ("mAP @ IoU=0.5", f"{mAP:.4f}"),
        ("Precision",     f"{p_agg:.4f}"),
        ("Recall",        f"{r_agg:.4f}"),
        ("F1",            f"{f1_agg:.4f}"),
        ("GT boxes",      f"{total_gt:,}"),
        ("TP",            f"{total_tp:,}  ({100*total_tp/max(total_gt,1):.1f}%)"),
        ("FP",            f"{total_fp:,}"),
        ("FN",            f"{total_fn:,}"),
    ]
    for row, (k, v) in enumerate(kv):
        c = "#2C3E50" if row < 4 else "#566573"
        ax1.text(0.05, 1 - (row + 0.5) / len(kv), k,
                 transform=ax1.transAxes, ha="left", va="center",
                 fontsize=11, color=c, fontweight="bold")
        ax1.text(0.95, 1 - (row + 0.5) / len(kv), v,
                 transform=ax1.transAxes, ha="right", va="center",
                 fontsize=11, color=c, fontfamily="monospace")
    ax1.set_facecolor("#EBF5FB")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#2980B9")
        sp.set_linewidth(1.5)
    ax1.set_title("Métricas globales", fontsize=12, fontweight="bold", pad=8)

    # ─ Panel 2: pie TP / FP / FN ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    pie_vals   = [total_tp, total_fp, total_fn]
    pie_labels = [f"TP\n{total_tp:,}", f"FP\n{total_fp:,}", f"FN\n{total_fn:,}"]
    pie_colors = [PLT_GREEN, PLT_RED, PLT_ORANGE]
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Distribución TP / FP / FN", fontsize=12, fontweight="bold")

    # ─ Panel 3: histograma AP ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ap_vals = [metrics["per_class"].get(i, 0.0)
               for i in range(len(idx_to_name))
               if per_cls_gt.get(i, 0) > 0]
    bins = np.linspace(0, 1, 11)
    counts, _, patches = ax3.hist(ap_vals, bins=bins, edgecolor="white", lw=0.8)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.RdYlGn(left))
    ax3.axvline(mAP, color="#2C3E50", ls="--", lw=1.5, label=f"mAP={mAP:.3f}")
    ax3.set_xlabel("AP")
    ax3.set_ylabel("N.° clases")
    ax3.set_title("Distribución de AP por clase")
    ax3.legend(fontsize=9)

    # ─ Panel 4: Precision vs Recall scatter por clase ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    precs_s, recs_s, aps_s, names_s = [], [], [], []
    for i in range(len(idx_to_name)):
        if per_cls_gt.get(i, 0) == 0:
            continue
        tp = per_cls_tp.get(i, 0)
        fp = per_cls_fp.get(i, 0)
        fn = per_cls_fn.get(i, 0)
        precs_s.append(tp / max(tp + fp, 1))
        recs_s.append(tp / max(tp + fn, 1))
        aps_s.append(metrics["per_class"].get(i, 0.0))
        names_s.append(idx_to_name[i])

    sc = ax4.scatter(recs_s, precs_s, c=aps_s, cmap="RdYlGn",
                     s=60, alpha=0.85, vmin=0, vmax=1,
                     edgecolors="white", linewidths=0.5)
    fig.colorbar(sc, ax=ax4, label="AP", fraction=0.04, pad=0.02)
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_title("Precision vs Recall por clase (color = AP)")
    ax4.plot([0, 1], [1, 0], color="#BDC3C7", ls="--", lw=0.8)

    fig.suptitle("iDoc-FCOS — Dashboard de Evaluación",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(plot_path / "04_summary_dashboard.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 05: Distribución de scores ────────────────────────────────────────────────

def _plot_score_distribution(all_scores_tp, all_scores_fp, plot_path) -> int:
    if not all_scores_tp and not all_scores_fp:
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Izquierda: histogramas superpuestos
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    if all_scores_tp:
        ax.hist(all_scores_tp, bins=bins, alpha=0.75,
                color=PLT_BLUE,  label=f"TP  (n={len(all_scores_tp):,})",
                edgecolor="white", lw=0.5)
    if all_scores_fp:
        ax.hist(all_scores_fp, bins=bins, alpha=0.75,
                color=PLT_RED,   label=f"FP  (n={len(all_scores_fp):,})",
                edgecolor="white", lw=0.5)
    ax.set_xlabel("Score de confianza")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de scores TP vs FP")
    ax.legend(fontsize=9)

    # Derecha: precision-score curve (precision en función del threshold)
    ax2 = axes[1]
    thresholds = np.linspace(0.05, 0.95, 50)
    all_s  = sorted(all_scores_tp + all_scores_fp, reverse=True)
    all_tp_arr = [1 if s in set(all_scores_tp) else 0 for s in all_s]

    prec_at_thr, rec_at_thr = [], []
    total_pos = len(all_scores_tp)
    for thr in thresholds:
        tp_cnt = sum(1 for s in all_scores_tp if s >= thr)
        fp_cnt = sum(1 for s in all_scores_fp if s >= thr)
        pr = tp_cnt / max(tp_cnt + fp_cnt, 1)
        rc = tp_cnt / max(total_pos, 1)
        prec_at_thr.append(pr)
        rec_at_thr.append(rc)

    ax2.plot(thresholds, prec_at_thr, color=PLT_BLUE,  lw=2, label="Precision")
    ax2.plot(thresholds, rec_at_thr,  color=PLT_GREEN, lw=2, label="Recall")
    ax2.axvline(0.25, color=PLT_RED, ls="--", lw=1.2, label="score_thr=0.25")
    ax2.set_xlabel("Score threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision y Recall vs Score threshold")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(plot_path / "05_score_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 06: Curvas P/R por clase ──────────────────────────────────────────────────

def _pr_curve(score_tp_list: list, n_gt: int):
    """Calcula curva P/R y AP a partir de [(score, is_tp), ...]."""
    if not score_tp_list or n_gt == 0:
        return [0], [0], 0.0
    sorted_list = sorted(score_tp_list, key=lambda x: -x[0])
    precs, recs = [], []
    tp_cum = 0
    for rank, (_, is_tp) in enumerate(sorted_list, 1):
        if is_tp:
            tp_cum += 1
        precs.append(tp_cum / rank)
        recs.append(tp_cum / n_gt)
    # Interpolated AP
    precs_arr = np.array(precs)
    recs_arr  = np.array(recs)
    ap = float(np.trapz(precs_arr, recs_arr)) if len(recs_arr) > 1 else 0.0
    return precs_arr.tolist(), recs_arr.tolist(), abs(ap)


def _plot_pr_curves(cls_score_tp, per_cls_gt, idx_to_name, metrics,
                    plot_path) -> int:
    # Solo clases con GT y con al menos una predicción
    eligible = [(i, idx_to_name[i])
                for i in range(len(idx_to_name))
                if per_cls_gt.get(i, 0) > 0 and cls_score_tp.get(i)]
    if not eligible:
        return 0

    # Ordenar por AP descendente, mostrar top 20
    eligible.sort(key=lambda x: -metrics["per_class"].get(x[0], 0.0))
    top = eligible[:20]

    cols = 4
    rows = math.ceil(len(top) / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.8, rows * 3.2),
                             facecolor="white")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    cmap = plt.cm.tab20
    for ax_i, (cls_idx, cls_name) in enumerate(top):
        ax  = axes_flat[ax_i]
        pr, rc, ap_local = _pr_curve(cls_score_tp[cls_idx],
                                      per_cls_gt[cls_idx])
        ap_official = metrics["per_class"].get(cls_idx, 0.0)
        color = cmap(ax_i / max(len(top) - 1, 1))
        ax.plot(rc, pr, color=color, lw=1.8)
        ax.fill_between(rc, pr, alpha=0.12, color=color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(f"{cls_name}\nAP={ap_official:.3f}",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=7.5)
        ax.set_ylabel("Precision", fontsize=7.5)
        ax.tick_params(labelsize=7)

    for ax_j in range(len(top), len(axes_flat)):
        axes_flat[ax_j].axis("off")

    fig.suptitle("Curvas Precision-Recall por clase (top 20 por AP)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path / "06_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 07: Confusion matrix ──────────────────────────────────────────────────────

def _plot_confusion_matrix(confusion, idx_to_name, plot_path) -> int:
    nc = confusion.shape[0]
    # Filtrar filas/columnas con actividad
    active = [i for i in range(nc)
              if confusion[i].sum() > 0 or confusion[:, i].sum() > 0]
    if len(active) < 2:
        return 0

    sub  = confusion[np.ix_(active, active)]
    # Normalizar por fila (recall)
    row_sum = sub.sum(axis=1, keepdims=True).clip(min=1)
    norm    = sub / row_sum

    n_act = len(active)
    sz    = max(6, min(18, n_act * 0.5))
    fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Recall normalizado", fraction=0.04, pad=0.02)

    ticks = list(range(n_act))
    names = [idx_to_name[active[i]] if active[i] < len(idx_to_name) else str(active[i])
             for i in range(n_act)]

    ax.set_xticks(ticks); ax.set_yticks(ticks)
    if n_act <= 20:
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real (GT)")
    ax.set_title("Matriz de confusión (normalizada por fila)")
    plt.tight_layout()
    fig.savefig(plot_path / "07_confusion_matrix.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 08: Grid mejores / peores clases ─────────────────────────────────────────

def _plot_worst_best_grid(metrics, per_cls_gt, idx_to_name, plot_path) -> int:
    cls_aps = [(i, metrics["per_class"].get(i, 0.0))
               for i in range(len(idx_to_name))
               if per_cls_gt.get(i, 0) > 0]
    if not cls_aps:
        return 0
    cls_aps.sort(key=lambda x: -x[1])

    top_n = min(10, len(cls_aps))
    best  = cls_aps[:top_n]
    worst = cls_aps[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, title, group, palette in [
        (axes[0], f"Top {top_n} mejores clases",  best,  "Greens"),
        (axes[1], f"Top {top_n} peores clases",   worst, "Reds"),
    ]:
        names = [idx_to_name[i] for i, _ in group]
        vals  = [ap for _, ap in group]
        cmap  = plt.get_cmap(palette)
        colors = [cmap(0.35 + 0.55 * (v / max(max(vals), 1e-6))) for v in vals]

        bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("AP")
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plot_path / "08_worst_best_grid.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 1


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN RÁPIDA (backward compat)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_set_simple(model, image_root, device):
    cfg = {
        "BACKBONE":     C.BACKBONE,  "FPN":           C.FPN,
        "FCOS_HEAD":    C.FCOS_HEAD, "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET":      C.DATASET,   "AUGMENTATION":  C.AUGMENTATION,
        "EVAL":         C.EVAL,
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nCargando modelo...")
    model = load_model(args.checkpoint, device)

    # ── Single inference ───────────────────────────────────────────────────────
    if args.page_img and args.query_img:
        print(f"\nInferencia: {args.page_img}  +  query: {args.query_img}")
        page_t, page_pil, img_shape = preprocess_page(
            args.page_img, C.DATASET["min_size"], C.DATASET["max_size"])
        query_t, query_pil = preprocess_query(
            args.query_img, C.QUERY_ENCODER.get("size", 224))

        dets = model.predict(page_t.to(device), query_t.to(device), img_shape)
        det  = dets[0]

        pred_is_tp = [False] * len(det["boxes"])
        out_img = draw_evaluation_image(
            page_pil, query_pil,
            det["boxes"], det["scores"], pred_is_tp,
            torch.zeros((0, 4)), [],
            "query", args.score_thr,
        )
        out_img.save(args.output_img)
        print(f"  {len(det['boxes'])} detecciones → {args.output_img}")
        return

    # ── Evaluación completa ────────────────────────────────────────────────────
    if not args.image_root:
        print("Especifica --image_root para evaluar, o --page_img + "
              "--query_img para inferencia single.")
        return

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
        max_det_imgs = args.max_det_imgs,
        kit_dir      = args.kit_dir,
        device       = device,
    )


if __name__ == "__main__":
    main()