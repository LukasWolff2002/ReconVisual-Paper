"""
evaluate.py  —  iDoc-FCOS  |  Evaluación completa

Genera automáticamente:
  eval_output/
  ├── report.txt                  — reporte de texto completo con todas las métricas
  ├── detections/                 — imágenes anotadas por muestra (TP/FP/FN/GT)
  ├── examples/                   — mejores y peores casos generados automáticamente
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

 python3 -m train_fcos.compare_backbones \
    --checkpoint_idoc train_fcos/run_01/best_model.pth \
    --pretrained_idoc /home/rvdl/rvdl_2/idoc_pretrained.pth \
    --checkpoint_dino train_fcos/run_02/best_model.pth \
    --pretrained_dino /home/rvdl/rvdl_2/dinov3_vitb16.pth \
    --image_root /home/rvdl/rvdl_2/ \
    --dataset_json /home/rvdl/rvdl_2/detection_dataset_sketches.json \
    --output_dir eval_comparison/ \
    --split val \
    --skip_full_kit

NOTA IMPORTANTE — --pretrained:
  Debe ser el MISMO archivo de pretrain usado durante el ENTRENAMIENTO del
  checkpoint que estás evaluando (idoc_pretrained.pth o dinov3_vitb16.pth).
  iDocBackbone auto-detecta el tipo de arquitectura (iDoc con pos_embed vs
  DINOv3 con RoPE + register tokens) a partir del CONTENIDO de este archivo,
  y construye el ViT correspondiente ANTES de cargar los pesos fine-tuned
  del checkpoint. Si pasas el archivo equivocado (o None), la arquitectura
  construida no coincidirá con la guardada en el checkpoint y el backbone
  quedará con pesos aleatorios sin que el script lo reporte como error
  (model.load_state_dict usa strict=False).
"""

import os
import sys
import json
import math
import time
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
# PALETA Y CONSTANTES VISUALES (ESTILO PAPER)
# ══════════════════════════════════════════════════════════════════════════════

# Colores para anotaciones en imágenes
COL_TP_PRED = "#005EB8"   # Azul cobalto — predicción correcta (TP)
COL_FP_PRED = "#D11141"   # Carmesí — predicción falsa   (FP)
COL_TP_GT   = "#00B159"   # Verde esmeralda — GT matcheado
COL_FN_GT   = "#FFC425"   # Amarillo dorado — GT perdido (FN) (usar dashed)
COL_TEXT    = "#000000"   # Negro absoluto para textos
COL_BG_FILL = "#FFFFFF"   # Blanco absoluto para fondo de etiquetas

# Líneas y Textos
BOX_THICKNESS = 2         # Grosor de línea de cuadro
TEXT_SIZE_MD  = 12        # Tamaño de texto base

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

def _load_serif_font(size: int = TEXT_SIZE_MD, bold: bool = True) -> ImageFont.ImageFont:
    """Intenta cargar una fuente Serif (DejaVu Serif) si está disponible."""
    variants = []
    if bold:
        variants = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        ]
    else:
        variants = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif.ttf",
        ]
        
    for path in variants:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    print("  ⚠ No se encontró fuente Serif ttf. Usando predeterminada.")
    return ImageFont.load_default()

FONT_SERIF_LG = _load_serif_font(15)
FONT_SERIF_MD = _load_serif_font(TEXT_SIZE_MD)
FONT_SERIF_SM = _load_serif_font(10)

def _text_size(font, text: str):
    """Fallback robusto para obtener el tamaño del texto."""
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return len(text) * 7, 12


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Evaluación completa iDoc-FCOS")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--pretrained",    type=str, required=True,
                   help="Ruta al .pth de pretrain usado en el ENTRENAMIENTO "
                        "de este checkpoint.")
    p.add_argument("--image_root",    type=str, default=None)
    p.add_argument("--dataset_json",  type=str, default=C.DATASET_JSON)
    p.add_argument("--output_dir",    type=str, default="eval_output")
    p.add_argument("--split",         type=str, default="val",
                   choices=["val", "test"])
    p.add_argument("--score_thr",     type=float, default=0.25)
    p.add_argument("--iou_thr",       type=float, default=0.5)
    p.add_argument("--save_images",   action="store_true", default=True)
    p.add_argument("--no_images",     action="store_true")
    p.add_argument("--max_det_imgs",  type=int, default=200)
    p.add_argument("--kit_dir",       type=str, default=None)
    p.add_argument("--skip_full_kit", action="store_true")
    p.add_argument("--pages_subdir",  type=str, default="DocExplore_images")
    p.add_argument("--max_pages",     type=int, default=None)
    p.add_argument("--iou_sweep",     type=str, default="0.3,0.5,0.7")
    p.add_argument("--subset_pages",  type=int, default=None)
    p.add_argument("--subset_seed",   type=int, default=42)
    p.add_argument("--gpu_ids",       type=str, default=None)
    # Single inference
    p.add_argument("--page_img",      type=str, default=None)
    p.add_argument("--query_img",     type=str, default=None)
    p.add_argument("--output_img",    type=str, default="result.jpg")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, pretrained_path: str,
               device: torch.device) -> FCOSDetector:
    cfg = {
        "BACKBONE":       C.BACKBONE,
        "FPN":            C.FPN,
        "FCOS_HEAD":      C.FCOS_HEAD,
        "QUERY_ENCODER":  C.QUERY_ENCODER,
        "EVAL":           C.EVAL,
        "PRETRAINED_PTH": pretrained_path,
    }
    model = FCOSDetector(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    msg = model.load_state_dict(state, strict=False)

    backbone_missing = [k for k in msg.missing_keys if "backbone.vit" in k]
    backbone_unexpected = [k for k in msg.unexpected_keys if "backbone.vit" in k]
    if backbone_missing or backbone_unexpected:
        print(f"  ⚠ ADVERTENCIA: {len(backbone_missing)} keys de backbone "
              f"faltantes y {len(backbone_unexpected)} inesperadas.")

    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Backbone   : {pretrained_path}")
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
# DIBUJO DE IMÁGENES ANOTADAS (ESTILO PAPER MINIMALISTA)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_dashed_line(draw, x1, y1, x2, y2, fill, width=BOX_THICKNESS, dash=(5, 3)):
    """Dibuja una línea punteada entre dos puntos."""
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    if dist < 1: return
    
    dash_len = dash[0]; gap_len = dash[1]
    dx = (x2-x1)/dist; dy = (y2-y1)/dist
    
    pos_x, pos_y = x1, y1
    step = 0
    while step < dist:
        end_step = min(step + dash_len, dist)
        next_x = x1 + dx * end_step
        next_y = y1 + dy * end_step
        draw.line([(pos_x, pos_y), (next_x, next_y)], fill=fill, width=width)
        pos_x = x1 + dx * (end_step + gap_len)
        pos_y = y1 + dy * (end_step + gap_len)
        step = end_step + gap_len


def _draw_paper_box(draw: ImageDraw.Draw, box, color: str, 
                    fill_text_bg: bool = False,
                    dashed: bool = False, font=None, text: str = ""):
    """Dibuja un cuadro minimalista estilo paper con etiqueta serif."""
    x1, y1, x2, y2 = [int(v) for v in box]
    
    if dashed:
        _draw_dashed_line(draw, x1, y1, x2, y1, fill=color)
        _draw_dashed_line(draw, x2, y1, x2, y2, fill=color)
        _draw_dashed_line(draw, x2, y2, x1, y2, fill=color)
        _draw_dashed_line(draw, x1, y2, x1, y1, fill=color)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_THICKNESS)
    
    if text:
        tw, th = _text_size(font, text)
        pad = 2
        tx = x1 + BOX_THICKNESS + pad
        ty = y1 + BOX_THICKNESS + pad
        draw.rectangle([tx-pad, ty-pad, tx+tw+pad, ty+th+pad], fill=COL_BG_FILL)
        draw.text((tx, ty), text, fill=COL_TEXT, font=font)


def _draw_paper_legend(draw, x_start, y_start):
    """Dibuja una leyenda minimalista de texto directo sobre la imagen."""
    items = [
        (COL_TP_PRED, "Pred TP", False),
        (COL_FP_PRED, "Pred FP", False),
        (COL_TP_GT,   "GT Matched", False),
        (COL_FN_GT,   "GT Missed (FN)", True),
    ]
    x, y = x_start, y_start
    box_sz = 10
    sep = 6
    row_h = 16

    for color, label, is_dashed in items:
        if is_dashed:
            _draw_dashed_line(draw, x, y+2, x+box_sz, y+2, fill=color, width=max(1, BOX_THICKNESS//2))
            _draw_dashed_line(draw, x+box_sz, y+2, x+box_sz, y+box_sz+2, fill=color, width=max(1, BOX_THICKNESS//2))
            _draw_dashed_line(draw, x+box_sz, y+box_sz+2, x, y+box_sz+2, fill=color, width=max(1, BOX_THICKNESS//2))
            _draw_dashed_line(draw, x, y+box_sz+2, x, y+2, fill=color, width=max(1, BOX_THICKNESS//2))
        else:
            draw.rectangle([x, y+2, x+box_sz, y+box_sz+2], outline=color, width=max(1, BOX_THICKNESS//2))
        
        draw.text((x + box_sz + sep, y), label, fill=COL_TEXT, font=FONT_SERIF_SM)
        y += row_h


def draw_paper_styled_image(
    page_pil:    Image.Image,
    query_pil:   Image.Image,
    pred_boxes:  torch.Tensor,
    pred_scores: torch.Tensor,
    pred_is_tp:  list,
    gt_boxes:    torch.Tensor,
    gt_matched:  list,
    sample_num:  int   = 1,
    score_thr:   float = 0.25,
) -> Image.Image:
    """
    Imagen de inferencia limpia para publicación (estilo paper).
    """
    img  = page_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Identificador superior izquierdo
    #draw.text((10, 10), f"Sample {sample_num:03d}", fill=COL_TEXT, font=FONT_SERIF_LG)

    # Thumbnail query sketch
    thumb_sz = 180
    thumb = query_pil.copy().convert("RGB").resize((thumb_sz, thumb_sz), Image.LANCZOS)
    mx, my = W - thumb_sz - 15, 15
    draw.rectangle([mx-1, my-1, mx+thumb_sz+1, my+thumb_sz+1], outline="#CCCCCC", width=1)
    img.paste(thumb, (mx, my))

    # GT boxes
    for j, box in enumerate(gt_boxes):
        if gt_matched[j]:
            _draw_paper_box(draw, box, COL_TP_GT, font=FONT_SERIF_SM, text="GT")
        else:
            _draw_paper_box(draw, box, COL_FN_GT, dashed=True, font=FONT_SERIF_SM, text="FN")

    # Pred boxes
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        if float(score) < score_thr:
            continue
        color = COL_TP_PRED if pred_is_tp[i] else COL_FP_PRED
        is_tp = "TP" if pred_is_tp[i] else "FP"
        _draw_paper_box(draw, box, color, font=FONT_SERIF_SM, text=f"{is_tp} {score:.2f}")

    # Leyenda inferior derecha
    l_h = 16 * 4
    lx, ly = mx + 5, H - l_h - 20
    #_draw_paper_legend(draw, lx, ly)

    return img


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
        "BACKBONE":     C.BACKBONE,  "FPN":            C.FPN,
        "FCOS_HEAD":    C.FCOS_HEAD, "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET":      C.DATASET,   "AUGMENTATION":  C.AUGMENTATION,
        "EVAL":         C.EVAL,
    }
    train_ds, val_ds, test_ds = build_datasets(C.DATASET_JSON, image_root, cfg)
    ds = val_ds if split == "val" else test_ds
    print(f"  Split '{split}': {len(ds)} muestras")

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=False)

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

    cls_score_tp: dict[int, list] = defaultdict(list)
    per_cls_gt  = defaultdict(int)
    per_cls_tp  = defaultdict(int)
    per_cls_fp  = defaultdict(int)
    per_cls_fn  = defaultdict(int)

    all_scores_tp = []
    all_scores_fp = []
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    # ── EJEMPLOS AUTOMÁTICOS ───────────────────────────────────────────────────
    best_ex     = {"tp": -1, "fp": 999}
    worst_fp_ex = {"fp": -1}
    worst_fn_ex = {"fn": -1}

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

        all_preds.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
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

            pred_is_tp, gt_matched = match_preds_to_gt(pred_cls, scr_cls, gt_cls, iou_thr)
            tp = sum(pred_is_tp)
            fp = n_pred - tp
            fn = n_gt - sum(gt_matched)
            per_cls_tp[cls_idx] += tp
            per_cls_fp[cls_idx] += fp
            per_cls_fn[cls_idx] += fn

            for score, is_tp in zip(scr_cls.tolist(), pred_is_tp):
                cls_score_tp[cls_idx].append((score, is_tp))
                if cls_idx == main_cls:
                    (all_scores_tp if is_tp else all_scores_fp).append(score)

        # ── Confusion ─────────────────────────────────────────────────────────
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

        # ── Guardar imagen anotada (FORMATO PAPER) ─────────────────────────────
        if save_images and imgs_saved < max_det_imgs and class_id is not None:
            cls_idx = id_to_idx.get(class_id, 0)
            
            cls_mask = (pred_labels == cls_idx)
            cb = pred_boxes[cls_mask]
            cs = pred_scores[cls_mask]
            gb = gt_boxes[(gt_labels == cls_idx)]
            pred_is_tp, gt_matched = match_preds_to_gt(cb, cs, gb, iou_thr)

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

                out_img = draw_paper_styled_image(
                    page_pil, query_pil,
                    cb, cs, pred_is_tp,
                    gb, gt_matched,
                    sample_num = i_batch,
                    score_thr = score_thr,
                )
                
                cls_name  = id_to_name.get(class_id, "unknown")
                page_stem = Path(meta.get("page_path", f"s{sid}")).stem
                fname     = f"{i_batch:04d}_{cls_name}_{page_stem}.jpg"
                out_img.save(det_path / fname, quality=88)
                imgs_saved += 1
                
                # --- Guardar meta para ejemplos automáticos ---
                tp_cnt = sum(pred_is_tp)
                fp_cnt = len(cb) - tp_cnt
                fn_cnt = len(gb) - sum(gt_matched)
                
                _sample_data = {
                    "img": out_img, "sid": sid, "cls": cls_name, "page": page_stem,
                    "tp": tp_cnt, "fp": fp_cnt, "fn": fn_cnt,
                }
                
                if tp_cnt > best_ex["tp"] or (tp_cnt == best_ex["tp"] and fp_cnt < best_ex["fp"]):
                    best_ex = _sample_data
                if fp_cnt > worst_fp_ex["fp"]:
                    worst_fp_ex = _sample_data
                if fn_cnt > worst_fn_ex["fn"]:
                    worst_fn_ex = _sample_data

            except Exception as e:
                print(f"    [!] Imagen {sid}: {e}")

        if i_batch % 20 == 0 or i_batch == total:
            print(f"    {i_batch}/{total} procesadas...", end="\r")

    print()

    # ── Guardar Ejemplos Automáticos en Disco ──────────────────────────────────
    example_dir = out_path / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    if "img" in best_ex:
        best_ex["img"].save(example_dir / f"01_best_inference.jpg", quality=90)
    if "img" in worst_fp_ex:
        worst_fp_ex["img"].save(example_dir / f"02_worst_fp.jpg", quality=90)
    if "img" in worst_fn_ex:
        worst_fn_ex["img"].save(example_dir / f"03_worst_fn.jpg", quality=90)

    # ── mAP ───────────────────────────────────────────────────────────────────
    metrics = compute_map(all_preds, all_targets, num_classes=num_classes, iou_threshold=iou_thr)

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

    # ── Kit oficial BYPASS ─────────────────────────────────────────────────────
    kit_results = {}
    print(f"\n  {'─'*50}")
    print(f"  ✓  mAP interno  @{iou_thr:.2f} = {metrics['mAP']:.4f}")
    print(f"  {'─'*50}\n")

    return {
        "metrics":     metrics,
        "per_cls_gt":  dict(per_cls_gt),
        "per_cls_tp":  dict(per_cls_tp),
        "per_cls_fp":  dict(per_cls_fp),
        "per_cls_fn":  dict(per_cls_fn),
        "idx_to_name": idx_to_name,
        "kit_results": kit_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REPORTE DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════

def generate_text_report(
    metrics, per_cls_gt, per_cls_tp, per_cls_fp, per_cls_fn,
    idx_to_name, checkpoint, split, n_samples,
    iou_thr, score_thr, out_path,
) -> Path:
    W = 80
    lines = []
    def sep(char="─"): lines.append(char * W)
    def section(title):
        lines.append("")
        lines.append(f"{'── ' + title + ' ':─<{W}}")

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
        f"  Score thresh  : {score_thr:.2f}",
    ]

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
        f"  Total TP         :  {total_tp:,}",
        f"  Total FP         :  {total_fp:,}",
        f"  Total FN         :  {total_fn:,}",
    ]

    section("AP Y ESTADÍSTICAS POR CLASE")
    hdr = (f"  {'Rk':>3}  {'Clase':<20}  {'AP':>6}  {'GT':>5}  "
           f"{'TP':>5}  {'FP':>5}  {'FN':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    lines.append(hdr)
    lines.append("  " + "─" * (len(hdr) - 2))

    class_aps = [(i, metrics["per_class"].get(i, 0.0)) for i in range(len(idx_to_name))]
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

    lines.append("")
    lines.append("─" * W)

    report_path = out_path / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS (Mantenemos la estructura simplificada para evaluación individual)
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
    return n # Graficas omitidas para ahorrar espacio, ya que usamos el dashboard de comparison_backbones


# ══════════════════════════════════════════════════════════════════════════════
# MAIN 
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nCargando modelo...")
    model = load_model(args.checkpoint, args.pretrained, device)

    if args.page_img and args.query_img:
        print(f"\nInferencia: {args.page_img}  +  query: {args.query_img}")
        page_t, page_pil, img_shape = preprocess_page(
            args.page_img, C.DATASET["min_size"], C.DATASET["max_size"])
        query_t, query_pil = preprocess_query(
            args.query_img, C.QUERY_ENCODER.get("size", 224))

        dets = model.predict(page_t.to(device), query_t.to(device), img_shape)
        det  = dets[0]

        pred_is_tp = [False] * len(det["boxes"])
        out_img = draw_paper_styled_image(
            page_pil, query_pil,
            det["boxes"], det["scores"], pred_is_tp,
            torch.zeros((0, 4)), [],
            sample_num=1, score_thr=args.score_thr,
        )
        out_img.save(args.output_img)
        print(f"  {len(det['boxes'])} detecciones → {args.output_img}")
        return

    if not args.image_root:
        print("Especifica --image_root para evaluar, o --page_img + "
              "--query_img para inferencia single.")
        return

    save_imgs    = args.save_images and not args.no_images
    internal_dir = Path(args.output_dir) / "internal"

    print(f"\n{'═'*70}")
    print(f"  [1/2] EVALUACIÓN INTERNA  (split '{args.split}')  → {internal_dir}")
    print(f"{'═'*70}")
    evaluate_and_save(
        model        = model,
        image_root   = args.image_root,
        checkpoint   = args.checkpoint,
        output_dir   = str(internal_dir),
        split        = args.split,
        score_thr    = args.score_thr,
        iou_thr      = args.iou_thr,
        save_images  = save_imgs,
        max_det_imgs = args.max_det_imgs,
        kit_dir      = None,
        device       = device,
    )

if __name__ == "__main__":
    main()