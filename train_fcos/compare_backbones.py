"""
compare_backbones.py — iDoc-FCOS  |  Comparación iDoc vs DINOv3 (Evaluación Interna)

Para CADA backbone (iDoc y DINOv3) corre la evaluación interna de evaluate.py
(split val/test) y al final construye reportes + gráficas comparativas.

Estructura de salida:
    eval_comparison/
    ├── idoc/
    │   └── internal/        — report.txt, plots/, detections/ (split val/test)
    ├── dino/
    │   └── internal/        — report.txt, plots/, detections/ (split val/test)
    └── comparison/
        ├── comparison_report.txt        — interno, iDoc vs DINOv3
        └── plots/
            ├── 01_global_metrics_comparison.png       (interno)
            ├── 02_ap_per_class_comparison.png         (interno)
            ├── 03_delta_ap_per_class.png               (interno)
            └── 04_summary_dashboard.png                (interno)
"""

import os
import sys
import gc
import time
import argparse
from pathlib import Path

import torch
import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
ROOT  = os.path.abspath(os.path.join(_HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_fcos.config as C
# ── Se eliminó 'evaluate_full_kit_corpus' de la importación ──
from train_fcos.evaluate import (
    load_model, evaluate_and_save,
    PLT_BLUE, PLT_GREEN, PLT_RED, PLT_ORANGE, PLT_PURPLE, MPL_RC,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[compare_backbones] matplotlib no disponible — se omiten gráficas.")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Comparación iDoc vs DINOv3 — iDoc-FCOS")
    p.add_argument("--checkpoint_idoc", type=str, required=True)
    p.add_argument("--pretrained_idoc", type=str, required=True)
    p.add_argument("--checkpoint_dino", type=str, required=True)
    p.add_argument("--pretrained_dino", type=str, required=True)
    p.add_argument("--image_root",      type=str, required=True)
    p.add_argument("--dataset_json",    type=str, default=C.DATASET_JSON)
    p.add_argument("--output_dir",      type=str, default="eval_comparison")
    p.add_argument("--split",           type=str, default="val",
                   choices=["val", "test"])
    p.add_argument("--score_thr",       type=float, default=0.25)
    p.add_argument("--iou_thr",         type=float, default=0.5)
    p.add_argument("--max_det_imgs",    type=int, default=50,
                   help="Imágenes de detección por modelo (bajo por defecto).")
    p.add_argument("--no_images",       action="store_true",
                   help="Omite guardar imágenes anotadas (más rápido).")
    
    # Mantenemos skip_full_kit por retrocompatibilidad con el comando, 
    # aunque ahora se ignora silenciosamente al haber eliminado el kit.
    p.add_argument("--skip_full_kit", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE UN BACKBONE: evaluación interna
# ══════════════════════════════════════════════════════════════════════════════

def _run_one(name: str, checkpoint: str, pretrained: str,
             args, device: torch.device) -> dict:
    
    print(f"\n{'═'*70}")
    print(f"  BACKBONE: {name}")
    print(f"{'═'*70}")

    model    = load_model(checkpoint, pretrained, device)
    base_dir = Path(args.output_dir) / name.lower()

    print(f"\n  [1/2] Evaluación interna (split '{args.split}')")
    res_internal = evaluate_and_save(
        model        = model,
        image_root   = args.image_root,
        checkpoint   = checkpoint,
        output_dir   = str(base_dir / "internal"),
        split        = args.split,
        score_thr    = args.score_thr,
        iou_thr      = args.iou_thr,
        save_images  = not args.no_images,
        max_det_imgs = args.max_det_imgs,
        kit_dir      = None,
        device       = device,
    )

    print(f"\n  [2/2] Evaluación oficial OMITIDA (bypass activado)")
    res_official = None

    # Liberar memoria antes de cargar el siguiente backbone
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"internal": res_internal, "official": res_official}


def _global_stats(result: dict) -> dict:
    internal = result["internal"]
    gt = sum(internal["per_cls_gt"].values())
    tp = sum(internal["per_cls_tp"].values())
    fp = sum(internal["per_cls_fp"].values())
    fn = sum(internal["per_cls_fn"].values())
    p  = tp / max(tp + fp, 1)
    r  = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"mAP": internal["metrics"]["mAP"], "precision": p, "recall": r,
            "f1": f1, "gt": gt, "tp": tp, "fp": fp, "fn": fn}


def _per_class_rows(res_idoc: dict, res_dino: dict) -> list:
    internal_i = res_idoc["internal"]
    internal_d = res_dino["internal"]
    idx_to_name = internal_i["idx_to_name"]
    ap_i_all    = internal_i["metrics"]["per_class"]
    ap_d_all    = internal_d["metrics"]["per_class"]
    gt_i        = internal_i["per_cls_gt"]
    gt_d        = internal_d["per_cls_gt"]

    rows = []
    for idx, name in enumerate(idx_to_name):
        if gt_i.get(idx, 0) + gt_d.get(idx, 0) == 0:
            continue
        rows.append((name, ap_i_all.get(idx, 0.0), ap_d_all.get(idx, 0.0)))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# REPORTE COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════════

def generate_comparison_report(res_idoc: dict, res_dino: dict,
                                args, out_dir: Path) -> Path:
    W = 90
    lines = []
    lines.append("╔" + "═"*(W-2) + "╗")
    lines.append("║" + "  iDoc-FCOS — Backbone Comparison: iDoc vs DINOv3".center(W-2) + "║")
    lines.append("╚" + "═"*(W-2) + "╝")
    lines.append("")
    lines.append(f"  Split            : {args.split}")
    lines.append(f"  IoU threshold    : {args.iou_thr:.2f}")
    lines.append(f"  Score threshold  : {args.score_thr:.2f}")
    lines.append(f"  Checkpoint iDoc  : {args.checkpoint_idoc}")
    lines.append(f"  Checkpoint DINOv3: {args.checkpoint_dino}")

    g_idoc = _global_stats(res_idoc)
    g_dino = _global_stats(res_dino)

    # ── Métricas internas globales ────────────────────────────────────────────
    lines.append("")
    lines.append(f"{'── INTERNAL METRICS (split ' + args.split + ') ':─<{W}}")
    hdr = (f"  {'Metric':<14}  {'iDoc':>10}  {'DINOv3':>10}  "
           f"{'Δ (v3-iDoc)':>14}  {'Winner':>10}")
    lines.append(hdr)
    lines.append("  " + "─"*(len(hdr)-2))
    for key, label in [("mAP", "mAP"), ("precision", "Precision"),
                       ("recall", "Recall"), ("f1", "F1")]:
        v_i, v_d = g_idoc[key], g_dino[key]
        delta  = v_d - v_i
        winner = ("DINOv3" if delta > 1e-6 else
                 "iDoc"   if delta < -1e-6 else "Tie")
        lines.append(f"  {label:<14}  {v_i:>10.4f}  {v_d:>10.4f}  "
                     f"{delta:>+14.4f}  {winner:>10}")
    lines.append("")
    lines.append(f"  {'GT boxes':<14}  {g_idoc['gt']:>10,}  {g_dino['gt']:>10,}")
    lines.append(f"  {'TP':<14}  {g_idoc['tp']:>10,}  {g_dino['tp']:>10,}")
    lines.append(f"  {'FP':<14}  {g_idoc['fp']:>10,}  {g_dino['fp']:>10,}")
    lines.append(f"  {'FN':<14}  {g_idoc['fn']:>10,}  {g_dino['fn']:>10,}")

    # ── Tabla por clase (interna) ─────────────────────────────────────────────
    lines.append("")
    lines.append(f"{'── AP PER CLASS — internal (iDoc vs DINOv3) ':─<{W}}")
    rows = _per_class_rows(res_idoc, res_dino)
    rows_with_delta = [(name, ap_i, ap_d, ap_d - ap_i) for name, ap_i, ap_d in rows]
    rows_with_delta.sort(key=lambda x: -abs(x[3]))

    hdr2 = (f"  {'Class':<22}  {'AP iDoc':>9}  {'AP DINOv3':>10}  "
            f"{'Δ':>8}  {'Winner':>10}")
    lines.append(hdr2)
    lines.append("  " + "─"*(len(hdr2)-2))

    n_idoc_wins, n_dino_wins, n_tie = 0, 0, 0
    for name, ap_i, ap_d, delta in rows_with_delta:
        winner = ("DINOv3" if delta > 1e-6 else
                 "iDoc"   if delta < -1e-6 else "Tie")
        if winner == "DINOv3": n_dino_wins += 1
        elif winner == "iDoc": n_idoc_wins += 1
        else: n_tie += 1
        lines.append(f"  {name:<22}  {ap_i:>9.3f}  {ap_d:>10.3f}  "
                     f"{delta:>+8.3f}  {winner:>10}")

    lines.append("")
    lines.append(f"{'── INTERNAL SUMMARY ':─<{W}}")
    lines.append(f"  Classes won by iDoc     : {n_idoc_wins}")
    lines.append(f"  Classes won by DINOv3   : {n_dino_wins}")
    lines.append(f"  Classes tied            : {n_tie}")
    overall_winner = "DINOv3" if g_dino["mAP"] > g_idoc["mAP"] else "iDoc"
    lines.append(f"  Backbone with best mAP  : {overall_winner}  "
                 f"(Δ mAP = {g_dino['mAP']-g_idoc['mAP']:+.4f})")

    lines.append("")
    lines.append("─"*W)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "comparison_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS COMPARATIVAS (FORMATO PAPER - INGLÉS)
# ══════════════════════════════════════════════════════════════════════════════

def generate_comparison_plots(res_idoc: dict, res_dino: dict,
                              plot_path: Path) -> int:
    
    # ── CONFIGURACIÓN PAPER (FONT FALLBACK & ENGLISH) ────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Bitstream Vera Serif", "serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })
    # ─────────────────────────────────────────────────────────────────────────

    plot_path.mkdir(parents=True, exist_ok=True)
    n = 0

    g_idoc   = _global_stats(res_idoc)
    g_dino   = _global_stats(res_dino)
    cls_rows = _per_class_rows(res_idoc, res_dino)

    # ── 01: Global Metrics ───────────────────────────────────────────────────
    try:
        metrics_names = ["mAP", "Precision", "Recall", "F1"]
        v_idoc = [g_idoc["mAP"], g_idoc["precision"], g_idoc["recall"], g_idoc["f1"]]
        v_dino = [g_dino["mAP"], g_dino["precision"], g_dino["recall"], g_dino["f1"]]
        x = np.arange(len(metrics_names)); w = 0.32

        fig, ax = plt.subplots(figsize=(6, 3))
        b1 = ax.bar(x - w/2, v_idoc, w, label="iDoc",   color=PLT_BLUE,   alpha=0.88)
        b2 = ax.bar(x + w/2, v_dino, w, label="DINOv3", color=PLT_ORANGE, alpha=0.88)
        
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_path / "01_global_metrics_comparison.png")
        plt.close(fig)
        n += 1
        print("    ✓ 01_global_metrics_comparison.png")
    except Exception as e:
        print(f"    ✗ 01_global_metrics_comparison: {e}")

    # ── 02: AP per class ─────────────────────────────────────────────────────
    try:
        rows_sorted = sorted(cls_rows, key=lambda r: max(r[1], r[2]))
        names = [r[0] for r in rows_sorted]
        ap_i  = [r[1] for r in rows_sorted]
        ap_d  = [r[2] for r in rows_sorted]
        y = np.arange(len(names)); h = 0.36

        fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.35)))
        ax.barh(y - h/2, ap_i, h, label="iDoc",   color=PLT_BLUE,   alpha=0.88)
        ax.barh(y + h/2, ap_d, h, label="DINOv3", color=PLT_ORANGE, alpha=0.88)
        
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 1.05)
        ax.axvline(g_idoc["mAP"], color=PLT_BLUE,   ls="--", lw=1.2, alpha=0.7)
        ax.axvline(g_dino["mAP"], color=PLT_ORANGE, ls="--", lw=1.2, alpha=0.7)
        ax.set_xlabel("Average Precision (AP)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(plot_path / "02_ap_per_class_comparison.png")
        plt.close(fig)
        n += 1
        print("    ✓ 02_ap_per_class_comparison.png")
    except Exception as e:
        print(f"    ✗ 02_ap_per_class_comparison: {e}")

    # ── 03: Delta AP per class ───────────────────────────────────────────────
    try:
        rows_delta = sorted(cls_rows, key=lambda r: (r[2] - r[1]))
        names  = [r[0] for r in rows_delta]
        deltas = [r[2] - r[1] for r in rows_delta]
        colors = [PLT_GREEN if d > 0 else PLT_RED for d in deltas]

        fig, ax = plt.subplots(figsize=(7, max(4, len(names) * 0.35)))
        bars = ax.barh(names, deltas, color=colors, edgecolor="black", height=0.72, linewidth=0.5)
        
        for bar, d in zip(bars, deltas):
            ax.text(d + (0.004 if d >= 0 else -0.004),
                    bar.get_y() + bar.get_height()/2,
                    f"{d:+.3f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9)
        
        ax.axvline(0, color="black", lw=1.0)
        ax.set_xlabel("$\Delta$ AP (DINOv3 − iDoc)")
        plt.tight_layout()
        fig.savefig(plot_path / "03_delta_ap_per_class.png")
        plt.close(fig)
        n += 1
        print("    ✓ 03_delta_ap_per_class.png")
    except Exception as e:
        print(f"    ✗ 03_delta_ap_per_class: {e}")

    # ── 04: Summary Dashboard ────────────────────────────────────────────────
    try:
        fig = plt.figure(figsize=(10, 7), facecolor="white")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel 1: Metrics Table
        ax1 = fig.add_subplot(gs[0, 0]); ax1.axis("off")
        kv = [
            ("mAP",       g_idoc["mAP"],       g_dino["mAP"]),
            ("Precision", g_idoc["precision"], g_dino["precision"]),
            ("Recall",    g_idoc["recall"],    g_dino["recall"]),
            ("F1",        g_idoc["f1"],        g_dino["f1"]),
        ]
        ax1.text(0.05, 0.95, "Metric", fontweight="bold", transform=ax1.transAxes)
        ax1.text(0.55, 0.95, "iDoc", fontweight="bold", transform=ax1.transAxes, color=PLT_BLUE)
        ax1.text(0.80, 0.95, "DINOv3", fontweight="bold", transform=ax1.transAxes, color=PLT_ORANGE)
        for row, (k, vi, vd) in enumerate(kv):
            y_pos = 0.95 - (row + 1) * 0.18
            ax1.text(0.05, y_pos, k, transform=ax1.transAxes)
            ax1.text(0.55, y_pos, f"{vi:.4f}", transform=ax1.transAxes, color=PLT_BLUE)
            ax1.text(0.80, y_pos, f"{vd:.4f}", transform=ax1.transAxes, color=PLT_ORANGE)

        # Panel 2: Pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        n_dino_wins = sum(1 for _, ai, ad in cls_rows if ad > ai + 1e-6)
        n_idoc_wins = sum(1 for _, ai, ad in cls_rows if ai > ad + 1e-6)
        n_tie       = len(cls_rows) - n_dino_wins - n_idoc_wins
        ax2.pie([n_idoc_wins, n_dino_wins, n_tie],
                labels=[f"iDoc\n({n_idoc_wins})", f"DINOv3\n({n_dino_wins})", f"Tie\n({n_tie})"],
                colors=[PLT_BLUE, PLT_ORANGE, "#BDC3C7"],
                autopct="%1.0f%%", startangle=90,
                wedgeprops={"edgecolor": "black", "linewidth": 0.5})

        # Panel 3: Scatter plot
        ax3 = fig.add_subplot(gs[1, 0])
        ap_i_s   = [r[1] for r in cls_rows]
        ap_d_s   = [r[2] for r in cls_rows]
        colors_s = [PLT_GREEN if d > i else (PLT_RED if i > d else "#7F8C8D") for _, i, d in cls_rows]
        ax3.scatter(ap_i_s, ap_d_s, c=colors_s, s=40, alpha=0.8, edgecolors="black", linewidths=0.5)
        ax3.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.0)
        ax3.set_xlim(-0.02, 1.02); ax3.set_ylim(-0.02, 1.02)
        ax3.set_xlabel("AP iDoc")
        ax3.set_ylabel("AP DINOv3")

        # Panel 4: Histograma de deltas
        ax4 = fig.add_subplot(gs[1, 1])
        deltas_all = [d - i for _, i, d in cls_rows]
        bins = np.linspace(-1, 1, 21)
        ax4.hist(deltas_all, bins=bins, color=PLT_PURPLE, edgecolor="black", alpha=0.85, linewidth=0.5)
        ax4.axvline(0, color="black", lw=1.2, ls="--")
        ax4.set_xlabel("$\Delta$ AP (DINOv3 − iDoc)")
        ax4.set_ylabel("Frequency")

        plt.tight_layout()
        fig.savefig(plot_path / "04_summary_dashboard.png")
        plt.close(fig)
        n += 1
        print("    ✓ 04_summary_dashboard.png")
    except Exception as e:
        print(f"    ✗ 04_summary_dashboard: {e}")

    return n


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()

    # ── 1. iDoc (interna) ────────────────────────────────────────────────────
    res_idoc = _run_one("iDoc", args.checkpoint_idoc, args.pretrained_idoc,
                        args, device)

    # ── 2. DINOv3 (interna) ──────────────────────────────────────────────────
    res_dino = _run_one("DINOv3", args.checkpoint_dino, args.pretrained_dino,
                        args, device)

    # ── 3. Comparación ───────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  GENERANDO COMPARACIÓN")
    print(f"{'═'*70}")
    comp_dir     = Path(args.output_dir) / "comparison"
    report_path  = generate_comparison_report(res_idoc, res_dino, args, comp_dir)
    print(f"  Reporte comparativo → {report_path}")

    if HAS_MPL:
        n_plots = generate_comparison_plots(res_idoc, res_dino, comp_dir / "plots")
        print(f"  Gráficas comparativas → {comp_dir/'plots'}  ({n_plots} archivos)")
    else:
        print("  matplotlib no disponible — se omiten gráficas comparativas.")

    elapsed = (time.time() - t0) / 60
    print(f"\n  Listo en {elapsed:.1f} min.")
    print(f"  mAP interno iDoc   = {res_idoc['internal']['metrics']['mAP']:.4f}")
    print(f"  mAP interno DINOv3 = {res_dino['internal']['metrics']['mAP']:.4f}")


if __name__ == "__main__":
    main()