"""
train.py — Entrenamiento FCOS sobre backbone iDoc congelado.

Uso (desde la raiz del proyecto, e.g. /home/rvdl/rvdl_2/):
    CUDA_VISIBLE_DEVICES=0,1 python3 -m train_fcos.train \
        --image_root /home/rvdl/rvdl_2/ \
        --dataset_json /home/rvdl/rvdl_2/detection_dataset_sketches.json \
        --pretrained /home/rvdl/rvdl_2/idoc_pretrained.pth \
        --output_dir /home/rvdl/rvdl_2/train_fcos/run_01/ \
        --epochs 50 --batch_size 4 --lr 1e-4 --num_workers 4
"""

import os, sys, argparse, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# ── MULTI-GPU: Wrapper para exponer forward unificado ───────────────────────
# Encapsula forward_with_embeddings (train) y predict (val) en un único
# nn.Module.forward(), compatible con nn.DataParallel.
# En single-GPU también se usa el wrapper para mantener una sola
# interfaz en train_one_epoch y evaluate.
class MultiGPUWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, pages, queries, mode="train", shapes=None):
        if mode == "train":
            return self.base_model.forward_with_embeddings(pages, queries)
        else:
            return self.base_model.predict(pages, queries, shapes)


# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.abspath(os.path.dirname(__file__))   # /home/rvdl_2/train_fcos/
ROOT  = os.path.abspath(os.path.join(_HERE, ".."))   # /home/rvdl_2/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_fcos.models.detector        import FCOSDetector
from train_fcos.losses.fcos_loss       import FCOSLoss
from train_fcos.datasets               import build_datasets, collate_fn
from train_fcos.utils.metrics          import DetectionEvaluator
import train_fcos.config               as C

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[train] TensorBoard no disponible, se omite.")


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Entrenamiento FCOS sobre iDoc")
    p.add_argument("--image_root",    type=str, required=True)
    p.add_argument("--output_dir",    type=str, default=C.OUTPUT_DIR)
    p.add_argument("--pretrained",    type=str, default=C.PRETRAINED_PTH)
    p.add_argument("--dataset_json",  type=str, default=C.DATASET_JSON)
    p.add_argument("--epochs",        type=int,   default=C.TRAIN["epochs"])
    p.add_argument("--batch_size",    type=int,   default=C.TRAIN["batch_size"])
    p.add_argument("--lr",            type=float, default=C.OPTIMIZER["lr"],
                        help="LR base (FPN, convs, level_norms)")
    p.add_argument("--lr_crossattn",  type=float, default=None,
                        help="LR para CrossAttn layers. Default: 4 × lr")
    p.add_argument("--lr_film",       type=float, default=None,
                        help="LR para FiLM layers. Default: 0.2 × lr")
    p.add_argument("--warmup_epochs", type=int,   default=None,
                        help="Override warmup (default: del config)")
    p.add_argument("--accum_steps",   type=int,   default=1,
                        help="Pasos de gradient accumulation. "
                             "Batch efectivo = batch_size × accum_steps.")
    p.add_argument("--freeze_except", type=str,   default=None,
                        choices=["crossattn", "film", "other"],
                        help="FASE 1: congela todo excepto el grupo indicado.")
    p.add_argument("--num_workers",   type=int,   default=C.TRAIN["num_workers"])
    p.add_argument("--resume",        type=str,   default=None)
    p.add_argument("--seed",          type=int,   default=C.TRAIN["seed"])
    return p.parse_args()


# ─── Utilidades ───────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cosine_lr_groups(optimizer, epoch, total_epochs, warmup_epochs, base_lrs, min_lr):
    """
    Scheduler cosine con LR diferente por param group.
    base_lrs: lista con un LR base por group (mismo orden que optimizer.param_groups).
    """
    if epoch < warmup_epochs:
        factor = (epoch + 1) / warmup_epochs
    else:
        denom = max(total_epochs - warmup_epochs, 1)
        t = (epoch - warmup_epochs) / denom
        factor = 0.5 * (1.0 + math.cos(math.pi * t))

    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = min_lr + (base_lr - min_lr) * factor

    return optimizer.param_groups[0]["lr"]   # retorna LR del primer grupo (log)


def build_param_groups(model, lr, lr_crossattn, lr_film):
    """
    Agrupa los parámetros entrenables por tipo de módulo:
        cross_attn  → lr_crossattn  (empieza desde cero, necesita LR alto)
        film        → lr_film       (ya entrenado, LR bajo)
        refinement  → lr_crossattn  (SketchRefinementEncoder, nuevo)
        resto       → lr            (FPN, convs, level_norms, scales, predictors)

    NOTA: recibe el FCOSDetector original (sin wrappers) para que
    named_parameters() devuelva los nombres reales del modelo.
    """
    groups = {"crossattn": [], "film": [], "refinement": [], "other": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "cross_attn" in name:
            groups["crossattn"].append(param)
        elif "film" in name:
            groups["film"].append(param)
        elif "refinement" in name:
            groups["refinement"].append(param)
        else:
            groups["other"].append(param)

    n_ca  = sum(p.numel() for p in groups["crossattn"])
    n_ref = sum(p.numel() for p in groups["refinement"])
    n_fil = sum(p.numel() for p in groups["film"])
    n_oth = sum(p.numel() for p in groups["other"])
    print(f"  Param groups → crossattn: {n_ca:,} (lr={lr_crossattn:.1e})  "
          f"refinement: {n_ref:,} (lr={lr_crossattn:.1e})  "
          f"film: {n_fil:,} (lr={lr_film:.1e})  "
          f"other: {n_oth:,} (lr={lr:.1e})")

    all_groups = [
        {"params": groups["crossattn"],  "lr": lr_crossattn, "name": "cross_attn"},
        {"params": groups["refinement"], "lr": lr_crossattn, "name": "refinement"},
        {"params": groups["film"],       "lr": lr_film,      "name": "film"},
        {"params": groups["other"],      "lr": lr,           "name": "other"},
    ]
    all_lrs = [lr_crossattn, lr_crossattn, lr_film, lr]
    param_groups = [g for g in all_groups if len(g["params"]) > 0]
    base_lrs     = [l for g, l in zip(all_groups, all_lrs) if len(g["params"]) > 0]
    return param_groups, base_lrs


def save_ckpt(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint guardado: {path}")


def load_ckpt(path, model, optimizer=None, scaler=None):
    """
    Carga checkpoint en el FCOSDetector original (sin wrappers).
    strict=False para tolerar cambios de arquitectura; loggea las diferencias.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["model"], strict=False)
    if result.missing_keys:
        print(f"  [load_ckpt] Keys faltantes ({len(result.missing_keys)}): "
              f"{result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
    if result.unexpected_keys:
        print(f"  [load_ckpt] Keys inesperadas ({len(result.unexpected_keys)}): "
              f"{result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    ep   = ckpt.get("epoch",    0)
    bmap = ckpt.get("best_map", 0.0)
    print(f"  ✓ Resumiendo desde {path} (epoch {ep}, best mAP={bmap:.4f})")
    return ep, bmap


def apply_phase_freeze(model, freeze_except: str) -> int:
    """
    Entrenamiento en dos fases:
    Fase 1 (freeze_except='crossattn'): congela FiLM + FPN + convs.
    Fase 2 (sin freeze): fine-tuning conjunto de todo.

    NOTA: aplicar sobre FCOSDetector original, antes de crear el wrapper.
    Retorna el número de parámetros activos.
    """
    keyword_map = {
        "crossattn": "cross_attn",
        "film":      "film",
        "other":     None,
    }
    keyword = keyword_map[freeze_except]

    for name, param in model.named_parameters():
        if not param.requires_grad:   # ViT ya frozen, no tocar
            continue
        if freeze_except == "other":
            param.requires_grad = ("cross_attn" not in name and "film" not in name)
        else:
            param.requires_grad = (keyword in name)

    n_active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Freeze mode '{freeze_except}': "
          f"{n_active:,} entrenables | {n_frozen:,} congelados")
    return n_active


def unwrap_model(model):
    """
    Extrae el FCOSDetector original navegando por:
      nn.DataParallel  →  .module
      MultiGPUWrapper  →  .base_model
    """
    m = model
    if hasattr(m, "module"):        # DataParallel
        m = m.module
    if hasattr(m, "base_model"):    # MultiGPUWrapper
        m = m.base_model
    return m


# ─── Train epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loss_fn, optimizer, base_model, loader, epoch, device,
                    scaler=None, clip_grad=1.0, log_freq=20, accum_steps=1):
    """
    Un epoch de entrenamiento con soporte de gradient accumulation.

    model      : MultiGPUWrapper (con o sin DataParallel) — usado en el forward.
    base_model : FCOSDetector original — usado para clip_grad_norm_ sobre los
                 parámetros reales (evita ambigüedad con el wrapper).
    accum_steps: acumula gradientes N iters antes de optimizer.step().
                 Batch efectivo = batch_size × accum_steps.
    """
    model.train()
    optimizer.zero_grad()
    total, n = 0.0, 0

    for it, batch in enumerate(loader):
        pages   = batch["page_imgs"].to(device, non_blocking=True)
        queries = batch["query_imgs"].to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]

        batch_cls_labels = torch.tensor(
            [t["labels"][0].item() if len(t["labels"]) > 0 else -1
             for t in batch["targets"]],
            dtype=torch.long, device=device,
        )

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            cls_, bbox_, ctr_, cls_tokens = model(pages, queries, mode="train")

            valid_mask   = batch_cls_labels >= 0
            cls_tok_cont = cls_tokens[valid_mask] if valid_mask.any() else None
            lbl_cont     = batch_cls_labels[valid_mask] if valid_mask.any() else None

            losses = loss_fn(
                cls_, bbox_, ctr_, targets,
                cls_tokens          = cls_tok_cont,
                labels_for_contrast = lbl_cont,
            )
            loss = losses["loss"] / accum_steps

        raw_loss = loss.item() * accum_steps

        if not math.isfinite(raw_loss):
            print(f"  ✗ Loss no finita en iter {it} ({raw_loss:.4f}). Saltando.")
            optimizer.zero_grad(); continue

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last_iter  = (it + 1 == len(loader))
        is_accum_step = ((it + 1) % accum_steps == 0)
        if is_accum_step or is_last_iter:
            if scaler:
                scaler.unscale_(optimizer)
                # Clip sobre base_model para operar solo en los parámetros reales
                nn.utils.clip_grad_norm_(base_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(base_model.parameters(), clip_grad)
                optimizer.step()
            optimizer.zero_grad()

        total += raw_loss; n += 1
        if it % log_freq == 0:
            contrast_val = losses.get("loss_contrast", torch.tensor(0.0)).item()
            print(f"  [{it}/{len(loader)}] "
                  f"loss={raw_loss:.4f} cls={losses['loss_cls'].item():.4f} "
                  f"bbox={losses['loss_bbox'].item():.4f} "
                  f"ctr={losses['loss_ctr'].item():.4f} "
                  f"contrast={contrast_val:.4f} "
                  f"n_pos={losses['n_pos']:.0f}")

    return total / max(n, 1)


# ─── Val epoch ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, num_classes, eval_cfg):
    """
    Evaluación: extrae el FCOSDetector base con unwrap_model() para
    usar su método .predict() nativo directamente, sin depender del wrapper.
    """
    actual_model = unwrap_model(model)
    actual_model.eval()

    ev = DetectionEvaluator(num_classes=num_classes,
                            iou_threshold=eval_cfg["iou_threshold"])
    for batch in loader:
        pages   = batch["page_imgs"].to(device)
        queries = batch["query_imgs"].to(device)
        shapes  = batch["img_shapes"]
        dets = actual_model.predict(pages, queries, shapes[0])
        ev.update(
            [{"boxes": d["boxes"].cpu(), "scores": d["scores"].cpu(),
              "labels": d["labels"].cpu()} for d in dets],
            batch["targets"]
        )
    return ev.compute()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs")) if HAS_TB else None

    cfg = {
        "BACKBONE":       C.BACKBONE,
        "FPN":            C.FPN,
        "FCOS_HEAD":      C.FCOS_HEAD,
        "QUERY_ENCODER":  C.QUERY_ENCODER,
        "DATASET":        C.DATASET,
        "AUGMENTATION":   C.AUGMENTATION,
        "EVAL":           C.EVAL,
        "PRETRAINED_PTH": args.pretrained,
    }

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Cargando datasets...")
    train_ds, val_ds, _ = build_datasets(args.dataset_json, args.image_root, cfg)
    eff_batch = args.batch_size * args.accum_steps
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    if args.accum_steps > 1:
        print(f"  Gradient accumulation: {args.accum_steps} steps "
              f"→ batch efectivo = {eff_batch}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    # ── Modelo base ───────────────────────────────────────────────────────────
    # Todas las operaciones sobre parámetros (freeze, param groups, optimizer,
    # checkpoints) se hacen sobre el FCOSDetector original (model).
    # El wrapper/DataParallel solo se usa en el forward pass de entrenamiento.
    print("Construyendo modelo...")
    model = FCOSDetector(cfg).to(device)

    # ── Freeze de fase ────────────────────────────────────────────────────────
    if args.freeze_except:
        apply_phase_freeze(model, args.freeze_except)
    else:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_froz  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  Entrenables: {n_train:,} | Congelados: {n_froz:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    cw = train_ds.class_weights.to(device) if C.LOSS["use_class_weights"] else None
    loss_fn = FCOSLoss(
        num_classes=C.FCOS_HEAD["num_classes"], strides=C.FCOS_HEAD["strides"],
        regress_ranges=C.FCOS_HEAD["regress_ranges"],
        focal_alpha=C.LOSS["focal_alpha"], focal_gamma=C.LOSS["focal_gamma"],
        lambda_cls=C.LOSS["lambda_cls"], lambda_bbox=C.LOSS["lambda_bbox"],
        lambda_ctr=C.LOSS["lambda_ctr"],
        lambda_contrast=C.LOSS.get("lambda_contrast", 0.1),
        contrast_temp=C.LOSS.get("contrast_temp", 0.07),
        norm_on_bbox=C.FCOS_HEAD["norm_on_bbox"],
        class_weights=cw,
    ).to(device)

    # ── Optimizador con LR diferenciales ─────────────────────────────────────
    lr_crossattn = args.lr_crossattn if args.lr_crossattn is not None else args.lr * 4.0
    lr_film      = args.lr_film      if args.lr_film      is not None else args.lr * 0.2
    warmup_ep    = (args.warmup_epochs if args.warmup_epochs is not None
                    else C.SCHEDULER["warmup_epochs"])

    # build_param_groups sobre model (FCOSDetector), no sobre el wrapper
    param_groups, base_lrs = build_param_groups(model, args.lr, lr_crossattn, lr_film)
    optimizer = torch.optim.AdamW(param_groups,
                                  weight_decay=C.OPTIMIZER["weight_decay"],
                                  betas=C.OPTIMIZER["betas"])
    scaler = torch.amp.GradScaler() if C.TRAIN["use_fp16"] else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_map, no_improve = 0, 0.0, 0
    if args.resume:
        start_epoch, best_map = load_ckpt(args.resume, model, optimizer, scaler)

    end_epoch = start_epoch + args.epochs

    # ── Multi-GPU: siempre usar el Wrapper para tener una interfaz unificada ──
    # Sin DataParallel (1 GPU):  model_gpu = MultiGPUWrapper(model)
    # Con DataParallel (N GPUs): model_gpu = DataParallel(MultiGPUWrapper(model))
    # En ambos casos train_one_epoch llama model_gpu(pages, queries, mode="train")
    # y evaluate() usa unwrap_model() para acceder a FCOSDetector.predict().
    wrapped = MultiGPUWrapper(model)
    n_gpus  = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"\n[INFO] {n_gpus} GPUs detectadas → activando DataParallel")
        model_gpu = nn.DataParallel(wrapped)
    else:
        print(f"\n[INFO] {n_gpus} GPU → modo single-GPU")
        model_gpu = wrapped

    # ── Loop ──────────────────────────────────────────────────────────────────
    print(f"\nEntrenando {args.epochs} épocas "
          f"(epochs {start_epoch+1}→{end_epoch})...\n")
    t0 = time.time()

    for epoch in range(start_epoch, end_epoch):
        # epoch_rel: posición relativa dentro de este tramo de entrenamiento.
        epoch_rel = epoch - start_epoch
        lr = cosine_lr_groups(optimizer, epoch_rel, args.epochs,
                              warmup_ep, base_lrs, C.SCHEDULER["min_lr"])

        _lrs     = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
        _lr_ca   = _lrs.get("cross_attn", 0.0)
        _lr_film = _lrs.get("film",       0.0)
        _lr_oth  = _lrs.get("other",      lr)
        print(f"{'='*60}")
        print(f"Epoch {epoch+1}/{end_epoch}  (rel {epoch_rel+1}/{args.epochs})  "
              f"lr_ca={_lr_ca:.1e}  lr_film={_lr_film:.1e}  lr={_lr_oth:.2e}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(
            model_gpu, loss_fn, optimizer, model,   # model = base para clip_grad
            train_loader, epoch, device, scaler,
            C.TRAIN["clip_grad_norm"], C.TRAIN["log_freq"],
            accum_steps=args.accum_steps,
        )

        val_m   = evaluate(model_gpu, val_loader, device,
                           C.FCOS_HEAD["num_classes"], C.EVAL)
        val_map = val_m["mAP"]
        print(f"  → loss={train_loss:.4f}  val_mAP={val_map:.4f}")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("mAP/val",    val_map,    epoch)
            _lrs_tb = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
            writer.add_scalar("LR/crossattn", _lrs_tb.get("cross_attn", 0), epoch)
            writer.add_scalar("LR/film",      _lrs_tb.get("film",       0), epoch)
            writer.add_scalar("LR/other",     _lrs_tb.get("other",     lr), epoch)
            for cid, ap in val_m["per_class"].items():
                writer.add_scalar(f"AP/{cid}", ap, epoch)

        # Checkpoints: guardar siempre model.state_dict() (FCOSDetector sin wrapper)
        # para mantener compatibilidad con scripts de inferencia externos.
        if (epoch + 1) % C.TRAIN["save_every"] == 0:
            save_ckpt({"epoch": epoch+1, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict() if scaler else None,
                       "best_map": best_map, "metrics": val_m},
                      os.path.join(args.output_dir, f"checkpoint_ep{epoch+1:03d}.pth"))

        if val_map > best_map:
            best_map = val_map; no_improve = 0
            save_ckpt({"epoch": epoch+1, "model": model.state_dict(),
                       "best_map": best_map, "metrics": val_m},
                      os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ★ Nuevo mejor mAP: {best_map:.4f}")
        else:
            no_improve += 1
            print(f"  Sin mejora ({no_improve}/{C.TRAIN['early_stop_patience']})")
            if no_improve >= C.TRAIN["early_stop_patience"]:
                print("Early stopping."); break

    elapsed = (time.time() - t0) / 3600
    print(f"\nListo en {elapsed:.1f}h | "
          f"Epochs {start_epoch+1}→{end_epoch} | Mejor mAP: {best_map:.4f}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()