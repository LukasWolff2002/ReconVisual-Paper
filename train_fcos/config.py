"""
config.py
Hiperparámetros del entrenamiento FCOS sobre backbone iDoc congelado.
Arquitectura: FiLM(CLS) + CrossAttn(patch tokens) para condicionamiento por query.

Cambios respecto a la versión anterior:
  AUGMENTATION: nuevas claves para zoom-in, copy-paste enriquecido y hard-negatives.
  FCOS_HEAD:    cross_attn_start_level (0=P2, 1=P3 recomendado, 2=P4)
  EVAL:         parámetros de adaptive threshold y context-aware NMS.
"""

import os

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IDOC_DIR        = os.path.join(ROOT_DIR, "iDoc")
PRETRAINED_PTH  = os.path.join(ROOT_DIR, "idoc_pretrained.pth")
DATASET_JSON    = os.path.join(ROOT_DIR, "detection_dataset_sketches.json")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "train_fcos", "outputs")

# ─── Backbone (ViT-Base, frozen) ──────────────────────────────────────────────
BACKBONE = dict(
    arch           = "vit_base",
    patch_size     = 16,
    embed_dim      = 768,
    depth          = 12,
    num_heads      = 12,
    extract_layers = [2, 5, 8, 11],
    out_channels   = [768, 768, 768, 768],
)

# ─── FPN ──────────────────────────────────────────────────────────────────────
FPN = dict(
    in_channels  = BACKBONE["out_channels"],   # [768, 768, 768, 768]
    out_channels = 256,
    num_levels   = 5,                          # P2-P6
)

# ─── Query encoder ────────────────────────────────────────────────────────────
QUERY_ENCODER = dict(
    embed_dim        = BACKBONE["embed_dim"],   # 768
    film_out_dim     = FPN["out_channels"],     # 256
    # [ENC-1] 7×7 = 49 tokens (antes 4×4=16). Más detalle espacial para
    # símbolos con estructura fina. Coste: O(N_s) lineal, impacto moderado.
    sketch_pool_size = 7,
    size             = 224,
    # [ENC-2] SketchRefinementEncoder entrenable (2 capas Transformer).
    # Near-identity init → compatible con checkpoints FiLM existentes.
    # use_refinement=False para ablación o para ckpt antiguo.
    use_refinement   = True,
    refine_layers    = 2,
    refine_heads     = 8,
    refine_ffn_dim   = 2048,
    refine_dropout   = 0.1,
)

# ─── FCOS Head ────────────────────────────────────────────────────────────────
FCOS_HEAD = dict(
    in_channels  = FPN["out_channels"],    # 256
    num_convs    = 4,
    num_classes  = 22,
    regress_ranges = (
        (0,   32),    # P2 stride 4
        (32,  64),    # P3 stride 8
        (64,  128),   # P4 stride 16
        (128, 256),   # P5 stride 32
        (256, 1e8),   # P6 stride 64
    ),
    strides           = [4, 8, 16, 32, 64],
    centerness_on_reg = True,
    norm_on_bbox      = True,
    cross_attn_heads  = 8,
    cross_attn_drop   = 0.1,
    # [ARCH] 0=P2 (más agresivo, +memoria), 1=P3 (recomendado), 2=P4 (más ligero)
    # Bajar a 0 si los falsos positivos persisten en texturas finas pequeñas.
    cross_attn_start_level = 1,
)

# ─── Loss ─────────────────────────────────────────────────────────────────────
LOSS = dict(
    focal_alpha            = 0.25,
    focal_gamma            = 2.0,
    use_class_weights      = True,
    loss_bbox_weight       = 1.0,
    loss_centerness_weight = 1.0,
    lambda_cls             = 1.0,
    lambda_bbox            = 1.0,
    lambda_ctr             = 1.0,
    # [CONTRAST] SupCon loss sobre embeddings CLS del sketch encoder.
    # Separa clases similares en el espacio latente.
    # lambda_contrast=0 desactiva la pérdida (backward compatible).
    # Empezar con 0.05 y subir a 0.1 si las clases similares siguen confundiéndose.
    lambda_contrast        = 0.1,
    contrast_temp          = 0.07,   # temperatura SupCon (0.07 es estándar)
)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET = dict(
    train_ratio = 0.8,
    val_ratio   = 0.1,
    test_ratio  = 0.1,
    seed        = 42,
    min_size    = 800,
    max_size    = 1333,
    pixel_mean  = [0.485, 0.456, 0.406],
    pixel_std   = [0.229, 0.224, 0.225],
)

# ─── Augmentación ─────────────────────────────────────────────────────────────
AUGMENTATION = dict(
    # ── Multi-scale y flip ────────────────────────────────────────────────────
    multi_scale_sizes = [640, 720, 800, 900, 1024],
    flip_prob         = 0.5,
    color_jitter_prob = 0.8,
    brightness        = 0.3,
    contrast          = 0.3,
    saturation        = 0.2,
    hue               = 0.05,

    # ── [AUG-2] Zoom-in sobre instancias pequeñas ─────────────────────────────
    # Probabilidad de aplicar zoom-in forzado sobre una instancia positiva.
    # Aumenta la resolución efectiva de objetos pequeños en P2/P3.
    zoom_in_prob   = 0.3,   # 0=desactivado, 0.3=recomendado, 0.5=agresivo
    zoom_in_margin = 0.15,  # margen relativo alrededor del objeto (15%)

    # ── [AUG-1] Copy-paste enriquecido ───────────────────────────────────────
    copy_paste_prob             = 0.5,
    copy_paste_max_objects      = 8,
    copy_paste_priority_classes = ["marqeur", "croix", "pdp", "S", "T", "petit_A"],
    # Escala del crop (0.6–1.4 simula variabilidad de tamaño)
    cp_scale_range = (0.6, 1.4),
    # Flip del crop pegado
    cp_flip_prob   = 0.5,
    cp_vflip_prob  = 0.2,
    # Jitter de color independiente del crop
    cp_color_prob  = 0.7,
    cp_brightness  = 0.4,
    cp_contrast    = 0.4,
    cp_saturation  = 0.3,
    cp_hue         = 0.08,

    # ── [AUG-3] Hard-negative context pages ──────────────────────────────────
    # Probabilidad de aplicar blending con página densa (sin clase objetivo).
    # Solo se activa cuando la muestra ya no tiene instancias positivas.
    hard_neg_prob = 0.2,   # 0=desactivado, 0.2=recomendado, 0.4=agresivo
)

# ─── Optimizador ──────────────────────────────────────────────────────────────
OPTIMIZER = dict(
    type         = "AdamW",
    lr           = 1e-4,
    weight_decay = 1e-4,
    betas        = (0.9, 0.999),
)

SCHEDULER = dict(
    type          = "cosine",
    warmup_epochs = 5,
    min_lr        = 1e-6,
)

# ─── Entrenamiento ────────────────────────────────────────────────────────────
TRAIN = dict(
    epochs              = 80,
    batch_size          = 4,
    num_workers         = 4,
    clip_grad_norm      = 1.0,
    save_every          = 10,
    early_stop_patience = 1000,
    use_fp16            = True,
    log_freq            = 20,
    seed                = 42,
)

# ─── Evaluación ───────────────────────────────────────────────────────────────
EVAL = dict(
    iou_threshold   = 0.5,

    # ── [NMS-2] Adaptive score threshold ─────────────────────────────────────
    # Si pre-NMS hay más de density_trigger candidatos, se usa el percentil
    # adaptive_percentile de los scores como threshold mínimo.
    score_threshold    = 0.05,   # threshold base (páginas simples)
    density_trigger    = 50,     # nº candidatos que activa el modo adaptivo
    adaptive_percentile = 0.90,  # percentil → top 10% más seguros pasan

    # ── NMS estándar ─────────────────────────────────────────────────────────
    nms_iou_thresh  = 0.5,
    max_dets        = 100,

    # ── [NMS-1] Context-aware NMS ─────────────────────────────────────────────
    # cluster_iou_thr:  IoU para considerar que dos dets son vecinas (< NMS thr)
    # cluster_min_det:  mínimo de dets para declarar un "cluster de textura"
    # score_spread_thr: si max-min scores del cluster < este valor, es textura
    cluster_iou_thr  = 0.20,
    cluster_min_det  = 4,
    score_spread_thr = 0.15,
)
