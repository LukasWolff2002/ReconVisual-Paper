"""
models/fcos_head.py

Cabeza FCOS con condicionamiento híbrido por query sketch:

    1. FiLM(CLS token)        — gate global de canal (mismo que antes)
    2. CrossAttn(patch tokens) — selección espacial local (NUEVO)

El flujo por rama (cls y reg) en cada nivel FPN es:

    feat [B, 256, H, W]
        → FiLM(cls_token)          γ·x + β     — atenúa/amplifica canales globales
        → CrossAttn(patches)       Q=feat K/V=patches — filtra ubicaciones no matching
        → 4 × conv-GN-ReLU
        → predicción (cls / reg+ctr)

CrossAttn se inicializa como identidad (out_proj=0), por lo que el modelo
arranca exactamente desde los pesos FiLM ya entrenados.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attn import SketchCrossAttnLayer


# ─── FiLM ─────────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Genera (γ, β) desde el CLS token del sketch.
    """

    def __init__(self, query_dim: int, feat_channels: int):
        super().__init__()
        self.proj = nn.Linear(query_dim, 2 * feat_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:feat_channels] = 1.0   # γ init = 1

    def forward(self, x: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """x: [B,C,H,W] | query_emb: [B, query_dim] → [B,C,H,W]"""
        params = self.proj(query_emb)
        gamma  = params[:, :x.shape[1]].unsqueeze(-1).unsqueeze(-1)
        beta   = params[:, x.shape[1]:].unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ─── Conv block ───────────────────────────────────────────────────────────────

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.ReLU(inplace=True),
    )


# ─── Scale ────────────────────────────────────────────────────────────────────

class Scale(nn.Module):
    def __init__(self, init_val: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_val], dtype=torch.float))

    def forward(self, x):
        return x * self.scale


# ─── FCOS Head ────────────────────────────────────────────────────────────────

class FCOSHead(nn.Module):
    """
    Cabeza FCOS compartida entre niveles FPN.
    Condicionamiento: FiLM(CLS) + CrossAttn(patch tokens).
    """

    def __init__(
        self,
        in_channels:       int   = 256,
        num_convs:         int   = 4,
        num_classes:       int   = 22,
        query_dim:         int   = 768,   # dim CLS token (para FiLM)
        sketch_dim:        int   = 768,   # dim patch tokens (para CrossAttn)
        cross_attn_heads:  int   = 8,
        cross_attn_drop:   float = 0.0,
        strides:           list  = None,
        norm_on_bbox:      bool  = True,
        centerness_on_reg: bool  = True,
        cross_attn_start_level=1,
    ):
        super().__init__()
        self.num_classes       = num_classes
        self.strides           = strides or [4, 8, 16, 32, 64]
        self.norm_on_bbox      = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        num_levels = len(self.strides)

        # ── FiLM (condicionamiento global desde CLS) ──────────────────────────
        self.film_cls = nn.ModuleList([
            FiLMLayer(query_dim, in_channels) for _ in range(num_levels)
        ])
        self.film_reg = nn.ModuleList([
            FiLMLayer(query_dim, in_channels) for _ in range(num_levels)
        ])

        # ── CrossAttn (condicionamiento local desde patches) ──────────────────
        self.cross_attn_cls = nn.ModuleList([
            SketchCrossAttnLayer(in_channels, sketch_dim,
                                 cross_attn_heads, cross_attn_drop)
            for _ in range(num_levels)
        ])
        self.cross_attn_reg = nn.ModuleList([
            SketchCrossAttnLayer(in_channels, sketch_dim,
                                 cross_attn_heads, cross_attn_drop)
            for _ in range(num_levels)
        ])

        # ── Ramas convolucionales (pesos compartidos entre niveles) ───────────
        self.cls_convs = nn.Sequential(
            *[conv_block(in_channels, in_channels) for _ in range(num_convs)]
        )
        self.reg_convs = nn.Sequential(
            *[conv_block(in_channels, in_channels) for _ in range(num_convs)]
        )

        # ── Predictores de salida ─────────────────────────────────────────────
        self.cls_pred = nn.Conv2d(in_channels, num_classes,
                                  kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4,
                                  kernel_size=3, padding=1)
        self.ctr_pred = nn.Conv2d(in_channels, 1,
                                  kernel_size=3, padding=1)

        # Escala aprendible por nivel
        self.scales = nn.ModuleList([
            Scale(init_val=1.0) for _ in range(num_levels)
        ])

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Prior de probabilidad para cls: sigmoid(bias) ≈ 0.01
        prior_prob  = 0.01
        bias_value  = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    # ─────────────────────────────────────────────────────────────────────────
    def forward_single_level(
        self,
        feat:           torch.Tensor,   # [B, C, H, W]
        query_emb:      torch.Tensor,   # [B, D]        — CLS token
        sketch_patches: torch.Tensor,   # [B, P², D]    — patch tokens pooled
        level_idx:      int,
    ):
        """
        Procesa un nivel FPN con condicionamiento híbrido FiLM + CrossAttn.
        """
        # ── Rama clasificación ────────────────────────────────────────────────
        cls_feat = self.film_cls[level_idx](feat, query_emb)         # gate global
        cls_feat = self.cross_attn_cls[level_idx](cls_feat,
                                                   sketch_patches)   # selección local
        cls_feat   = self.cls_convs(cls_feat)
        cls_logits = self.cls_pred(cls_feat)

        # ── Rama regresión ────────────────────────────────────────────────────
        reg_feat  = self.film_reg[level_idx](feat, query_emb)
        reg_feat  = self.cross_attn_reg[level_idx](reg_feat, sketch_patches)
        reg_feat  = self.reg_convs(reg_feat)
        bbox_pred = self.scales[level_idx](self.reg_pred(reg_feat))
        bbox_pred = F.relu(bbox_pred)   # distancias ltrb ≥ 0

        # Centerness desde features de regresión
        ctr_logits = self.ctr_pred(reg_feat)

        return cls_logits, bbox_pred, ctr_logits

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        features:       list,           # [P2, P3, P4, P5, P6]
        query_emb:      torch.Tensor,   # [B, D]       — CLS token
        sketch_patches: torch.Tensor,   # [B, P², D]   — patch tokens pooled
    ):
        """
        Returns:
            all_cls   — lista [B, num_cls, Hi, Wi] por nivel
            all_bbox  — lista [B, 4, Hi, Wi] por nivel
            all_ctr   — lista [B, 1, Hi, Wi] por nivel
        """
        all_cls, all_bbox, all_ctr = [], [], []
        for i, feat in enumerate(features):
            cls_l, bbox_l, ctr_l = self.forward_single_level(
                feat, query_emb, sketch_patches, i
            )
            all_cls.append(cls_l)
            all_bbox.append(bbox_l)
            all_ctr.append(ctr_l)
        return all_cls, all_bbox, all_ctr

    # ─────────────────────────────────────────────────────────────────────────
    def decode_predictions(
        self,
        all_cls:         list,
        all_bbox:        list,
        all_ctr:         list,
        img_shape:       tuple,
        score_threshold: float = 0.05,
    ) -> list:
        """
        Convierte salidas raw a detecciones en coordenadas de imagen.
        Retorna lista de dicts {boxes, scores, labels} por imagen del batch.
        Sin cambios respecto a la versión anterior.
        """
        B = all_cls[0].shape[0]
        H_img, W_img = img_shape

        batch_boxes  = [[] for _ in range(B)]
        batch_scores = [[] for _ in range(B)]
        batch_labels = [[] for _ in range(B)]

        for lvl_idx, (cls_l, bbox_l, ctr_l) in enumerate(
            zip(all_cls, all_bbox, all_ctr)
        ):
            stride  = self.strides[lvl_idx]
            B, C, H, W = cls_l.shape

            ys = (torch.arange(H, device=cls_l.device).float() + 0.5) * stride
            xs = (torch.arange(W, device=cls_l.device).float() + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            points = torch.stack(
                [grid_x.flatten(), grid_y.flatten()], dim=-1
            )   # [H*W, 2]

            cls_scores = cls_l.sigmoid()
            ctr_scores = ctr_l.sigmoid()
            scores     = (cls_scores * ctr_scores).permute(0, 2, 3, 1)
            scores     = scores.reshape(B, H * W, C)

            bbox = bbox_l * stride if self.norm_on_bbox else bbox_l
            bbox = bbox.permute(0, 2, 3, 1).reshape(B, H * W, 4)

            for b in range(B):
                max_scores, max_labels = scores[b].max(dim=-1)
                keep_mask = max_scores > score_threshold
                if keep_mask.sum() == 0:
                    continue

                kp       = keep_mask.nonzero(as_tuple=False).squeeze(1)
                pts_k    = points[kp]
                bbox_k   = bbox[b][kp]
                scores_k = max_scores[kp]
                labels_k = max_labels[kp]

                x1 = (pts_k[:, 0] - bbox_k[:, 0]).clamp(0, W_img)
                y1 = (pts_k[:, 1] - bbox_k[:, 1]).clamp(0, H_img)
                x2 = (pts_k[:, 0] + bbox_k[:, 2]).clamp(0, W_img)
                y2 = (pts_k[:, 1] + bbox_k[:, 3]).clamp(0, H_img)
                boxes_k = torch.stack([x1, y1, x2, y2], dim=-1)

                batch_boxes[b].append(boxes_k)
                batch_scores[b].append(scores_k)
                batch_labels[b].append(labels_k)

        results = []
        for b in range(B):
            if batch_boxes[b]:
                boxes  = torch.cat(batch_boxes[b],  dim=0)
                scores = torch.cat(batch_scores[b], dim=0)
                labels = torch.cat(batch_labels[b], dim=0)
            else:
                device = all_cls[0].device
                boxes  = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,),   device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})

        return results
