"""
models/detector.py

Detector completo. API externa sin cambios:
    forward(page_imgs, query_imgs)  → (all_cls, all_bbox, all_ctr)
    predict(page_imgs, query_imgs, img_shape) → lista de dicts

Cambios respecto a la versión anterior:
  [ENC]   forward_with_embeddings() expone cls_tokens para la pérdida contrastiva.
  [MQ]    predict_multi_query(): fusiona N sketches promediando embeddings.
  [NMS-1] context_aware_nms en lugar de batch_nms.
  [NMS-2] Adaptive score threshold por imagen.
  [ARCH]  cross_attn_start_level soportado en FCOSHead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_fcos.models.backbone  import iDocBackbone, iDocQueryEncoder
from train_fcos.models.fpn       import FPN
from train_fcos.models.fcos_head import FCOSHead
from train_fcos.utils.box_utils  import (
    context_aware_nms,
    adaptive_score_threshold,
    clip_boxes_to_image,
)


class FCOSDetector(nn.Module):
    """
    Detector FCOS condicionado por query sketch.
    Backbone iDoc + FPN + FCOSHead (FiLM + co-attention bidireccional).

    Parámetros entrenables:
        FPN, FCOSHead (FiLM, CrossAttn, convs, predictors, scales)
        iDocBackbone.level_norms
        iDocQueryEncoder.refinement (SketchRefinementEncoder)

    Parámetros frozen:
        iDocBackbone.vit  (ViT-Base completo)
    """

    def __init__(self, config: dict):
        super().__init__()

        bb_cfg   = config["BACKBONE"]
        fpn_cfg  = config["FPN"]
        head_cfg = config["FCOS_HEAD"]
        qe_cfg   = config["QUERY_ENCODER"]

        self.backbone = iDocBackbone(
            arch            = bb_cfg["arch"],
            patch_size      = bb_cfg.get("patch_size",     16),
            embed_dim       = bb_cfg.get("embed_dim",     768),
            depth           = bb_cfg.get("depth",          12),
            num_heads       = bb_cfg.get("num_heads",      12),
            extract_layers  = bb_cfg.get("extract_layers", [2, 5, 8, 11]),
            pretrained_path = config.get("PRETRAINED_PTH"),
        )

        self.query_encoder = iDocQueryEncoder(
            backbone        = self.backbone,
            query_dim       = bb_cfg.get("embed_dim", 768),
            patch_pool_size = qe_cfg.get("sketch_pool_size", 7),
            use_refinement  = qe_cfg.get("use_refinement",  True),
            refine_layers   = qe_cfg.get("refine_layers",   2),
            refine_heads    = qe_cfg.get("refine_heads",    8),
            refine_ffn_dim  = qe_cfg.get("refine_ffn_dim",  2048),
            refine_dropout  = qe_cfg.get("refine_dropout",  0.1),
        )

        self.fpn = FPN(
            in_channels  = fpn_cfg["in_channels"],
            out_channels = fpn_cfg["out_channels"],
        )

        self.head = FCOSHead(
            in_channels            = fpn_cfg["out_channels"],
            num_convs              = head_cfg["num_convs"],
            num_classes            = head_cfg["num_classes"],
            query_dim              = qe_cfg["embed_dim"],
            sketch_dim             = bb_cfg.get("embed_dim", 768),
            cross_attn_heads       = head_cfg.get("cross_attn_heads", 8),
            cross_attn_drop        = head_cfg.get("cross_attn_drop",  0.0),
            strides                = head_cfg["strides"],
            norm_on_bbox           = head_cfg["norm_on_bbox"],
            centerness_on_reg      = head_cfg["centerness_on_reg"],
            cross_attn_start_level = head_cfg.get("cross_attn_start_level", 1),
        )

        self.eval_cfg = config.get("EVAL", {})

    # ─────────────────────────────────────────────────────────────────────────
    def _encode_query(self, query_imgs: torch.Tensor):
        """Codifica sketches. ViT con no_grad; SketchRefinementEncoder con grad."""
        return self.query_encoder(query_imgs)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, page_imgs: torch.Tensor, query_imgs: torch.Tensor):
        """API pública sin cambios. Returns: (all_cls, all_bbox, all_ctr)"""
        cls_token, sketch_patches = self._encode_query(query_imgs)
        backbone_feats = self.backbone(page_imgs)
        fpn_feats      = self.fpn(backbone_feats)
        return self.head(fpn_feats, cls_token, sketch_patches)

    # ─────────────────────────────────────────────────────────────────────────
    def forward_with_embeddings(self, page_imgs: torch.Tensor, query_imgs: torch.Tensor):
        """
        Como forward() pero también retorna cls_tokens para la pérdida contrastiva.
        Returns: all_cls, all_bbox, all_ctr, cls_tokens [B, D]
        """
        cls_token, sketch_patches = self._encode_query(query_imgs)
        backbone_feats = self.backbone(page_imgs)
        fpn_feats      = self.fpn(backbone_feats)
        all_cls, all_bbox, all_ctr = self.head(fpn_feats, cls_token, sketch_patches)
        return all_cls, all_bbox, all_ctr, cls_token

    # ─────────────────────────────────────────────────────────────────────────
    def _decode_and_nms(self, all_cls, all_bbox, all_ctr, img_shape):
        """Adaptive threshold + context-aware NMS. Retorna lista de dicts."""
        base_score_thr   = self.eval_cfg.get("score_threshold",    0.05)
        nms_thr          = self.eval_cfg.get("nms_iou_thresh",     0.5)
        max_dets         = self.eval_cfg.get("max_dets",           100)
        density_trigger  = self.eval_cfg.get("density_trigger",    50)
        adaptive_pct     = self.eval_cfg.get("adaptive_percentile", 0.90)
        cluster_iou_thr  = self.eval_cfg.get("cluster_iou_thr",   0.20)
        cluster_min_det  = self.eval_cfg.get("cluster_min_det",   4)
        score_spread_thr = self.eval_cfg.get("score_spread_thr",  0.15)

        B = all_cls[0].shape[0]

        score_thrs = []
        for b in range(B):
            raw = torch.cat([
                (all_cls[l][b].sigmoid() * all_ctr[l][b].sigmoid())
                .max(dim=0).values.flatten()
                for l in range(len(all_cls))
            ])
            score_thrs.append(adaptive_score_threshold(
                raw, base_score_thr, density_trigger, adaptive_pct
            ))

        final_dets = []
        for b in range(B):
            cls_b  = [c[b:b+1] for c in all_cls]
            bbox_b = [bx[b:b+1] for bx in all_bbox]
            ctr_b  = [ct[b:b+1] for ct in all_ctr]

            raw = self.head.decode_predictions(
                cls_b, bbox_b, ctr_b, img_shape, score_thrs[b]
            )
            det = raw[0]

            if len(det["boxes"]) == 0:
                final_dets.append(det)
                continue

            keep = context_aware_nms(
                det["boxes"], det["scores"], det["labels"],
                iou_threshold    = nms_thr,
                cluster_iou_thr  = cluster_iou_thr,
                cluster_min_det  = cluster_min_det,
                score_spread_thr = score_spread_thr,
            )

            if len(keep) > max_dets:
                _, si = det["scores"][keep].sort(descending=True)
                keep  = keep[si[:max_dets]]

            final_dets.append({
                "boxes":  clip_boxes_to_image(det["boxes"][keep],  img_shape),
                "scores": det["scores"][keep],
                "labels": det["labels"][keep],
            })

        return final_dets

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, page_imgs, query_imgs, img_shape=None) -> list:
        """Inferencia estándar (un sketch por imagen)."""
        self.eval()
        all_cls, all_bbox, all_ctr = self.forward(page_imgs, query_imgs)
        if img_shape is None:
            img_shape = (page_imgs.shape[-2], page_imgs.shape[-1])
        return self._decode_and_nms(all_cls, all_bbox, all_ctr, img_shape)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_multi_query(
        self,
        page_imgs:     torch.Tensor,
        query_tensors: list,           # lista de tensores [3, 224, 224]
        img_shape:     tuple = None,
    ) -> list:
        """
        [MQ] Inferencia con multi-query fusion.

        Promedia los embeddings CLS y patches de N sketches de la misma clase.
        El prototipo resultante es más estable que un único sketch.

        Uso típico:
            tensors = dataset.load_multi_query(class_id, n_queries=3)
            dets = model.predict_multi_query(page_imgs, tensors)
        """
        self.eval()
        if not query_tensors:
            raise ValueError("predict_multi_query requiere al menos un sketch.")
        if img_shape is None:
            img_shape = (page_imgs.shape[-2], page_imgs.shape[-1])

        device = page_imgs.device
        B      = page_imgs.shape[0]

        all_cls_tokens    = []
        all_sketch_patches = []
        for qt in query_tensors:
            q = qt.unsqueeze(0).to(device).expand(B, -1, -1, -1)
            cls_t, patches_t = self._encode_query(q)
            all_cls_tokens.append(cls_t)
            all_sketch_patches.append(patches_t)

        cls_proto     = torch.stack(all_cls_tokens,    dim=0).mean(dim=0)  # [B, D]
        patches_proto = torch.stack(all_sketch_patches, dim=0).mean(dim=0) # [B, P², D]

        backbone_feats = self.backbone(page_imgs)
        fpn_feats      = self.fpn(backbone_feats)
        all_cls, all_bbox, all_ctr = self.head(fpn_feats, cls_proto, patches_proto)

        return self._decode_and_nms(all_cls, all_bbox, all_ctr, img_shape)

    # ─────────────────────────────────────────────────────────────────────────
    def get_trainable_params(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]
