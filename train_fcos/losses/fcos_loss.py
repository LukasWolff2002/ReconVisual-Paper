"""
losses/fcos_loss.py

Función de pérdida FCOS completa:
  L = λ_cls * L_focal  +  λ_bbox * L_giou  +  λ_ctr * L_centerness
    + λ_contrast * L_contrastive   ← NUEVO

Cambios respecto a la versión anterior:

  [CONTRAST] SketchContrastiveLoss:
    Pérdida auxiliar que empuja los embeddings CLS de sketches de clases
    distintas a separarse en el espacio latente, y los de la misma clase
    a aproximarse.

    Formulación: SupCon (Supervised Contrastive Loss, Khosla et al. 2020)
    adaptada para embeddings de sketch:

        L = -1/|P(i)| * Σ_{p∈P(i)} log [
              exp(sim(z_i, z_p) / τ)
              / Σ_{a∈A(i)} exp(sim(z_i, z_a) / τ)
            ]

    donde:
      z_i     = embedding CLS del sketch i (L2-normalizado)
      P(i)    = conjunto de muestras del mismo batch con la misma clase que i
      A(i)    = todos los demás embeddings del batch (positivos + negativos)
      sim     = similitud coseno
      τ       = temperatura (0.07 por defecto)

    Por qué ayuda:
      - Clases similares entre sí (e.g. "S" vs "s", "croix" vs "marqeur"):
        la pérdida las separa explícitamente en el espacio de embedding.
      - Clases con pocas instancias: el CLS token de sus sketches se acerca
        a otras instancias del mismo pool → prototipos más consistentes.
      - No requiere pares hardcoded: funciona con cualquier batch que contenga
        múltiples clases (lo que es siempre el caso con batch_size ≥ 4).

    La pérdida actúa sobre el cls_token que produce el SketchRefinementEncoder,
    por lo que los gradientes fluyen solo a través de los parámetros entrenables
    (SketchRefinementEncoder, FiLM), no al ViT frozen.

    Integración en train.py:
        loss_dict = loss_fn(all_cls, all_bbox, all_ctr, targets,
                            cls_tokens=cls_tokens, labels_for_contrast=batch_labels)
    Si cls_tokens es None (por compatibilidad con código antiguo), la pérdida
    contrastiva se omite silenciosamente.

Asignación de targets FCOS:
  1. Un punto (x, y) es positivo para GT box si está dentro de la box.
  2. La distancia max(l,t,r,b) debe caer dentro del rango de regresión del nivel.
  3. Si un punto cae en múltiples boxes, se asigna a la de menor área.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_fcos.utils.box_utils import compute_centerness_targets, box_giou


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def sigmoid_focal_loss(
    inputs:    torch.Tensor,
    targets:   torch.Tensor,
    alpha:     float = 0.25,
    gamma:     float = 2.0,
    reduction: str   = "sum",
) -> torch.Tensor:
    """Focal Loss estándar."""
    p       = inputs.sigmoid()
    p_t     = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss    = -alpha_t * (1 - p_t) ** gamma * \
              torch.log(p_t.clamp(min=1e-7))

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    return loss


# ─── [CONTRAST] Supervised Contrastive Loss ───────────────────────────────────

class SketchContrastiveLoss(nn.Module):
    """
    SupCon loss sobre embeddings CLS del sketch encoder.

    Separa clases similares en el espacio latente sin necesidad de pares
    hardcoded — funciona con cualquier batch que contenga ≥2 clases.

    Args:
        temperature:    τ de la pérdida contrastiva (0.07 recomendado).
        min_positives:  mínimo de positivos por ancla para calcular la pérdida.
                        Si un ancla no tiene suficientes positivos en el batch,
                        se omite (evita división por cero en batches pequeños).
    """

    def __init__(self, temperature: float = 0.07, min_positives: int = 1):
        super().__init__()
        self.temperature    = temperature
        self.min_positives  = min_positives

    def forward(
        self,
        cls_tokens: torch.Tensor,   # [B, D]  — embeddings CLS del sketch
        labels:     torch.Tensor,   # [B]     — class ids (int)
    ) -> torch.Tensor:
        """
        Calcula SupCon loss sobre el batch.

        Pasos:
          1. L2-normalizar todos los embeddings.
          2. Calcular matriz de similitudes coseno [B, B].
          3. Para cada ancla i, los positivos son los j con labels[j]==labels[i], j≠i.
          4. SupCon loss = promedio sobre anclas con al menos min_positives positivos.
        """
        B, D = cls_tokens.shape
        device = cls_tokens.device

        # L2-normalización
        z = F.normalize(cls_tokens, dim=-1)   # [B, D]

        # Matriz de similitudes coseno escalada por temperatura
        sim = torch.matmul(z, z.T) / self.temperature   # [B, B]

        # Máscara de identidad (excluir i==j)
        eye_mask = torch.eye(B, dtype=torch.bool, device=device)

        # Máscara de positivos: misma clase, distinto índice
        label_eq  = labels.unsqueeze(0) == labels.unsqueeze(1)   # [B, B]
        pos_mask  = label_eq & ~eye_mask

        # Si ningún ancla tiene positivos → batch monoclase, skip
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Denominador: todos los pares excepto i==j
        # Usamos log-sum-exp estable
        sim_no_diag = sim.masked_fill(eye_mask, float("-inf"))
        log_denom   = torch.logsumexp(sim_no_diag, dim=1, keepdim=True)  # [B, 1]

        # Numerador: similitudes con positivos
        # Para anclas sin positivos, contribución = 0
        n_pos = pos_mask.sum(dim=1).float().clamp(min=1)   # [B]

        # log-prob de cada par positivo: sim(i,j)/τ - log Σ_a sim(i,a)/τ
        log_prob_pos = sim - log_denom   # [B, B]  — broadcast

        # Promedio de log-probs sobre los positivos de cada ancla
        loss_per_anchor = -(log_prob_pos * pos_mask.float()).sum(dim=1) / n_pos  # [B]

        # Solo promediar anclas que tienen al menos min_positives positivos
        valid = pos_mask.sum(dim=1) >= self.min_positives
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss_per_anchor[valid].mean()


# ─── FCOS Loss ────────────────────────────────────────────────────────────────

class FCOSLoss(nn.Module):
    """
    Calcula la pérdida FCOS dados los outputs del head y los targets GT.
    Incluye opcionalmente la pérdida contrastiva sobre cls_tokens del sketch.
    """

    def __init__(
        self,
        num_classes:      int   = 22,
        strides:          list  = None,
        regress_ranges:   tuple = None,
        focal_alpha:      float = 0.25,
        focal_gamma:      float = 2.0,
        lambda_cls:       float = 1.0,
        lambda_bbox:      float = 1.0,
        lambda_ctr:       float = 1.0,
        lambda_contrast:  float = 0.1,   # peso de la pérdida contrastiva
        contrast_temp:    float = 0.07,  # temperatura SupCon
        norm_on_bbox:     bool  = True,
        class_weights:    torch.Tensor = None,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.strides         = strides      or [4, 8, 16, 32, 64]
        self.regress_ranges  = regress_ranges or (
            (0, 32), (32, 64), (64, 128), (128, 256), (256, 1e8)
        )
        self.focal_alpha     = focal_alpha
        self.focal_gamma     = focal_gamma
        self.lambda_cls      = lambda_cls
        self.lambda_bbox     = lambda_bbox
        self.lambda_ctr      = lambda_ctr
        self.lambda_contrast = lambda_contrast
        self.norm_on_bbox    = norm_on_bbox

        self.register_buffer("class_weights",
                             class_weights if class_weights is not None
                             else torch.ones(num_classes))

        # [CONTRAST] pérdida contrastiva auxiliar
        self.contrast_loss = SketchContrastiveLoss(temperature=contrast_temp)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_points(self, all_cls: list, device: torch.device) -> list:
        all_points = []
        for lvl, stride in enumerate(self.strides):
            _, _, H, W = all_cls[lvl].shape
            ys = (torch.arange(H, device=device).float() + 0.5) * stride
            xs = (torch.arange(W, device=device).float() + 0.5) * stride
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
            all_points.append(pts)
        return all_points

    # ─────────────────────────────────────────────────────────────────────────
    def _assign_targets(
        self,
        all_points: list,
        gt_boxes:   torch.Tensor,
        gt_labels:  torch.Tensor,
    ):
        points_all = torch.cat(all_points, dim=0)
        N = points_all.shape[0]
        M = gt_boxes.shape[0]

        if M == 0:
            return (
                torch.full((N,), self.num_classes, dtype=torch.long,
                           device=points_all.device),
                torch.zeros((N, 4), device=points_all.device),
            )

        lvl_ranges = []
        for lvl, pts in enumerate(all_points):
            lvl_ranges.append(torch.full((len(pts),), lvl,
                                         device=points_all.device))
        lvl_idx = torch.cat(lvl_ranges)

        x = points_all[:, 0][:, None]
        y = points_all[:, 1][:, None]
        x1, y1, x2, y2 = (gt_boxes[:, i][None, :] for i in range(4))

        l = x - x1
        t = y - y1
        r = x2 - x
        b = y2 - y

        ltrb       = torch.stack([l, t, r, b], dim=-1)
        inside_box = ltrb.min(dim=-1).values > 0
        max_ltrb   = ltrb.max(dim=-1).values

        for lvl, (r_min, r_max) in enumerate(self.regress_ranges):
            lvl_mask  = (lvl_idx == lvl)[:, None].expand_as(max_ltrb)
            in_range  = (max_ltrb >= r_min) & (max_ltrb <= r_max)
            inside_box = inside_box & (in_range | ~lvl_mask)

        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                   (gt_boxes[:, 3] - gt_boxes[:, 1])
        areas = gt_areas[None, :].expand(N, M).clone()
        areas[~inside_box] = float("inf")

        min_areas, min_idx = areas.min(dim=-1)
        assigned = min_areas < float("inf")

        cls_targets  = torch.full((N,), self.num_classes, dtype=torch.long,
                                   device=points_all.device)
        bbox_targets = torch.zeros((N, 4), device=points_all.device)

        if assigned.sum() > 0:
            cls_targets[assigned]  = gt_labels[min_idx[assigned]]
            bbox_targets[assigned] = ltrb[assigned, min_idx[assigned]]

        return cls_targets, bbox_targets

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        all_cls:   list,
        all_bbox:  list,
        all_ctr:   list,
        targets:   list,
        # [CONTRAST] embeddings CLS del sketch y labels de clase para el batch
        # Si None, la pérdida contrastiva se omite (backward compatible)
        cls_tokens:           torch.Tensor = None,   # [B, D]
        labels_for_contrast:  torch.Tensor = None,   # [B]  — class indices
    ) -> dict:

        device = all_cls[0].device
        B      = all_cls[0].shape[0]

        all_points = self._get_points(all_cls, device)

        flat_cls  = torch.cat(
            [c.permute(0,2,3,1).reshape(B, -1, self.num_classes) for c in all_cls], dim=1
        )
        flat_bbox = torch.cat(
            [b.permute(0,2,3,1).reshape(B, -1, 4) for b in all_bbox], dim=1
        )
        flat_ctr  = torch.cat(
            [c.permute(0,2,3,1).reshape(B, -1, 1) for c in all_ctr], dim=1
        )

        if self.norm_on_bbox:
            strides_tensor = []
            for lvl, pts in enumerate(all_points):
                strides_tensor.append(
                    torch.full((len(pts),), self.strides[lvl], device=device)
                )
            strides_per_point = torch.cat(strides_tensor)[None, :, None]
            flat_bbox = flat_bbox * strides_per_point

        loss_cls_total  = torch.tensor(0.0, device=device)
        loss_bbox_total = torch.tensor(0.0, device=device)
        loss_ctr_total  = torch.tensor(0.0, device=device)
        n_pos_total     = 0

        for b in range(B):
            gt_boxes  = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)

            cls_t, bbox_t = self._assign_targets(all_points, gt_boxes, gt_labels)

            pos_mask = cls_t < self.num_classes
            n_pos    = pos_mask.sum().item()
            n_pos_total += max(n_pos, 1)

            cls_logits = flat_cls[b]
            cls_oh     = F.one_hot(
                cls_t.clamp(0, self.num_classes - 1),
                num_classes=self.num_classes
            ).float()
            cls_oh[~pos_mask] = 0.0

            loss_cls = sigmoid_focal_loss(
                cls_logits, cls_oh,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction="sum"
            )
            loss_cls_total += loss_cls

            if n_pos == 0:
                continue

            pred_ltrb = flat_bbox[b][pos_mask]
            tgt_ltrb  = bbox_t[pos_mask]

            points_pos = torch.cat(all_points, dim=0)[pos_mask]
            pred_x1 = (points_pos[:, 0] - pred_ltrb[:, 0]).clamp(min=0)
            pred_y1 = (points_pos[:, 1] - pred_ltrb[:, 1]).clamp(min=0)
            pred_x2 = (points_pos[:, 0] + pred_ltrb[:, 2])
            pred_y2 = (points_pos[:, 1] + pred_ltrb[:, 3])
            pred_boxes_xy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

            tgt_x1 = points_pos[:, 0] - tgt_ltrb[:, 0]
            tgt_y1 = points_pos[:, 1] - tgt_ltrb[:, 1]
            tgt_x2 = points_pos[:, 0] + tgt_ltrb[:, 2]
            tgt_y2 = points_pos[:, 1] + tgt_ltrb[:, 3]
            tgt_boxes_xy = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=-1)

            giou = box_giou(pred_boxes_xy, tgt_boxes_xy)
            loss_bbox_total += (1 - giou).sum()

            pred_ctr = flat_ctr[b][pos_mask].squeeze(-1)
            tgt_ctr  = compute_centerness_targets(tgt_ltrb)
            loss_ctr_total += F.binary_cross_entropy_with_logits(
                pred_ctr, tgt_ctr, reduction="sum"
            )

        denom     = max(n_pos_total, 1)
        loss_cls  = self.lambda_cls  * loss_cls_total  / denom
        loss_bbox = self.lambda_bbox * loss_bbox_total / denom
        loss_ctr  = self.lambda_ctr  * loss_ctr_total  / denom

        # ── [CONTRAST] Pérdida contrastiva auxiliar ───────────────────────────
        loss_contrast = torch.tensor(0.0, device=device)
        if (cls_tokens is not None and
                labels_for_contrast is not None and
                self.lambda_contrast > 0):
            loss_contrast = self.lambda_contrast * self.contrast_loss(
                cls_tokens, labels_for_contrast
            )

        total = loss_cls + loss_bbox + loss_ctr + loss_contrast

        return {
            "loss":          total,
            "loss_cls":      loss_cls,
            "loss_bbox":     loss_bbox,
            "loss_ctr":      loss_ctr,
            "loss_contrast": loss_contrast,
            "n_pos":         n_pos_total / B,
        }
