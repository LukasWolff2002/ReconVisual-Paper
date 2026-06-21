"""
utils/box_utils.py

Operaciones geométricas sobre bounding boxes.
Formato esperado: [x1, y1, x2, y2] (xyxy).

Mejoras respecto a la versión anterior:
  [NMS-1] context_aware_nms():
            Post-NMS que detecta zonas de activación por textura repetitiva:
            si hay muchas detecciones de score similar en un área compacta,
            suprime las de score más bajo.  Sin reentrenamiento.
  [NMS-2] adaptive_score_threshold():
            Threshold por imagen: en páginas con muchos candidatos usa
            el percentil 90 de scores como piso mínimo.  Reduce ruido
            en páginas densas sin afectar páginas simples.
"""

import torch
import torch.nn.functional as F


# ─── Primitivas ───────────────────────────────────────────────────────────────

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Área de cada box. boxes: [N, 4] en formato xyxy."""
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
           (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU entre dos conjuntos de boxes.
    boxes1: [N, 4], boxes2: [M, 4]
    Returns: [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * \
            (inter_y2 - inter_y1).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU entre pares de boxes (N == M).
    boxes1: [N, 4], boxes2: [N, 4]
    Returns: [N]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * \
            (inter_y2 - inter_y1).clamp(min=0)

    union = area1 + area2 - inter
    iou   = inter / union.clamp(min=1e-6)

    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * \
               (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def compute_centerness_targets(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Calcula el target de centerness desde las distancias l, t, r, b.
    ltrb: [N, 4] con (left, top, right, bottom)
    Returns: [N]
    """
    l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
    centerness = torch.sqrt(
        (torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)) *
        (torch.min(t, b) / torch.max(t, b).clamp(min=1e-6))
    )
    return centerness.clamp(min=0, max=1)


def ltrb_to_xyxy(ltrb: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Convierte predicciones FCOS (l, t, r, b) relativas a puntos a formato xyxy.
    ltrb:   [N, 4]
    points: [N, 2] con (x, y) del centro de cada celda
    Returns: [N, 4] en xyxy
    """
    x1 = points[:, 0] - ltrb[:, 0]
    y1 = points[:, 1] - ltrb[:, 1]
    x2 = points[:, 0] + ltrb[:, 2]
    y2 = points[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_ltrb(boxes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Convierte boxes xyxy a distancias (l, t, r, b) desde los puntos.
    boxes:  [N, 4]
    points: [N, 2]
    Returns: [N, 4]
    """
    l = points[:, 0] - boxes[:, 0]
    t = points[:, 1] - boxes[:, 1]
    r = boxes[:, 2] - points[:, 0]
    b = boxes[:, 3] - points[:, 1]
    return torch.stack([l, t, r, b], dim=-1)


# ─── NMS estándar ─────────────────────────────────────────────────────────────

def batch_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
              iou_threshold: float = 0.5) -> torch.Tensor:
    """
    NMS por clase (class-aware NMS).
    boxes:  [N, 4], scores: [N], labels: [N]
    Returns: índices kept [K]
    """
    from torchvision.ops import nms
    keep_all = []
    unique_labels = labels.unique()
    for cls in unique_labels:
        mask = labels == cls
        idx  = mask.nonzero(as_tuple=False).squeeze(1)
        kept = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.append(idx[kept])
    if not keep_all:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    return torch.cat(keep_all)


# ─── [NMS-2] Adaptive score threshold ────────────────────────────────────────

def adaptive_score_threshold(
    scores:           torch.Tensor,
    base_threshold:   float = 0.05,
    density_trigger:  int   = 50,
    percentile:       float = 0.90,
) -> float:
    """
    Calcula un threshold adaptivo por imagen.

    Lógica:
      - Si hay pocas detecciones candidatas (<= density_trigger), usa base_threshold.
      - Si hay muchas (página densa), usa max(base_threshold, percentil_90 de scores).
        Esto elimina la larga cola de activaciones débiles por textura/decoración.

    Args:
        scores:          tensor de scores pre-NMS [N]
        base_threshold:  threshold mínimo absoluto (0.05 por defecto)
        density_trigger: número de candidatos a partir del cual se activa el modo
                         adaptivo (50 por defecto)
        percentile:      percentil usado como threshold adaptivo (0.90 = top 10%)

    Returns:
        float — threshold a aplicar antes del NMS
    """
    if len(scores) == 0:
        return base_threshold

    if len(scores) <= density_trigger:
        return base_threshold

    # Umbral adaptivo: percentil de los scores disponibles
    thr_adaptive = float(torch.quantile(scores, percentile))
    return max(base_threshold, thr_adaptive)


# ─── [NMS-1] Context-aware NMS ────────────────────────────────────────────────

def context_aware_nms(
    boxes:           torch.Tensor,
    scores:          torch.Tensor,
    labels:          torch.Tensor,
    iou_threshold:   float = 0.5,
    cluster_iou_thr: float = 0.20,
    cluster_min_det: int   = 4,
    score_spread_thr: float = 0.15,
) -> torch.Tensor:
    """
    NMS por clase seguido de supresión de clusters de textura.

    Un "cluster de textura" es un grupo de detecciones que:
      1. Se solapan entre sí más de cluster_iou_thr (zona compacta)
      2. Tienen scores muy similares (spread < score_spread_thr) → no hay
         un candidato claramente dominante, señal de activación por patrón
         repetitivo en lugar de objeto real
      3. Contienen al menos cluster_min_det detecciones

    En ese caso, sólo se conserva la detección de mayor score del cluster.

    Nota: este paso se aplica DESPUÉS del NMS estándar, sobre las detecciones
    supervivientes.

    Args:
        boxes:            [N, 4] xyxy tras NMS estándar
        scores:           [N]
        labels:           [N]
        iou_threshold:    IoU para el NMS estándar por clase
        cluster_iou_thr:  IoU mínima para considerar que dos dets están en
                          el mismo cluster (más bajo que NMS: detecta vecindad)
        cluster_min_det:  mínimo de dets en el cluster para aplicar supresión
        score_spread_thr: rango max(scores) - min(scores) dentro del cluster;
                          si es menor que este valor, el cluster se considera
                          de textura y se suprime

    Returns:
        índices kept [K] sobre el tensor de entrada (post-NMS estándar)
    """
    # Primero: NMS estándar por clase
    keep_nms = batch_nms(boxes, scores, labels, iou_threshold)
    if len(keep_nms) == 0:
        return keep_nms

    boxes_k  = boxes[keep_nms]
    scores_k = scores[keep_nms]
    labels_k = labels[keep_nms]
    N        = len(keep_nms)

    if N < cluster_min_det:
        # Pocas detecciones → no hay riesgo de cluster de textura
        return keep_nms

    # Calcular IoU entre todas las detecciones supervivientes
    iou_mat = box_iou(boxes_k, boxes_k)   # [N, N]

    suppressed = torch.zeros(N, dtype=torch.bool, device=boxes.device)

    # Orden por score descendente para procesar los mejores primero
    order = scores_k.argsort(descending=True).tolist()

    for i in order:
        if suppressed[i]:
            continue

        # Encontrar vecinos con solapamiento > cluster_iou_thr
        neighbors = (iou_mat[i] > cluster_iou_thr).nonzero(as_tuple=False).squeeze(1)
        neighbors = [j.item() for j in neighbors
                     if not suppressed[j.item()] and j.item() != i]

        if len(neighbors) + 1 < cluster_min_det:
            # Cluster demasiado pequeño → no es textura repetitiva
            continue

        cluster_idx   = [i] + neighbors
        cluster_scores = scores_k[cluster_idx]

        score_spread = float(cluster_scores.max() - cluster_scores.min())
        if score_spread >= score_spread_thr:
            # Hay un candidato claramente mejor → no es cluster de textura
            continue

        # Cluster de textura: suprimir todos excepto el de mayor score
        best_in_cluster = cluster_idx[int(cluster_scores.argmax())]
        for j in cluster_idx:
            if j != best_in_cluster:
                suppressed[j] = True

    # Reconstruir índices en el tensor original
    final_mask = ~suppressed
    return keep_nms[final_mask.nonzero(as_tuple=False).squeeze(1)]


def clip_boxes_to_image(boxes: torch.Tensor, size: tuple) -> torch.Tensor:
    """Clipa boxes al tamaño de imagen (H, W)."""
    H, W = size
    boxes[:, 0].clamp_(min=0, max=W)
    boxes[:, 1].clamp_(min=0, max=H)
    boxes[:, 2].clamp_(min=0, max=W)
    boxes[:, 3].clamp_(min=0, max=H)
    return boxes