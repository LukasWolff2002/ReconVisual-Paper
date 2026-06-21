"""
models/cross_attn.py

Atención cruzada sketch ↔ imagen para la cabeza FCOS.

Versión anterior: unidireccional — la imagen atiende al sketch, pero el sketch
es estático (no se adapta a lo que ve en la imagen).

Versión nueva: co-attention bidireccional en dos pasos:

  Paso 1 — sketch → imagen  (igual que antes, "¿qué patches del sketch
            coinciden con esta región?")
  Paso 2 — imagen → sketch  (NUEVO: "¿qué regiones de la imagen son más
            relevantes para cada patch del sketch?")

  El resultado del paso 2 actualiza los patch tokens del sketch con contexto
  de imagen. En el paso 1, el sketch ya sabe qué buscar en este documento
  concreto, no solo en general.

Por qué importa:
  - Clases con apariencia similar: el sketch "se diferencia" de la clase vecina
    usando contexto de la imagen actual (e.g. el entorno de decoración).
  - Falsos positivos contextuales: si el sketch actualizado no "reconoce"
    ninguna región, la atención imagen→sketch produce señal difusa → scores bajos.

Inicialización identity:
  Ambas direcciones arrancan con out_proj=0 → identidad perfecta al inicio.
  Compatible con checkpoints existentes de SketchCrossAttnLayer (mismo nombre
  de clase, misma interfaz pública forward(x, sketch_patches)).

Complejidad adicional respecto a la versión unidireccional:
  - Un segundo MultiheadAttention (mismo tamaño que el primero)
  - Una proyección sketch_back_proj: feat_channels → sketch_dim  (para devolver
    los patches actualizados al espacio original 768 del ViT)
  - Los patches actualizados se usan solo dentro de esta capa; la interfaz
    externa no cambia (sketch_patches originales se siguen pasando desde fuera).
"""

import torch
import torch.nn as nn


class SketchCrossAttnLayer(nn.Module):
    """
    Co-attention bidireccional: imagen ↔ sketch.

    Interfaz pública idéntica a la versión anterior:
        forward(x, sketch_patches) → [B, C, H, W]

    Args:
        feat_channels:  canales del feature map FPN (256).
        sketch_dim:     dimensión de los patch tokens iDoc (768).
        num_heads:      cabezas de atención (8).
        dropout:        dropout en ambas direcciones de atención.
        bidirectional:  si False, degrada a la versión unidireccional original
                        (útil para ablación o para niveles con memory pressure).
    """

    def __init__(
        self,
        feat_channels: int   = 256,
        sketch_dim:    int   = 768,
        num_heads:     int   = 8,
        dropout:       float = 0.0,
        bidirectional: bool  = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        assert feat_channels % num_heads == 0, (
            f"feat_channels ({feat_channels}) debe ser divisible "
            f"entre num_heads ({num_heads})."
        )

        # ── Proyecciones del sketch al espacio de feat_channels ───────────────
        # sketch_dim (768) → feat_channels (256)
        self.sketch_proj = nn.Linear(sketch_dim, feat_channels, bias=False)

        # ── Paso 1: imagen atiende al sketch (imagen como Q, sketch como K/V) ─
        self.img_attn_sketch = nn.MultiheadAttention(
            embed_dim   = feat_channels,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm_img = nn.LayerNorm(feat_channels)

        # ── Paso 2 (bidireccional): sketch atiende a la imagen ────────────────
        if bidirectional:
            # sketch como Q, imagen como K/V — en espacio feat_channels
            self.sketch_attn_img = nn.MultiheadAttention(
                embed_dim   = feat_channels,
                num_heads   = num_heads,
                dropout     = dropout,
                batch_first = True,
            )
            self.norm_sketch = nn.LayerNorm(feat_channels)

            # Proyección de vuelta: feat_channels → sketch_dim
            # para que los patches actualizados tengan la dim original del ViT
            self.sketch_back_proj = nn.Linear(feat_channels, sketch_dim, bias=False)

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        nn.init.xavier_uniform_(self.sketch_proj.weight)

        # Paso 1: arranca como identidad
        nn.init.zeros_(self.img_attn_sketch.out_proj.weight)
        nn.init.zeros_(self.img_attn_sketch.out_proj.bias)

        if self.bidirectional:
            # Paso 2: arranca como identidad
            nn.init.zeros_(self.sketch_attn_img.out_proj.weight)
            nn.init.zeros_(self.sketch_attn_img.out_proj.bias)
            # back_proj: también zeros → al inicio no modifica nada
            nn.init.zeros_(self.sketch_back_proj.weight)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x:              torch.Tensor,   # [B, C, H, W]   feature map FPN
        sketch_patches: torch.Tensor,   # [B, N_s, D]    patch tokens del sketch
    ) -> torch.Tensor:                  # [B, C, H, W]
        B, C, H, W = x.shape

        # Aplanar imagen → secuencia [B, HW, C]
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Proyectar patches al espacio feat_channels: [B, N_s, C]
        kv_sketch = self.sketch_proj(sketch_patches)

        # ── Paso 2 (primero en orden temporal): sketch atiende a la imagen ───
        # El sketch "mira" la imagen y actualiza sus tokens antes de que la
        # imagen los use. Así el sketch sabe qué hay en este documento.
        if self.bidirectional:
            # sketch como Q [B, N_s, C], imagen como K/V [B, HW, C]
            sketch_update, _ = self.sketch_attn_img(
                query = kv_sketch,
                key   = x_seq,
                value = x_seq,
            )   # [B, N_s, C]

            # Residual + norma en espacio feat_channels
            kv_sketch_refined = self.norm_sketch(kv_sketch + sketch_update)

            # Proyección de vuelta a sketch_dim (opcional, para uso externo)
            # Aquí sólo la usamos internamente en el paso 1
            # kv_sketch_for_img = kv_sketch_refined  (ya en feat_channels)
        else:
            kv_sketch_refined = kv_sketch

        # ── Paso 1: imagen atiende al sketch (refinado) ───────────────────────
        img_update, _ = self.img_attn_sketch(
            query = x_seq,
            key   = kv_sketch_refined,
            value = kv_sketch_refined,
        )   # [B, HW, C]

        # Residual + norma
        out = self.norm_img(x_seq + img_update)   # [B, HW, C]

        # Restaurar forma espacial
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]
