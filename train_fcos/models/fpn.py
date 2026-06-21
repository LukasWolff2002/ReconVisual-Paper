"""
models/fpn.py

Feature Pyramid Network (FPN) entrenable sobre las features del backbone congelado.
Produce P2-P6 con 256 canales en cada nivel.

Arquitectura:
  C2 (128ch) → lateral conv → P2 (256ch)
  C3 (256ch) → lateral conv → P3 (256ch)
  C4 (512ch) → lateral conv → P4 (256ch)
  C5 (1024ch)→ lateral conv → P5 (256ch)
  P5 → stride-2 → P6 (256ch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    FPN estándar con top-down pathway.
    Todos los niveles se proyectan a out_channels=256.
    """

    def __init__(self, in_channels: list, out_channels: int = 256):
        """
        in_channels: [C2_ch, C3_ch, C4_ch, C5_ch] = [128, 256, 512, 1024]
        out_channels: dimensión de salida de cada nivel (256)
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral convolutions: 1x1 para proyectar a out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])

        # Output convolutions: 3x3 para suavizar artifacts del upsampling
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

        # P6 generado desde P5 con stride-2 (para objetos muy grandes)
        self.p6_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: list) -> list:
        """
        features: [C2, C3, C4, C5] con shapes [B, Ci, Hi, Wi]
        Returns:  [P2, P3, P4, P5, P6]
        """
        # Proyecciones laterales
        laterals = [conv(feat) for conv, feat in
                    zip(self.lateral_convs, features)]

        # Top-down pathway: fusión desde el nivel más grueso (C5) al más fino (C2)
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample el nivel superior y sumarlo al lateral inferior
            target_size = laterals[i].shape[-2:]
            up = F.interpolate(laterals[i + 1], size=target_size,
                               mode="nearest")
            laterals[i] = laterals[i] + up

        # Aplicar output convs (3x3) para cada nivel
        fpn_outs = [conv(lat) for conv, lat in
                    zip(self.output_convs, laterals)]
        # fpn_outs = [P2, P3, P4, P5]

        # P6: stride-2 desde P5
        p6 = self.p6_conv(fpn_outs[-1])
        fpn_outs.append(p6)

        # Retorna [P2, P3, P4, P5, P6]
        return fpn_outs
