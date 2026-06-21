"""
backbone.py

Backbone ViT-Base congelado para detección multi-escala.
Auto-detecta el tipo de checkpoint al cargar:
  · iDoc  (pos_embed / LoRA)      → usa VisionTransformer de iDoc
  · DINOv3 (rope_embed.periods)   → usa DINOv3ViT con RoPE 2D + register tokens

Fix DataParallel:
  DataParallel._replicate_for_data_parallel hace shallow-copy del __dict__,
  por lo que cualquier backreference (_vit) en módulos hijo apunta al ViT
  ORIGINAL, no a la réplica. h/w/rope/n_pre se pasan explícitamente en cada
  forward para evitar estado mutable compartido entre réplicas.
"""

import sys, os, math, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── iDoc module loader ───────────────────────────────────────────────────────

IDOC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "iDoc")
)

def _load_idoc_module(filename, module_name):
    path = os.path.join(IDOC_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró {filename} en {IDOC_DIR}")
    if IDOC_DIR not in sys.path:
        sys.path.insert(0, IDOC_DIR)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec   = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

_vit_module       = _load_idoc_module("models/vision_transformer.py", "idoc_vision_transformer")
VisionTransformer = _vit_module.VisionTransformer


# ─── LoRA fusion (iDoc) ───────────────────────────────────────────────────────

def _merge_lora_weights(state: dict, embed_dim: int = 768, depth: int = 12) -> dict:
    """Fusiona pesos LoRA al formato estándar del ViT: W = W_base + B @ A."""
    new_state = {}
    for k, v in state.items():
        if any(s in k for s in ["q_proj", "k_proj", "v_proj", "out_proj",
                                 "w_lora_A", "w_lora_B"]):
            continue
        new_state[k] = v

    n_merged = 0
    for i in range(depth):
        pref = f"blocks.{i}.attn"
        if f"{pref}.q_proj.weight" not in state:
            continue
        merged_w, merged_b = [], []
        for proj in ["q_proj", "k_proj", "v_proj"]:
            W  = state[f"{pref}.{proj}.weight"]
            b  = state.get(f"{pref}.{proj}.bias", torch.zeros(embed_dim))
            if f"{pref}.{proj}.w_lora_A" in state:
                W = W + state[f"{pref}.{proj}.w_lora_B"] @ state[f"{pref}.{proj}.w_lora_A"]
            merged_w.append(W); merged_b.append(b)
        new_state[f"{pref}.qkv.weight"] = torch.cat(merged_w, dim=0)
        new_state[f"{pref}.qkv.bias"]   = torch.cat(merged_b,  dim=0)
        n_merged += 1
        for name in ["out_proj", "proj"]:
            wk = f"{pref}.{name}.weight"
            if wk in state:
                new_state[f"{pref}.proj.weight"] = state[wk]
                new_state[f"{pref}.proj.bias"]   = state.get(
                    f"{pref}.{name}.bias", torch.zeros(embed_dim))
                break

    if n_merged:
        print(f"[iDocBackbone] LoRA fusionado en {n_merged} bloques.")
    return new_state


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv3 ViT con RoPE 2D Axial
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Conv2d patch embedding compatible con nombres del checkpoint DINOv3."""
    def __init__(self, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)   # [B, N, D]


class Axial2DRoPE(nn.Module):
    """
    RoPE 2D axial con periodos aprendibles (cargados del checkpoint DINOv3).

    Interpretación de periods [n_periods]:
      · Se divide en mitad-x (P_x = n//2) y mitad-y (P_y = n - n//2).
      · Cada periodo i define un par de dimensiones de la cabeza:
            dims rotadas = 2*P_x + 2*P_y = 2*n_periods = rope_dim
        (para ViT-B/16 con n_periods=16 → rope_dim=32 de 64 dims por cabeza)
      · Las dims restantes [rope_dim:head_dim] no se rotan.

    API: apply(q, k, h, w, n_pre) — sin estado interno, DataParallel-safe.
    """

    def __init__(self, n_periods: int, head_dim: int):
        super().__init__()
        self.n_periods = n_periods
        self.rope_dim  = 2 * n_periods
        self.register_parameter("periods",
                                nn.Parameter(torch.ones(n_periods)))

    @staticmethod
    def _apply_1d_pairs(
        x:     torch.Tensor,   # [B, H, N, 2*P]
        pos:   torch.Tensor,   # [N]  posiciones 1D
        freqs: torch.Tensor,   # [P]  periodos (una frecuencia por par)
    ) -> torch.Tensor:
        """
        RoPE 1D sobre P pares de dimensiones:
            new[..., i]   = x[..., i]   * cos - x[..., P+i] * sin
            new[..., P+i] = x[..., P+i] * cos + x[..., i]   * sin
        donde cos/sin = cos/sin(pos * 2π / freqs[i])
        """
        P      = freqs.shape[0]
        angles = pos.unsqueeze(-1) * (2.0 * math.pi / freqs.unsqueeze(0))
        cos    = angles.cos()[None, None]   # [1, 1, N, P]
        sin    = angles.sin()[None, None]
        x1, x2 = x[..., :P], x[..., P:]
        return torch.cat([x1 * cos - x2 * sin,
                          x2 * cos + x1 * sin], dim=-1)

    def apply(
        self,
        q:     torch.Tensor,   # [B, H, N_total, head_dim]
        k:     torch.Tensor,
        h:     int,            # filas del grid de patches
        w:     int,            # columnas del grid de patches
        n_pre: int,            # tokens de prefijo (CLS + registers), sin RoPE
    ):
        """Aplica RoPE 2D axial a los patch tokens de q y k."""
        P_x    = self.n_periods // 2
        P_y    = self.n_periods - P_x
        dev    = q.device

        # Posiciones 1D para cada patch en el grid h×w
        pos_y = torch.arange(h, dtype=torch.float32, device=dev)\
                     .unsqueeze(1).expand(h, w).reshape(-1)   # [h*w]
        pos_x = torch.arange(w, dtype=torch.float32, device=dev)\
                     .unsqueeze(0).expand(h, w).reshape(-1)   # [h*w]

        freqs_x = self.periods[:P_x]    # [P_x]
        freqs_y = self.periods[P_x:]    # [P_y]

        def rot(qk: torch.Tensor) -> torch.Tensor:
            pre   = qk[:, :, :n_pre]          # [B, H, n_pre, D]  — sin cambio
            patch = qk[:, :, n_pre:]           # [B, H, h*w, D]

            # Rotar componente x: primeros 2*P_x dims
            p_x = self._apply_1d_pairs(patch[..., :2*P_x],        pos_x, freqs_x)
            # Rotar componente y: siguientes 2*P_y dims
            p_y = self._apply_1d_pairs(patch[..., 2*P_x:2*P_x+2*P_y], pos_y, freqs_y)
            # Dims restantes sin RoPE
            p_rest = patch[..., 2*P_x + 2*P_y:]

            patch_new = (torch.cat([p_x, p_y, p_rest], dim=-1)
                         if p_rest.shape[-1] > 0
                         else torch.cat([p_x, p_y], dim=-1))
            return torch.cat([pre, patch_new], dim=2)

        return rot(q), rot(k)


class DINOv3MLP(nn.Module):
    """FFN estándar con nombres fc1/fc2 compatibles con checkpoint DINOv3."""
    def __init__(self, embed_dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DINOv3Attention(nn.Module):
    """
    Multi-head self-attention con RoPE 2D.

    DISEÑO DataParallel-safe:
      h, w, rope y n_pre se reciben como argumentos en forward() en vez de
      leerse de estado mutable interno. Esto evita que DataParallel (que hace
      shallow-copy del __dict__) deje referencias rotas al ViT padre.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x:     torch.Tensor,
        h:     int,
        w:     int,
        rope:  Axial2DRoPE,
        n_pre: int,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, hd   = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = rope.apply(q, k, h, w, n_pre)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class DINOv3Block(nn.Module):
    """
    Bloque Transformer DINOv3.

    forward(x, h, w, rope, n_pre) — argumentos explícitos, DataParallel-safe.
    Soporta Layer Scale opcional (detectado del checkpoint).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 use_layer_scale: bool = False, ls_init: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = DINOv3Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = DINOv3MLP(embed_dim, int(embed_dim * mlp_ratio))
        self.ls1   = LayerScale(embed_dim, ls_init) if use_layer_scale else nn.Identity()
        self.ls2   = LayerScale(embed_dim, ls_init) if use_layer_scale else nn.Identity()

    def forward(
        self,
        x:     torch.Tensor,
        h:     int,
        w:     int,
        rope:  Axial2DRoPE,
        n_pre: int,
    ) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), h, w, rope, n_pre))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv3ViT(nn.Module):
    """
    ViT-Base/16 con RoPE 2D axial y register tokens (storage_tokens).

    API diferenciada de iDoc VisionTransformer:
      prepare_tokens(x) → (tokens, h, w)   ← retorna h, w para evitar estado
      blocks            → DINOv3Block que requieren (x, h, w, rope, n_pre)
      norm              → LayerNorm final
      rope_embed        → Axial2DRoPE (único, compartido por todos los bloques)
      n_prefix_tokens   → int (1 CLS + n_register_tokens)

    Uso en _extract_intermediate / iDocQueryEncoder.forward:
        tokens, h, w = vit.prepare_tokens(x)
        for blk in vit.blocks:
            tokens = blk(tokens, h, w, vit.rope_embed, vit.n_prefix_tokens)
        tokens = vit.norm(tokens)
    """

    def __init__(
        self,
        patch_size:        int   = 16,
        embed_dim:         int   = 768,
        depth:             int   = 12,
        num_heads:         int   = 12,
        mlp_ratio:         float = 4.0,
        n_register_tokens: int   = 0,
        n_periods:         int   = 32,
        use_layer_scale:   bool  = False,
    ):
        super().__init__()
        self.patch_size        = patch_size
        self.embed_dim         = embed_dim
        self.n_register_tokens = n_register_tokens
        self.n_prefix_tokens   = 1 + n_register_tokens

        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token  = nn.Parameter(torch.zeros(1, embed_dim))
        if n_register_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.zeros(1, n_register_tokens, embed_dim))

        # RoPE: registrado solo aquí, pasado explícitamente a los bloques
        self.rope_embed = Axial2DRoPE(n_periods, embed_dim // num_heads)

        self.blocks = nn.ModuleList([
            DINOv3Block(embed_dim, num_heads, mlp_ratio, use_layer_scale)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def prepare_tokens(self, x: torch.Tensor):
        """
        Retorna (tokens, h, w) — sin estado interno mutable.
        h, w deben pasarse a cada bloque.forward() para el cálculo de RoPE.
        """
        B, C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        tokens = self.patch_embed(x)                      # [B, N, D]
        cls    = self.cls_token.expand(B, -1, -1)
        if self.n_register_tokens > 0:
            regs   = self.storage_tokens.expand(B, -1, -1)
            tokens = torch.cat([cls, regs, tokens], dim=1)
        else:
            tokens = torch.cat([cls, tokens], dim=1)

        return tokens, h, w   # h, w: Python ints, DataParallel-safe


# ═══════════════════════════════════════════════════════════════════════════════
#  Backbone unificado
# ═══════════════════════════════════════════════════════════════════════════════

class iDocBackbone(nn.Module):
    """
    ViT-Base congelado + Simple Feature Pyramid (strides 4, 8, 16, 32).
    Auto-detecta checkpoint iDoc (pos_embed / LoRA) o DINOv3 (rope_embed).
    """

    def __init__(
        self,
        arch:            str  = "vit_base",
        patch_size:      int  = 16,
        embed_dim:       int  = 768,
        depth:           int  = 12,
        num_heads:       int  = 12,
        extract_layers:  list = None,
        pretrained_path: str  = None,
    ):
        super().__init__()
        self.patch_size     = patch_size
        self.embed_dim      = embed_dim
        self.extract_layers = extract_layers or [2, 5, 8, 11]
        self.out_channels   = [embed_dim] * len(self.extract_layers)

        if pretrained_path and os.path.isfile(pretrained_path):
            ckpt  = torch.load(pretrained_path, map_location="cpu",
                               weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            if "rope_embed.periods" in state:
                vit = self._build_dinov3(state, patch_size, embed_dim,
                                         depth, num_heads)
            else:
                vit = self._build_idoc(state, patch_size, embed_dim,
                                       depth, num_heads)
        else:
            print("[iDocBackbone] ADVERTENCIA: pretrained_path no encontrado.")
            vit = self._build_idoc({}, patch_size, embed_dim, depth, num_heads)

        self.vit = vit
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        self.level_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in self.extract_layers
        ])

    # ── iDoc ──────────────────────────────────────────────────────────────────

    def _build_idoc(self, state, patch_size, embed_dim, depth, num_heads):
        vit = VisionTransformer(
            img_size           = [224],
            patch_size         = patch_size,
            embed_dim          = embed_dim,
            depth              = depth,
            num_heads          = num_heads,
            mlp_ratio          = 4,
            qkv_bias           = True,
            return_all_tokens  = True,
            masked_im_modeling = False,
        )
        vit.n_prefix_tokens = 1   # solo CLS

        if not state:
            return vit

        has_lora = any("w_lora_A" in k or "q_proj" in k for k in state)
        if has_lora:
            print("[iDocBackbone] LoRA detectado — fusionando pesos...")
            state = _merge_lora_weights(state, embed_dim, depth)

        vit_state = {k: v for k, v in state.items()
                     if not k.startswith(("head.", "fc_norm", "mask_token"))}
        msg = vit.load_state_dict(vit_state, strict=False)

        if any("pos_embed" in k for k in msg.missing_keys):
            nn.init.zeros_(vit.pos_embed)
            print("[iDocBackbone] pos_embed → zeros")

        skip = {"head", "masked_embed", "fc_norm", "pos_embed"}
        bad  = [k for k in msg.missing_keys if not any(s in k for s in skip)]
        if bad:
            print(f"[iDocBackbone] Missing ({len(bad)}): "
                  f"{bad[:3]}{'...' if len(bad) > 3 else ''}")
        else:
            print("[iDocBackbone] iDoc pesos cargados correctamente.")
        return vit

    # ── DINOv3 ────────────────────────────────────────────────────────────────

    def _build_dinov3(self, state, patch_size, embed_dim, depth, num_heads):
        n_reg     = state["storage_tokens"].shape[1] \
                    if "storage_tokens" in state else 0
        n_periods = state["rope_embed.periods"].shape[0]
        has_ls    = any("ls1.gamma" in k for k in state)

        print(f"[iDocBackbone] DINOv3 → n_register={n_reg}, "
              f"n_periods={n_periods}, rope_dim={2*n_periods}, "
              f"layer_scale={has_ls}")

        vit = DINOv3ViT(
            patch_size        = patch_size,
            embed_dim         = embed_dim,
            depth             = depth,
            num_heads         = num_heads,
            n_register_tokens = n_reg,
            n_periods         = n_periods,
            use_layer_scale   = has_ls,
        )

        msg = vit.load_state_dict(state, strict=False)

        # Filtrar claves esperadamente ausentes/extras del reporte
        skip_miss = {"mask_token"}
        # qkv.bias_mask = máscara de pruning del checkpoint, irrelevante para nosotros
        skip_unex = {"mask_token", "bias_mask"}
        missing   = [k for k in msg.missing_keys
                     if not any(s in k for s in skip_miss)]
        unexpected = [k for k in msg.unexpected_keys
                      if not any(s in k for s in skip_unex)]

        if missing:
            print(f"[iDocBackbone] DINOv3 missing   ({len(missing)}): "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[iDocBackbone] DINOv3 unexpected ({len(unexpected)}): "
                  f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if not missing and not unexpected:
            print("[iDocBackbone] DINOv3 pesos cargados correctamente.")
        return vit

    # ── Freeze override ───────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self

    # ── Feature extraction ────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_intermediate(self, x: torch.Tensor) -> list:
        is_dinov3 = isinstance(self.vit, DINOv3ViT)

        # El ViT frozen corre siempre en fp32 para evitar overflow en fp16
        # (especialmente con RoPE donde Q·Kᵀ puede ser muy grande).
        # No hay coste de memoria porque está bajo no_grad.
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()

            if is_dinov3:
                tokens, h, w = self.vit.prepare_tokens(x)
                rope  = self.vit.rope_embed
                n_pre = self.vit.n_prefix_tokens
                feats = {}
                for i, blk in enumerate(self.vit.blocks):
                    tokens = blk(tokens, h, w, rope, n_pre)
                    if i in self.extract_layers:
                        feats[i] = tokens[:, n_pre:]
            else:
                tokens = self.vit.prepare_tokens(x)
                feats  = {}
                for i, blk in enumerate(self.vit.blocks):
                    tokens = blk(tokens)
                    if i in self.extract_layers:
                        feats[i] = tokens[:, 1:]

        return [feats[i] for i in self.extract_layers]

    def forward(self, x: torch.Tensor) -> list:
        B, _, H, W = x.shape
        Hb = H // self.patch_size
        Wb = W // self.patch_size

        raw = self._extract_intermediate(x)
        out = []
        for tokens, norm, stride in zip(raw, self.level_norms, [4, 8, 16, 32]):
            f = norm(tokens).transpose(1, 2).reshape(B, self.embed_dim, Hb, Wb)
            if   stride == 4:
                f = F.interpolate(f, scale_factor=4.0, mode="bilinear",
                                  align_corners=False)
            elif stride == 8:
                f = F.interpolate(f, scale_factor=2.0, mode="bilinear",
                                  align_corners=False)
            elif stride == 32:
                f = F.max_pool2d(f, kernel_size=2, stride=2)
            out.append(f)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Sketch Refinement Encoder  [ENC-2]  (sin cambios)
# ═══════════════════════════════════════════════════════════════════════════════

class SketchRefinementEncoder(nn.Module):
    """
    Encoder Transformer entrenable sobre los tokens del sketch.
    Inicialización near-identity para compatibilidad con checkpoints FiLM.
    """

    def __init__(
        self,
        embed_dim:  int   = 768,
        num_heads:  int   = 8,
        num_layers: int   = 2,
        ffn_dim:    int   = 2048,
        dropout:    float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.encoder.layers:
            if hasattr(layer.self_attn, "out_proj"):
                nn.init.normal_(layer.self_attn.out_proj.weight, std=1e-4)
                nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.normal_(layer.linear2.weight, std=1e-4)
            nn.init.zeros_(layer.linear2.bias)

    def forward(
        self,
        cls_token:    torch.Tensor,   # [B, D]
        patch_tokens: torch.Tensor,   # [B, N, D]
    ) -> tuple:
        seq     = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        refined = self.encoder(seq)
        return refined[:, 0], refined[:, 1:]


# ═══════════════════════════════════════════════════════════════════════════════
#  Query Encoder  [ENC-1 + ENC-2]
# ═══════════════════════════════════════════════════════════════════════════════

class iDocQueryEncoder(nn.Module):
    """
    Codifica el sketch de query con ViT frozen + SketchRefinementEncoder.

    Maneja iDoc y DINOv3 con la misma API externa.
    Para DINOv3: usa prepare_tokens(x) → (tokens, h, w) y pasa
    h, w, rope, n_pre explícitamente a cada bloque (DataParallel-safe).
    """

    def __init__(
        self,
        backbone:        "iDocBackbone",
        query_dim:       int   = 768,
        patch_pool_size: int   = 7,
        use_refinement:  bool  = True,
        refine_layers:   int   = 2,
        refine_heads:    int   = 8,
        refine_ffn_dim:  int   = 2048,
        refine_dropout:  float = 0.1,
    ):
        super().__init__()
        self.backbone        = backbone
        self.query_dim       = query_dim
        self.patch_pool_size = patch_pool_size
        self.use_refinement  = use_refinement

        if use_refinement:
            self.refinement = SketchRefinementEncoder(
                embed_dim  = query_dim,
                num_heads  = refine_heads,
                num_layers = refine_layers,
                ffn_dim    = refine_ffn_dim,
                dropout    = refine_dropout,
            )

    def forward(self, query_img: torch.Tensor) -> tuple:
        vit       = self.backbone.vit
        is_dinov3 = isinstance(vit, DINOv3ViT)
        n_pre     = getattr(vit, "n_prefix_tokens", 1)

        # ViT frozen en fp32, igual que en _extract_intermediate
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                query_img_f = query_img.float()
                if is_dinov3:
                    tokens, h, w = vit.prepare_tokens(query_img_f)
                    rope = vit.rope_embed
                    for blk in vit.blocks:
                        tokens = blk(tokens, h, w, rope, n_pre)
                else:
                    tokens = vit.prepare_tokens(query_img_f)
                    for blk in vit.blocks:
                        tokens = blk(tokens)
                tokens = vit.norm(tokens)

        cls_token = tokens[:, 0]        # [B, D]
        patches   = tokens[:, n_pre:]   # [B, N_patch, D]  (sin CLS ni registers)

        # ── [ENC-1] Pool 14×14 → P×P ─────────────────────────────────────────
        B, N, D = patches.shape
        H_p = W_p = int(N ** 0.5)      # 14 para ViT-B/16 a 224×224

        spatial = patches.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2)
        pooled  = F.adaptive_avg_pool2d(
            spatial, (self.patch_pool_size, self.patch_pool_size))
        patches_pooled = pooled.permute(0, 2, 3, 1)\
                               .reshape(B, self.patch_pool_size ** 2, D)

        # ── [ENC-2] Sketch Refinement Encoder ────────────────────────────────
        if self.use_refinement:
            cls_token, patches_pooled = self.refinement(cls_token, patches_pooled)

        return cls_token, patches_pooled