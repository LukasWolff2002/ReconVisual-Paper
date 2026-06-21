# FCOSDetector — Documentación de Arquitectura y Cambios

**Proyecto:** Detección de elementos en documentos históricos condicionada por query sketch  
**Backbone:** iDoc ViT-Base (frozen) · **Cabeza:** FCOS · **Condicionamiento:** FiLM + Co-Attention  
**Última actualización:** Junio 2026  
**Estado del código:** revisado contra implementación real en `train_fcos/`

---

## Índice

1. [Visión general del sistema](#1-visión-general-del-sistema)
2. [Diagrama de arquitectura completo](#2-diagrama-de-arquitectura-completo)
3. [Componentes en detalle](#3-componentes-en-detalle)
   - 3.1 [Backbone — iDocBackbone](#31-backbone--idocbackbone)
   - 3.2 [Query Encoder — iDocQueryEncoder](#32-query-encoder--idocqueryencoder)
   - 3.3 [Sketch Refinement Encoder](#33-sketch-refinement-encoder-enc-2)
   - 3.4 [Feature Pyramid Network](#34-feature-pyramid-network)
   - 3.5 [Co-Attention Bidireccional](#35-co-attention-bidireccional)
   - 3.6 [FCOS Head](#36-fcos-head)
4. [Pipeline de pérdidas](#4-pipeline-de-pérdidas)
   - 4.1 [FCOS Loss (focal + GIoU + centerness)](#41-fcos-loss)
   - 4.2 [Supervised Contrastive Loss](#42-supervised-contrastive-loss)
5. [Data Augmentation](#5-data-augmentation)
   - 5.1 [Augmentaciones sobre la página](#51-augmentaciones-sobre-la-página)
   - 5.2 [Query Augmentation](#52-query-augmentation-aug-q)
   - 5.3 [Multi-Query Fusion en inferencia](#53-multi-query-fusion-en-inferencia-mq)
6. [Post-procesado de inferencia](#6-post-procesado-de-inferencia)
   - 6.1 [Adaptive Score Threshold](#61-adaptive-score-threshold-nms-2)
   - 6.2 [Context-Aware NMS](#62-context-aware-nms-nms-1)
7. [Estrategia de entrenamiento](#7-estrategia-de-entrenamiento)
   - 7.1 [Grupos de learning rate](#71-grupos-de-learning-rate)
   - 7.2 [Entrenamiento en fases](#72-entrenamiento-en-fases)
   - 7.3 [Multi-GPU y gradient accumulation](#73-multi-gpu-y-gradient-accumulation)
8. [Configuración de referencia](#8-configuración-de-referencia)
9. [Bugs conocidos en el código](#9-bugs-conocidos-en-el-código)
10. [Guía de migración desde versiones anteriores](#10-guía-de-migración-desde-versiones-anteriores)
11. [Tabla de ablación y decisiones de diseño](#11-tabla-de-ablación-y-decisiones-de-diseño)
12. [Parámetros totales y desglose de entrenable/frozen](#12-parámetros-totales-y-desglose-de-entrenablefrozen)

---

## 1. Visión general del sistema

El detector es un modelo **FCOS condicionado por query sketch**: dada una página de documento histórico y un boceto de referencia (sketch) de la clase objetivo, el modelo detecta todas las instancias de esa clase en la página.

El sistema resuelve tres problemas simultáneamente:

- **Pocas instancias por clase** — clases con <50 ejemplos en el dataset, mediante augmentación agresiva y pérdida contrastiva.
- **Alta variabilidad visual** — el mismo símbolo puede variar mucho entre páginas o escribas, mediante query augmentation y multi-query fusion.
- **Falsos positivos en páginas densas** — decoraciones y texturas que se confunden con objetos de interés, mediante co-attention bidireccional y context-aware NMS.

**Archivos del proyecto:**

| Archivo | Rol |
|---|---|
| `models/backbone.py` | iDocBackbone (ViT frozen) + SketchRefinementEncoder + iDocQueryEncoder |
| `models/cross_attn.py` | SketchCrossAttnLayer — co-attention bidireccional |
| `models/fpn.py` | FPN estándar top-down |
| `models/fcos_head.py` | FCOSHead — FiLM + CrossAttn + ramas conv |
| `models/detector.py` | FCOSDetector — integra todo; forward, predict, multi-query |
| `losses/fcos_loss.py` | FCOSLoss + SketchContrastiveLoss (SupCon) |
| `datasets/detection_dataset.py` | Dataset con todas las augmentaciones y multi-query |
| `utils/box_utils.py` | IoU, GIoU, centerness, adaptive threshold, context-aware NMS |
| `config.py` | Todos los hiperparámetros |
| `train.py` | Loop de entrenamiento, MultiGPUWrapper, param groups |

---

## 2. Diagrama de arquitectura completo

```
                         ENTRADA
          ┌──────────────────────────────────────────┐
          │  page_img  [B, 3, H, W]                  │
          │  query_img [B, 3, 224, 224]               │
          └───────────┬──────────────────────────────┘
                      │
          ┌───────────┴──────────────────────────┐
          │                                      │
          ▼                                      ▼
┌─────────────────────┐              ┌──────────────────────────────┐
│    iDocBackbone      │  (frozen)    │   iDocQueryEncoder           │
│                      │              │                              │
│  ViT-Base/16         │              │  Shared ViT-Base  (frozen)   │
│  extract layers      │              │    ↓ all 12 blocks           │
│  [2, 5, 8, 11]       │              │  cls_token   [B, 768]        │
│    ↓ tokens→2D       │              │  patches     [B, 196, 768]   │
│  C2 [B,768,H/4,W/4]  │              │    ↓ pool 14×14 → 7×7       │
│  C3 [B,768,H/8,W/8]  │              │  patches_pooled [B, 49, 768] │
│  C4 [B,768,H/16,W/16]│              │    ↓ [ENC-2]                 │
│  C5 [B,768,H/32,W/32]│              │  SketchRefinementEncoder     │
│    ↓ level_norms ✎   │              │  (2× Transformer, trainable) │
└──────────┬───────────┘              │    ↓           ↓             │
           │                          │  cls_refined  patches_refined│
           │                          │  [B, 768]     [B, 49, 768]   │
           ▼                          └────────┬─────────────────────┘
┌─────────────────────┐                        │
│        FPN  ✎       │                        │  cls_refined  → FiLM
│                      │                        │  patches_refined → Co-Attn
│  P2 [B,256,H/4,W/4] │                        │
│  P3 [B,256,H/8,W/8] │◄───────────────────────┤
│  P4 [B,256,H/16,W/16│                        │
│  P5 [B,256,H/32,W/32│                        │
│  P6 [B,256,H/64,W/64│                        │
└──────────┬───────────┘                        │
           │                                    │
           ▼                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                          FCOSHead  ✎                             │
│                                                                  │
│  Para cada nivel Pᵢ  (i = 0..4):                                │
│                                                                  │
│    feat [B, 256, Hᵢ, Wᵢ]                                       │
│      │                                                           │
│      ├─► FiLM(cls_refined)     γ·feat + β      [todos los niveles]
│      │                                                           │
│      │   [si i ≥ cross_attn_start_level = 1]                   │
│      │       ├─► Co-Attention bidireccional                     │
│      │       │     Paso 1: sketch_patches ← feat  (sketch se    │
│      │       │             adapta al documento actual)           │
│      │       │     Paso 2: feat ← sketch_patches* (imagen usa   │
│      │       │             sketch contextualizado)               │
│      │                                                           │
│      ├─► cls_branch: 4× conv(3×3)-GN(32)-ReLU → cls_pred       │
│      │                                    [B, C, Hᵢ, Wᵢ]       │
│      └─► reg_branch: 4× conv(3×3)-GN(32)-ReLU → bbox_pred      │
│                                            [B, 4, Hᵢ, Wᵢ]      │
│                                       → ctr_pred [B, 1, Hᵢ, Wᵢ]│
└──────────────────────────┬───────────────────────────────────────┘
                           │  score = σ(cls) × σ(centerness)
                           ▼
              ┌─────────────────────────┐
              │    Post-procesado       │
              │   (inferencia only)     │
              │                         │
              │  Adaptive threshold     │
              │  Context-aware NMS      │
              └────────────┬────────────┘
                           │
                           ▼
              detecciones {boxes, scores, labels}

✎ = módulo con parámetros entrenables
```

---

## 3. Componentes en detalle

### 3.1 Backbone — `iDocBackbone`

**Archivo:** `models/backbone.py` → clase `iDocBackbone`

ViT-Base/16 preentrenado, **completamente frozen** durante el entrenamiento del detector. Soporta dos tipos de checkpoint detectados automáticamente:

| Tipo | Detección | Clase interna |
|---|---|---|
| iDoc | clave `pos_embed` en el checkpoint | `VisionTransformer` (cargado desde `iDoc/models/vision_transformer.py`) |
| DINOv3 | clave `rope_embed.periods` en el checkpoint | `DINOv3ViT` (implementado en `backbone.py`) |

**DINOv3ViT** añade sobre el ViT estándar:
- **RoPE 2D axial** (`Axial2DRoPE`): periodos aprendibles `[n_periods]`, mitad para eje x y mitad para eje y. Se aplica a Q y K (no a V) en cada bloque de atención. DataParallel-safe: `h, w, rope, n_pre` se pasan explícitamente en cada forward, evitando estado mutable compartido entre réplicas.
- **Register tokens** (`storage_tokens`): tokens extra de prefijo además del CLS. El número se detecta del checkpoint.
- **Layer Scale** opcional: detectado por la presencia de claves `ls1.gamma`.

**Extracción de features:**

```
Capa 2  → C2  (stride 4)   = tokens reshape + ×4 bilinear upsample
Capa 5  → C3  (stride 8)   = tokens reshape + ×2 bilinear upsample
Capa 8  → C4  (stride 16)  = tokens reshape (resolución nativa ViT)
Capa 11 → C5  (stride 32)  = tokens reshape + max_pool2d stride-2
```

El forward del ViT corre siempre en **fp32** bajo `torch.no_grad()`, incluso cuando el resto del modelo usa fp16, para evitar overflow numérico especialmente con RoPE (Q·Kᵀ puede ser grande).

**`level_norms` (entrenables):** cuatro `LayerNorm(768)`, uno por nivel extraído. Adaptan la distribución de features del ViT frozen al espacio esperado por el FPN sin modificar el ViT. Únicos parámetros entrenables dentro del backbone (6,144 parámetros en total).

**Freeze override:** el método `train(mode)` llama siempre `self.vit.eval()` al final, garantizando que BatchNorm/Dropout del ViT permanezcan en modo eval aunque el modelo global esté en `model.train()`.

**LoRA fusion:** si el checkpoint iDoc contiene claves `w_lora_A`/`w_lora_B`, se fusionan con `W = W_base + B @ A` antes de congelar. No se requiere código LoRA en inferencia.

---

### 3.2 Query Encoder — `iDocQueryEncoder`

**Archivo:** `models/backbone.py` → clase `iDocQueryEncoder`

Codifica el sketch de referencia usando el **mismo ViT-Base frozen** del backbone (pesos compartidos vía `self.backbone.vit`). Maneja iDoc y DINOv3 con la misma API externa.

Retorna dos tensores:

| Tensor | Shape | Uso |
|---|---|---|
| `cls_token` | `[B, 768]` | Embedding global → FiLM |
| `patches_pooled` | `[B, 49, 768]` | Patches locales → Co-Attention |

**[ENC-1] Pool 14×14 → 7×7 (49 tokens)**

El ViT-Base produce 196 patch tokens (14×14). Se reduce a 7×7 = **49 tokens** con `adaptive_avg_pool2d`. Preserva más estructura espacial que el pooling 4×4=16 tokens anterior. El coste de memoria en CrossAttn escala linealmente con N_s, por lo que el paso de 16 a 49 tokens es manejable.

Para DINOv3, los register tokens se excluyen: `patches = tokens[:, n_prefix_tokens:]` donde `n_prefix_tokens = 1 + n_register_tokens`.

---

### 3.3 Sketch Refinement Encoder `[ENC-2]`

**Archivo:** `models/backbone.py` → clase `SketchRefinementEncoder`

```python
seq     = cat([cls_token.unsqueeze(1), patch_tokens], dim=1)  # [B, 1+49, 768]
refined = TransformerEncoder(2 layers, Pre-LN)(seq)
cls_refined     = refined[:, 0]     # [B, 768]
patches_refined = refined[:, 1:]    # [B, 49, 768]
```

Dos capas `TransformerEncoderLayer` con:
- Pre-LayerNorm (`norm_first=True`) — más estable en fine-tuning
- `batch_first=True`
- FFN de dimensión 2048
- Dropout 0.1
- LayerNorm final sobre la secuencia completa

El CLS se concatena con los patches para que cada capa transformer tenga acceso al contexto global del sketch al refinar los tokens locales.

**Inicialización near-identity:** las proyecciones de salida (`out_proj` de self-attn y `linear2` de FFN) se inicializan con `std=1e-4`. Al inicio el módulo actúa como identidad: un checkpoint FiLM cargado produce exactamente los mismos outputs que antes de añadir este encoder.

**Por qué es necesario:** el ViT fue preentrenado para reconocimiento general de documentos, no para comparar sketch con región concreta. Este encoder aprende a re-ponderar los tokens del sketch para esa tarea: amplifica tokens que describen partes discriminativas del símbolo y atenúa los de fondo.

---

### 3.4 Feature Pyramid Network

**Archivo:** `models/fpn.py`

FPN estándar con top-down pathway. Recibe `[C2, C3, C4, C5]` en 768 canales y produce `[P2, P3, P4, P5, P6]` en 256 canales.

```
C5 (768ch) → lateral 1×1 → top-down ──────────────────┐
C4 (768ch) → lateral 1×1 → + upsample(C5_lat) → P4    │
C3 (768ch) → lateral 1×1 → + upsample(C4_lat) → P3    │
C2 (768ch) → lateral 1×1 → + upsample(C3_lat) → P2    │
                                                        │
cada nivel → output 3×3 conv → Pᵢ [B, 256, H/sᵢ, W/sᵢ]│
P5 → stride-2 conv → P6                                │
```

Todas las convoluciones usan GroupNorm(32) y ReLU. Completamente entrenable.

---

### 3.5 Co-Attention Bidireccional

**Archivo:** `models/cross_attn.py` → clase `SketchCrossAttnLayer`

**Versión anterior — unidireccional:**
```
imagen → atiende → sketch   (sketch estático)
```

**Versión actual — bidireccional:**
```
Paso 1:  sketch → atiende → imagen    (sketch se adapta al documento)
Paso 2:  imagen → atiende → sketch*   (imagen usa sketch contextualizado)
```

```python
# cross_attn.py — forward()

# sketch_proj: sketch_dim (768) → feat_channels (256)
kv_sketch = self.sketch_proj(sketch_patches)      # [B, 49, 256]

# Paso 1: sketch ← imagen
# Q = kv_sketch, K/V = x_seq
sketch_update, _ = self.sketch_attn_img(
    query = kv_sketch,    # [B, 49, 256]
    key   = x_seq,        # [B, HW, 256]
    value = x_seq,
)
kv_sketch_refined = self.norm_sketch(kv_sketch + sketch_update)

# Paso 2: imagen ← sketch refinado
# Q = x_seq, K/V = kv_sketch_refined
img_update, _ = self.img_attn_sketch(
    query = x_seq,
    key   = kv_sketch_refined,
    value = kv_sketch_refined,
)
out = self.norm_img(x_seq + img_update)          # [B, HW, 256]
```

Ambas direcciones arrancan con `out_proj` a ceros (inicialización identity). Esto preserva exactamente el comportamiento del modelo si se carga un checkpoint anterior.

**Atributos del módulo:**

| Atributo | Dimensión | Rol |
|---|---|---|
| `sketch_proj` | Linear(768→256) | Proyecta patch tokens al espacio del FPN |
| `sketch_attn_img` | MHA(256, 8 heads) | Paso 1: sketch como Q |
| `norm_sketch` | LayerNorm(256) | Residual del paso 1 |
| `img_attn_sketch` | MHA(256, 8 heads) | Paso 2: imagen como Q |
| `norm_img` | LayerNorm(256) | Residual del paso 2 |
| `sketch_back_proj` | Linear(256→768) | Proyección de vuelta — creada pero sin uso en forward |

> **Nota:** `sketch_back_proj` se inicializa pero no se llama en `forward()`. Es código muerto pendiente de eliminar o integrar.

**`cross_attn_start_level`:** controla desde qué nivel FPN se activa la co-attention:

| Valor | Co-Attention en | Cuándo usar |
|---|---|---|
| `0` | P2, P3, P4, P5, P6 | FP muy finos en P2 |
| `1` | P3, P4, P5, P6 (**recomendado**) | Caso general |
| `2` | P4, P5, P6 | GPU con poca VRAM |

En P2 con imagen de 1333px hay ~73K ubicaciones espaciales. Con `start_level=1`, P2 usa solo FiLM mientras los niveles con menos ubicaciones usan co-attention completa.

> **Bug activo:** `cross_attn_start_level` se acepta en `FCOSHead.__init__` pero no se almacena ni se aplica en `forward_single_level`. CrossAttn se aplica en todos los niveles. Ver §9.

---

### 3.6 FCOS Head

**Archivo:** `models/fcos_head.py` → clase `FCOSHead`

Cabeza FCOS compartida entre niveles (pesos de las ramas conv compartidos; FiLM y CrossAttn separados por nivel). Para cada nivel Pᵢ, el flujo de cada rama (cls y reg) es:

```
feat [B, 256, Hᵢ, Wᵢ]
  → FiLMLayer(cls_refined)          γ·feat + β   [todos los niveles]
  → SketchCrossAttnLayer(patches)   [si i ≥ start_level]
  → 4× conv(3×3)-GN(32)-ReLU
  → predictor
```

**FiLMLayer:**
```python
params = Linear(query_dim=768, 2×feat_channels=512)(cls_refined)
gamma  = params[:, :256]   # init = 1  (γ init bias = 1)
beta   = params[:, 256:]   # init = 0
out    = gamma * feat + beta
```

**Predictores:**

| Rama | Salida | Nota |
|---|---|---|
| `cls_pred` | `[B, C, Hᵢ, Wᵢ]` | bias init = sigmoid⁻¹(0.01) = -4.6 |
| `reg_pred` | `[B, 4, Hᵢ, Wᵢ]` | × Scale aprendible por nivel → ReLU |
| `ctr_pred` | `[B, 1, Hᵢ, Wᵢ]` | desde features de reg (no cls) |

El score final es `σ(cls) × σ(ctr)`, lo que suprime predicciones no centradas en el objeto.

**Módulos separados por nivel:** `film_cls[i]`, `film_reg[i]`, `cross_attn_cls[i]`, `cross_attn_reg[i]`, `scales[i]`. Total: 5 instancias de cada uno.

**Módulos compartidos entre niveles:** `cls_convs`, `reg_convs`, `cls_pred`, `reg_pred`, `ctr_pred`.

---

## 4. Pipeline de pérdidas

### 4.1 FCOS Loss

```
L_total = λ_cls  · L_focal
        + λ_bbox · L_GIoU
        + λ_ctr  · L_BCE_centerness
        + λ_contrast · L_SupCon
```

**Asignación de targets (`_assign_targets`):**
1. Un punto (x, y) es positivo para un GT box si: (a) está dentro del box, y (b) `max(l,t,r,b)` cae en el rango de regresión del nivel correspondiente.
2. Si un punto cae en múltiples GT boxes, se asigna al de menor área.
3. Las regress_ranges son: P2→(0,32), P3→(32,64), P4→(64,128), P5→(128,256), P6→(256,∞).

**Focal Loss** (α=0.25, γ=2): reduce la contribución de ejemplos fáciles. Se normalizan por `n_positivos` del batch completo.

**GIoU Loss:** convierte predicciones ltrb + punto central a boxes xyxy antes de calcular GIoU. Solo sobre puntos positivos.

**BCE Centerness:**
```
centerness = √( min(l,r)/max(l,r) · min(t,b)/max(t,b) )
```
Supervision desde features de regresión (`centerness_on_reg=True`).

**class_weights:** inverso de frecuencia con raíz cuadrada, normalizados por la media. Compensan el desbalance entre las 22 clases del dataset.

---

### 4.2 Supervised Contrastive Loss

**Archivo:** `losses/fcos_loss.py` → clase `SketchContrastiveLoss`

```python
L_SupCon = -1/|P(i)| · Σ_{p∈P(i)} log [
    exp(cos(z_i, z_p) / τ) / Σ_{a∈A(i)} exp(cos(z_i, z_a) / τ)
]
```

Donde:
- `z_i` = `cls_token` refinado del sketch i (L2-normalizado)
- `P(i)` = índices del batch con la misma clase que i (excluye i)
- `A(i)` = todos los índices distintos de i (positivos + negativos)
- `τ = 0.07` (temperatura)

**Implementación numérica estable:**
```python
z        = F.normalize(cls_tokens, dim=-1)
sim      = z @ z.T / temperature                    # [B, B]
sim_nodiag = sim.masked_fill(eye_mask, -inf)
log_denom  = logsumexp(sim_nodiag, dim=1)           # [B]
log_prob   = sim - log_denom                        # [B, B]
loss       = -(log_prob * pos_mask).sum(1) / n_pos  # [B]
```

Si un batch es monoclase (`pos_mask.sum() == 0`), retorna `tensor(0.0)` sin error.

**Integración en train.py:** `forward_with_embeddings()` retorna `cls_tokens` extra sin duplicar el forward pass. Solo se pasan a la pérdida los tokens de muestras con label válido (`batch_cls_labels >= 0`).

> **Nota de diseño:** `batch_cls_labels` toma `t["labels"][0]` por muestra, asumiendo que cada muestra (page, query_class) tiene una única clase activa. Válido para el diseño del dataset donde cada sample corresponde a una clase.

---

## 5. Data Augmentation

### 5.1 Augmentaciones sobre la página

**Archivo:** `datasets/detection_dataset.py` → `_augment()`

**Multi-scale resize** — la imagen se escala aleatoriamente al min_size elegido de `[640, 720, 800, 900, 1024]` con max_size=1333.

**[AUG-2] Zoom-in forzado** (`zoom_in_prob=0.3`) — antes del resize global, crop centrado en una instancia positiva aleatoria con margen del 15% (`zoom_in_margin=0.15`). El crop se escala al tamaño estándar. Solo se activa si hay al menos una instancia. Garantiza que objetos pequeños lleguen a P2/P3 con resolución suficiente.

**[AUG-1] Copy-paste enriquecido** — cada crop pegado recibe:
- Escala aleatoria ×0.6–1.4, limitada al 40% de la imagen destino
- Flip horizontal (p=0.5) y vertical (p=0.2) independientes
- Color jitter propio del crop: brillo±40%, contraste±40%, saturación±30%, hue±8%

Clases prioritarias en el pool de copy-paste: `["marqeur", "croix", "pdp", "S", "T", "petit_A"]`, seleccionadas con p=0.7 si el pool prioritario no está vacío.

**[AUG-3] Hard-negative pages** (`hard_neg_prob=0.2`) — solo cuando la muestra ya no tiene instancias positivas. Mezcla (alpha uniforme 15-35%) un fondo denso (>5 boxes totales) sobre la imagen. El modelo aprende que textura/decoración ≠ objeto sin anotaciones adicionales.

**Flip horizontal** (`flip_prob=0.5`): ajusta las coordenadas de boxes `x → W - x`.

**Color jitter** (`color_jitter_prob=0.8`): brillo±30%, contraste±30%, saturación±20%, hue±5%.

---

### 5.2 Query Augmentation `[AUG-Q]`

**Archivo:** `datasets/detection_dataset.py` → `_augment_query()`

Solo durante entrenamiento. Objetivo: robustez a variaciones de trazo, inclinación y presión típicas de escritura histórica.

| Transformación | Prob | Parámetros | Propósito |
|---|---|---|---|
| Color jitter | 60% | brightness±20%, contrast±30% | Variaciones de tinta/escáner |
| Rotación | 40% | ±8°, `fill=255` | Inclinación del trazo |
| Elastic distortion | 30% | alpha=50, sigma=5 | Deformación caligráfica |
| Gaussian blur | 20% | radius 0.3–1.2 | Sketches de baja resolución |

La rotación usa `fill=255` (fondo blanco) para evitar artefactos negros en los bordes. La elastic distortion requiere `torchvision >= 0.12`; si no está disponible se omite silenciosamente.

> **Nota:** Los parámetros de query augmentation (`query_jitter_prob`, `query_rotate_prob`, etc.) no están definidos explícitamente en `config.py` — el dataset los toma de `aug_cfg.get(key, default)`. Para controlarlos desde la configuración, añadirlos a la sección `AUGMENTATION` de `config.py`.

---

### 5.3 Multi-Query Fusion en inferencia `[MQ]`

**Archivo:** `datasets/detection_dataset.py` → `load_multi_query()` / `models/detector.py` → `predict_multi_query()`

```python
# En inferencia:
tensors = dataset.load_multi_query(class_id, n_queries=3)
dets    = model.predict_multi_query(page_imgs, tensors)
```

N sketches de la misma clase se codifican por separado y sus embeddings se **promedian**:

```python
cls_proto     = mean([cls_1, cls_2, cls_3])         # [B, 768]
patches_proto = mean([patches_1, patches_2, ...])    # [B, 49, 768]
```

El prototipo promediado cancela el ruido específico de cada sketch individual. Especialmente útil para clases con alta variabilidad visual.

En entrenamiento no se usa: `__getitem__` samplea un sketch al azar en cada epoch.

---

## 6. Post-procesado de inferencia

### 6.1 Adaptive Score Threshold `[NMS-2]`

**Archivo:** `utils/box_utils.py` → `adaptive_score_threshold()`

```python
if n_candidatos <= density_trigger (50):
    threshold = base_threshold (0.05)
else:
    threshold = max(base_threshold, quantile(scores, 0.90))
```

Se calcula por imagen sobre los scores pre-NMS `σ(cls) × σ(ctr)` de todos los niveles. En páginas simples se usa el threshold base. En páginas densas, el threshold sube automáticamente al top 10% de scores, eliminando la larga cola de activaciones débiles.

---

### 6.2 Context-Aware NMS `[NMS-1]`

**Archivo:** `utils/box_utils.py` → `context_aware_nms()`

Aplicado **después** del NMS estándar por clase. Detecta clusters de activación por textura repetitiva. Un cluster de textura cumple los tres criterios simultáneamente:

1. ≥ `cluster_min_det` (4) detecciones con IoU mutua > `cluster_iou_thr` (0.20)
2. `max(scores) - min(scores)` < `score_spread_thr` (0.15): sin candidato dominante
3. Criterio 2 es la señal de patrón repetitivo: un objeto real tendría un punto más centrado con score claramente mayor

Cuando se detecta un cluster de textura, se conserva solo la detección de mayor score.

**Complejidad:** O(N²) en IoU entre detecciones post-NMS. En la práctica N es pequeño después del NMS estándar.

---

## 7. Estrategia de entrenamiento

### 7.1 Grupos de learning rate

**Archivo:** `train.py` → `build_param_groups()`

Cuatro grupos con LR diferencial:

| Grupo | Parámetros (por nombre) | LR | Justificación |
|---|---|---|---|
| `cross_attn` | `"cross_attn"` en el nombre | `lr × 4.0` | Empieza desde cero (out_proj=0) |
| `refinement` | `"refinement"` en el nombre | `lr × 4.0` | Empieza desde cero (near-identity) |
| `film` | `"film"` en el nombre | `lr × 0.2` | Ya entrenado, LR bajo |
| `other` | resto de parámetros entrenables | `lr` (base) | FPN, convs, level_norms, scales, predictors |

El optimizador es **AdamW** (`weight_decay=1e-4`, `betas=(0.9, 0.999)`) con scheduler cosine + warmup:

```python
if epoch < warmup_epochs:
    factor = (epoch + 1) / warmup_epochs
else:
    t      = (epoch - warmup_epochs) / (total - warmup_epochs)
    factor = 0.5 * (1 + cos(π·t))
lr_actual = min_lr + (base_lr - min_lr) * factor
```

---

### 7.2 Entrenamiento en fases

**Fase 1** (10-15 épocas) — `--freeze_except crossattn`

Solo `cross_attn` y `refinement` son entrenables. Co-attention y SketchRefinementEncoder aprenden en aislamiento sin interferir con los pesos FiLM convergidos.

```bash
python -m train_fcos.train \
    --resume checkpoint_film.pth \
    --freeze_except crossattn \
    --epochs 15 --lr 1e-4
```

> **Bug activo:** `apply_phase_freeze(model, 'crossattn')` activa solo parámetros con `"cross_attn"` en el nombre. Los parámetros de `refinement` (con `"refinement"` en el nombre) quedan frozen en Fase 1. Fix: cambiar la condición a `"cross_attn" in name or "refinement" in name`. Ver §9.

**Fase 2** (hasta convergencia) — fine-tuning conjunto:

```bash
python -m train_fcos.train \
    --resume checkpoint_fase1.pth \
    --epochs 80 --lr 1e-4
```

---

### 7.3 Multi-GPU y gradient accumulation

**Archivo:** `train.py` → `MultiGPUWrapper`

```python
class MultiGPUWrapper(nn.Module):
    def forward(self, pages, queries, mode="train", shapes=None):
        if mode == "train":
            return self.base_model.forward_with_embeddings(pages, queries)
        else:
            return self.base_model.predict(pages, queries, shapes)
```

El wrapper unifica la interfaz para `DataParallel`. Todas las operaciones sobre parámetros (freeze, param groups, optimizer, checkpoints) se hacen sobre el `FCOSDetector` original, no sobre el wrapper.

`unwrap_model()` navega `DataParallel → .module → MultiGPUWrapper → .base_model` para recuperar el `FCOSDetector` en evaluación.

**Gradient accumulation** (`--accum_steps N`): el batch efectivo es `batch_size × N`. El optimizer step y `clip_grad_norm_` se ejecutan cada N iteraciones o al final del epoch.

**Gradient clipping:** `clip_grad_norm_(base_model.parameters(), max_norm=1.0)` sobre el `FCOSDetector` original para evitar ambigüedades con el wrapper.

**Mixed precision:** `torch.amp.autocast("cuda")` + `GradScaler` cuando `use_fp16=True`. El ViT frozen corre siempre en fp32 dentro del autocast (override explícito con `autocast(enabled=False)`).

**TensorBoard:** logs de `Loss/train`, `mAP/val`, `LR/crossattn`, `LR/film`, `LR/other` y `AP/{class_id}` por epoch.

**Early stopping:** `early_stop_patience=1000` épocas sin mejora en val mAP (efectivamente desactivado por defecto).

---

## 8. Configuración de referencia

```python
# config.py — valores actuales del código

BACKBONE = dict(
    arch           = "vit_base",
    patch_size     = 16,
    embed_dim      = 768,
    depth          = 12,
    num_heads      = 12,
    extract_layers = [2, 5, 8, 11],
    out_channels   = [768, 768, 768, 768],
)

QUERY_ENCODER = dict(
    embed_dim        = 768,
    film_out_dim     = 256,
    sketch_pool_size = 7,          # [ENC-1] 49 tokens (antes 16)
    size             = 224,
    use_refinement   = True,       # [ENC-2] SketchRefinementEncoder
    refine_layers    = 2,
    refine_heads     = 8,
    refine_ffn_dim   = 2048,
    refine_dropout   = 0.1,
)

FCOS_HEAD = dict(
    in_channels            = 256,
    num_convs              = 4,
    num_classes            = 22,
    regress_ranges         = ((0,32),(32,64),(64,128),(128,256),(256,1e8)),
    strides                = [4, 8, 16, 32, 64],
    centerness_on_reg      = True,
    norm_on_bbox           = True,
    cross_attn_heads       = 8,
    cross_attn_drop        = 0.1,
    cross_attn_start_level = 1,    # co-attention desde P3 en adelante
)

LOSS = dict(
    focal_alpha       = 0.25,
    focal_gamma       = 2.0,
    use_class_weights = True,
    lambda_cls        = 1.0,
    lambda_bbox       = 1.0,
    lambda_ctr        = 1.0,
    lambda_contrast   = 0.1,   # SupCon: empezar en 0.05, subir si mezcla clases
    contrast_temp     = 0.07,
)

AUGMENTATION = dict(
    multi_scale_sizes           = [640, 720, 800, 900, 1024],
    flip_prob                   = 0.5,
    color_jitter_prob           = 0.8,
    brightness                  = 0.3,
    contrast                    = 0.3,
    saturation                  = 0.2,
    hue                         = 0.05,
    zoom_in_prob                = 0.3,          # [AUG-2]
    zoom_in_margin              = 0.15,
    copy_paste_prob             = 0.5,          # [AUG-1]
    copy_paste_max_objects      = 8,
    copy_paste_priority_classes = ["marqeur", "croix", "pdp", "S", "T", "petit_A"],
    cp_scale_range              = (0.6, 1.4),
    cp_flip_prob                = 0.5,
    cp_vflip_prob               = 0.2,
    cp_color_prob               = 0.7,
    cp_brightness               = 0.4,
    cp_contrast                 = 0.4,
    cp_saturation               = 0.3,
    cp_hue                      = 0.08,
    hard_neg_prob               = 0.2,          # [AUG-3]
    # Query augmentation [AUG-Q] — no están en config.py, usar defaults del dataset:
    # query_jitter_prob  = 0.6,  query_brightness = 0.2, query_contrast = 0.3
    # query_rotate_prob  = 0.4,  query_rotate_deg = 8
    # query_elastic_prob = 0.3,  query_elastic_alpha = 50, query_elastic_sigma = 5
    # query_blur_prob    = 0.2
)

EVAL = dict(
    iou_threshold        = 0.5,
    score_threshold      = 0.05,
    density_trigger      = 50,          # [NMS-2]
    adaptive_percentile  = 0.90,
    nms_iou_thresh       = 0.5,
    max_dets             = 100,
    cluster_iou_thr      = 0.20,        # [NMS-1]
    cluster_min_det      = 4,
    score_spread_thr     = 0.15,
)

OPTIMIZER = dict(type="AdamW", lr=1e-4, weight_decay=1e-4, betas=(0.9,0.999))
SCHEDULER = dict(type="cosine", warmup_epochs=5, min_lr=1e-6)

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
```

---

## 9. Bugs conocidos en el código

### Bug 1 — `cross_attn_start_level` nunca se aplica (alta prioridad)

**Archivo:** `models/fcos_head.py`

`FCOSHead.__init__` recibe `cross_attn_start_level` como argumento pero no lo almacena. En `forward_single_level`, CrossAttn se aplica en todos los niveles sin excepción.

**Fix:**
```python
# En __init__, añadir:
self.cross_attn_start_level = cross_attn_start_level

# En forward_single_level, reemplazar llamadas directas por:
if level_idx >= self.cross_attn_start_level:
    cls_feat = self.cross_attn_cls[level_idx](cls_feat, sketch_patches)
    reg_feat = self.cross_attn_reg[level_idx](reg_feat, sketch_patches)
```

**Impacto:** P2 actualmente usa Co-Attention con ~73K ubicaciones espaciales (a 1333px), lo que aumenta el coste de memoria y puede degradar el rendimiento en ese nivel.

---

### Bug 2 — Fase 1 no activa `refinement` (alta prioridad)

**Archivo:** `train.py` → `apply_phase_freeze()`

Con `freeze_except='crossattn'`, solo se descongelan parámetros cuyo nombre contiene `"cross_attn"`. El `SketchRefinementEncoder` tiene `"refinement"` en el nombre y queda frozen.

**Fix:**
```python
# Línea ~206 de train.py:
param.requires_grad = ("cross_attn" in name or "refinement" in name)
```

**Impacto:** en Fase 1, el SketchRefinementEncoder no aprende nada. Se pierde el beneficio de entrenar los dos módulos nuevos en aislamiento.

---

### Bug menor 3 — `sketch_back_proj` es código muerto

**Archivo:** `models/cross_attn.py`

`self.sketch_back_proj = nn.Linear(feat_channels, sketch_dim)` se inicializa cuando `bidirectional=True`, pero nunca se llama en `forward()`. Son parámetros incluidos en el optimizer que no reciben gradiente funcional.

**Fix:** eliminar la declaración, o integrarlo si se desea devolver el sketch refinado en el espacio 768 para uso externo.

---

### Inconsistencia de documentación 4 — Labels PASO 1/2 en `cross_attn.py`

**Archivo:** `models/cross_attn.py`, comentarios en `forward()`

El código etiqueta la ejecución de `sketch_attn_img` como "Paso 2 (primero en orden temporal)" y la de `img_attn_sketch` como "Paso 1". La doc describe Paso 1 = sketch←imagen y Paso 2 = imagen←sketch. La lógica de ejecución es **correcta** pero los labels están invertidos respecto a la documentación.

---

### Pendiente 5 — Query aug params no expuestos en `config.py`

Los parámetros de query augmentation (`query_jitter_prob`, `query_rotate_prob`, `query_rotate_deg`, `query_elastic_prob`, `query_blur_prob`, etc.) no están definidos en `config.py`. El dataset los toma con `cfg.get(key, default_hardcoded)`. Para controlarlos desde experimentos, añadirlos explícitamente a `AUGMENTATION` en `config.py`.

---

## 10. Guía de migración desde versiones anteriores

### Desde checkpoint FiLM (sin CrossAttn)

```python
model.load_state_dict(torch.load("checkpoint_film.pth")["model"], strict=False)
```

Los pesos nuevos (co-attention, refinement, sketch_back_proj) se inicializan automáticamente como identidad. El modelo produce exactamente los mismos outputs que el checkpoint original hasta que los módulos nuevos empiezan a aprender.

### Desde checkpoint FiLM + CrossAttn unidireccional

La clase `SketchCrossAttnLayer` mantiene el mismo nombre. El atributo antes llamado `attn` ahora se llama `img_attn_sketch`. Si el checkpoint tiene la clave `attn.*`, renombrarla antes de cargar:

```python
state = torch.load("checkpoint.pth")["model"]
state = {
    k.replace("cross_attn_cls.0.attn.", "cross_attn_cls.0.img_attn_sketch.")
     .replace("cross_attn_reg.0.attn.", "cross_attn_reg.0.img_attn_sketch."): v
    for k, v in state.items()
}
model.load_state_dict(state, strict=False)
```

### Cambios en la API pública

| Función | Antes | Ahora |
|---|---|---|
| `forward()` | retorna `(cls, bbox, ctr)` | igual ✓ |
| `predict()` | inferencia estándar | igual + adaptive threshold + context NMS ✓ |
| `forward_with_embeddings()` | no existía | retorna `(cls, bbox, ctr, cls_tokens)` |
| `predict_multi_query()` | no existía | fusión de N sketches |
| `FCOSLoss.forward()` | `(cls, bbox, ctr, targets)` | + `cls_tokens`, `labels_for_contrast` (opcionales) |
| `dataset.load_multi_query()` | no existía | carga N sketches de una clase |

---

## 11. Tabla de ablación y decisiones de diseño

| Componente | Alternativa considerada | Decisión | Justificación |
|---|---|---|---|
| Co-attention bidireccional | Solo unidireccional | **Bidireccional** | El sketch estático no se adapta al contexto del documento; el sketch actualizado guía mejor la búsqueda |
| Pool size | 4×4=16, 8×8=64, 14×14=196 | **7×7=49** | Balance entre detalle espacial y coste de atención; 196 tokens aumenta la complejidad O(N_s·HW) considerablemente |
| SketchRefinementEncoder capas | 1, 2, 4 | **2** | 1 capa insuficiente; 4 capas añaden riesgo de sobreajuste con pocos datos |
| Contrastive loss | Triplet loss, NT-Xent | **SupCon** | SupCon usa todos los positivos del batch; no requiere minería de pares |
| Temperatura SupCon | 0.1, 0.07, 0.05 | **0.07** | Valor estándar; 0.05 es muy agresivo con clases visualmente similares |
| cross_attn_start_level | 0, 1, 2 | **1 (P3)** | P2 a 1333px tiene ~73K ubicaciones; coste de memoria justifica FiLM-only |
| Query aug — rotación | ±5°, ±8°, ±15° | **±8°** | ±15° demasiado agresivo para símbolos orientados (flechas, letras); ±8° cubre variabilidad real |
| Hard-negative blending | Crop-and-paste de negativos | **Blending alpha 15-35%** | El crop-and-paste crea artefactos de borde obvios; el blending es más realista |
| Adaptive threshold percentil | 80%, 90%, 95% | **90%** | 95% pierde detecciones reales en páginas densas; 80% insuficiente contra ruido |
| Backbone type | Solo iDoc | **iDoc + DINOv3** | Soporte automático para checkpoints DINOv3 con RoPE 2D y register tokens |

---

## 12. Parámetros totales y desglose de entrenable/frozen

| Módulo | Parámetros | Estado |
|---|---|---|
| `backbone.vit` (ViT-Base/16) | ~85.8M | **FROZEN** |
| `backbone.level_norms` (4× LayerNorm(768)) | 6,144 | Entrenable |
| `query_encoder.refinement` (2× TransformerLayer, 768-dim, ffn=2048) | ~6.0M | Entrenable |
| `fpn` (4 laterales 1×1 + 5 output 3×3, 768→256) | ~3.2M | Entrenable |
| `head.film_cls + film_reg` (5 niveles × 2 × Linear(768,512)) | ~3.9M | Entrenable |
| `head.cross_attn_cls + cross_attn_reg` (5 niveles × 2 × MHA bidirecional) | ~5.2M | Entrenable |
| `head.cls_convs + reg_convs` (4 conv × 2 ramas, 256→256) | ~1.2M | Entrenable |
| `head.cls_pred + reg_pred + ctr_pred` | ~0.6M | Entrenable |
| `head.scales` (5 escalares) | 5 | Entrenable |
| **Total frozen** | **~85.8M** | |
| **Total entrenable** | **~20.1M** | |
| **Total modelo** | **~105.9M** | |

> Los valores son aproximados. El número exacto depende de si se usa DINOv3 (con register tokens) y del número de clases (`num_classes=22`).
