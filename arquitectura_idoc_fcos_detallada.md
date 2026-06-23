# iDoc-FCOS — Referencia detallada de arquitectura

**Proyecto:** detección de patrones gráficos en documentos históricos condicionada por *sketch*
**Backbone:** ViT-Base/16 frozen (iDoc o DINOv3) · **Cabeza:** FCOS anchor-free · **Condicionamiento:** FiLM (global) + Co-Attention bidireccional (local)
**Objetivo de este doc:** que puedas explicar y justificar *cada* pieza de la arquitectura, con shapes, matemática y razonamiento de diseño.

> Nota de lectura: este documento describe el **comportamiento real del código** (`train_fcos/`). Donde el paper dice algo distinto a lo que corre, está marcado explícitamente en la sección 13. No te confíes solo del texto del paper para responderle al profe.

---

## 0. Idea en una frase

Dada una página `[3,H,W]` y un boceto a mano `[3,224,224]` de la clase objetivo, el modelo localiza **todas** las instancias de esa clase en la página. El mismo ViT frozen codifica página y sketch; el sketch condiciona una cabeza FCOS entrenable por dos vías complementarias (una global barata = FiLM, una local cara = co-attention).

El diseño ataca tres problemas simultáneos del dominio:

| Problema | Mecanismo de defensa |
|---|---|
| Pocas instancias por clase (long-tail, Gini 0.537) | SupCon loss + copy-paste + class weights |
| Mismo símbolo varía entre escribas/páginas | query augmentation + multi-query fusion (inferencia) |
| Páginas densas → muchos falsos positivos | co-attention + adaptive threshold + context-aware NMS |

---

## 1. Flujo end-to-end con shapes

```
page_img  [B,3,H,W]            query_img [B,3,224,224]
     │                              │
     ▼ (ViT frozen, fp32)           ▼ (MISMO ViT frozen, pesos compartidos)
C2 [B,768,H/4, W/4]            cls_token      [B,768]
C3 [B,768,H/8, W/8]            patches(196)   [B,196,768]
C4 [B,768,H/16,W/16]              │ adaptive_avg_pool2d 14×14→7×7
C5 [B,768,H/32,W/32]          patches_pooled [B,49,768]
     │ level_norms (4× LN, entrenables)│ SketchRefinementEncoder (2× Transformer)
     ▼                              ▼
   FPN  ───────────────►  cls_refined [B,768] ──► FiLM (todos los niveles)
P2..P6 [B,256,·]          patches_refined [B,49,768] ──► Co-Attention
     │                              │
     ▼ FCOSHead (por nivel)         │
  por cada Pᵢ:  FiLM → CoAttn → 4×conv → {cls, bbox, ctr}
     │
     ▼  score = σ(cls)·σ(ctr)
  Post-proceso (solo inferencia): adaptive threshold → NMS estándar → context-aware NMS
     ▼
  detecciones {boxes, scores, labels}
```

Dimensiones clave que conviene memorizar: **embed_dim ViT = 768**, **canal FPN = 256**, **49 tokens de sketch (7×7)**, **5 niveles de pirámide P2–P6**, **22 clases**, **B=4** por defecto.

---

## 2. Backbone — `iDocBackbone`

ViT-Base/16 preentrenado, **completamente frozen** durante el entrenamiento del detector. Soporta dos checkpoints distintos detectados automáticamente al cargar:

| Backbone | Cómo se detecta | Clase |
|---|---|---|
| **iDoc** | existe la clave `pos_embed` | `VisionTransformer` (de `iDoc/models/`) |
| **DINOv3** | existe la clave `rope_embed.periods` | `DINOv3ViT` (en `backbone.py`) |

### 2.1 Por qué un ViT da una pirámide multiescala

Un ViT/16 es de **una sola resolución** (stride 16). El truco: se extraen las secuencias de tokens en **cuatro bloques intermedios** `[2, 5, 8, 11]`, se reshapean a mapas 2D y se re-muestrean para fabricar una pirámide:

```
Bloque 2  → C2 (stride 4)  = reshape + ×4 bilinear upsample
Bloque 5  → C3 (stride 8)  = reshape + ×2 bilinear upsample
Bloque 8  → C4 (stride 16) = reshape (resolución nativa del ViT)
Bloque 11 → C5 (stride 32) = reshape + max_pool2d stride-2
```

El razonamiento: bloques tempranos conservan detalle espacial (mejor para C2/C3, objetos chicos), bloques tardíos tienen semántica más rica (C4/C5). Es una alternativa a usar un backbone jerárquico (Swin, ConvNeXt) sin perder el ViT preentrenado.

### 2.2 Especificidades de DINOv3 (`DINOv3ViT`)

- **RoPE 2D axial** (`Axial2DRoPE`): periodos aprendibles, mitad para eje *x* y mitad para *y*. Se aplica solo a **Q y K** (no a V) en cada bloque de atención. Implementación *DataParallel-safe*: `h, w, rope, n_pre` se pasan explícitos en cada forward para no compartir estado mutable entre réplicas de GPU.
- **Register tokens** (`storage_tokens`): tokens de prefijo extra además del CLS; el número se detecta del checkpoint. **Se excluyen** al extraer patches: `patches = tokens[:, n_prefix_tokens:]` con `n_prefix_tokens = 1 + n_register_tokens`. Si te olvidas de esto, mezclas register tokens con patches y rompes el reshape espacial.
- **Layer Scale** opcional, detectado por la clave `ls1.gamma`.

### 2.3 Detalles que un profe va a preguntar

- **Forward del ViT siempre en fp32** bajo `torch.no_grad()`, aunque el resto corra en fp16. Razón: con RoPE, `Q·Kᵀ` puede crecer y dar overflow en half precision. Se fuerza con `autocast(enabled=False)`.
- **`level_norms` (entrenables):** 4× `LayerNorm(768)`, uno por nivel extraído. Son los **únicos** parámetros entrenables dentro del backbone (6.144 en total). Adaptan la distribución del ViT frozen al espacio que espera el FPN, sin tocar los pesos del ViT.
- **Freeze robusto:** el `train(mode)` del backbone llama siempre a `self.vit.eval()` al final, para que Dropout/Norm del ViT queden en modo eval aunque pongas `model.train()`.
- **LoRA fusion:** si el checkpoint iDoc trae `w_lora_A`/`w_lora_B`, se fusionan `W = W_base + B @ A` **antes** de congelar. No se necesita código LoRA en inferencia. (iDoc fue domain-adaptado al corpus HORAE vía LoRA; acá esa adaptación queda "horneada" en los pesos.)

---

## 3. Query Encoder — `iDocQueryEncoder`

Codifica el sketch con el **mismo ViT frozen** (pesos compartidos vía `self.backbone.vit`). Devuelve dos representaciones con roles distintos:

| Tensor | Shape | Destino | Rol |
|---|---|---|---|
| `cls_token` | `[B,768]` | FiLM | prior global / "qué clase es" |
| `patches_pooled` | `[B,49,768]` | Co-Attention | estructura espacial / "cómo se ve" |

**Pooling 14×14 → 7×7 = 49 tokens** con `adaptive_avg_pool2d`. Decisión de diseño (ver §11 ablación): 196 tokens dispararían el coste de la co-attention `O(N_s · HW)`; 16 tokens (versión vieja) pierden estructura. 49 es el punto medio. El coste de la co-attention escala **lineal** en `N_s`, así que 16→49 es asumible.

---

## 4. Sketch Refinement Encoder `[ENC-2]`

```python
seq     = cat([cls_token.unsqueeze(1), patches_pooled], dim=1)  # [B, 1+49, 768]
refined = TransformerEncoder(2 capas, Pre-LN)(seq)
cls_refined     = refined[:, 0]    # [B,768]
patches_refined = refined[:, 1:]   # [B,49,768]
```

- 2× `TransformerEncoderLayer`, **Pre-LayerNorm** (`norm_first=True`, más estable en fine-tuning), `batch_first=True`, FFN dim 2048, dropout 0.1, LN final.
- Se concatena el CLS con los patches para que cada capa tenga **contexto global del sketch** al re-ponderar los tokens locales.

**Inicialización near-identity:** las proyecciones de salida (`out_proj` del self-attn y `linear2` del FFN) se inicializan con `std=1e-4`. Al arranque el módulo ≈ identidad, así un checkpoint FiLM previo produce **exactamente** los mismos outputs hasta que el módulo empieza a aprender. Esto permite añadir el módulo sin desestabilizar lo ya entrenado.

**Por qué existe:** el ViT fue preentrenado para representar documentos en general, no para comparar un sketch contra una región. Este encoder aprende a amplificar los tokens discriminativos del símbolo y atenuar el fondo del sketch.

---

## 5. Feature Pyramid Network

FPN top-down estándar: recibe `[C2,C3,C4,C5]` en 768 canales, produce `[P2,P3,P4,P5,P6]` en 256.

```
C5 → lateral 1×1 ──────────────► P5 (output 3×3)
C4 → lateral 1×1 + up(C5_lat) ─► P4
C3 → lateral 1×1 + up(C4_lat) ─► P3
C2 → lateral 1×1 + up(C3_lat) ─► P2
P5 → conv stride-2 ────────────► P6
```

Convoluciones con **GroupNorm(32)** + ReLU (GN y no BN porque el batch es chico, B=4, y BN se vuelve inestable). Totalmente entrenable.

---

## 6. Condicionamiento por sketch

Dos mecanismos complementarios: uno **global y barato** (FiLM, "qué buscar"), uno **local y caro** (co-attention, "dónde está").

### 6.1 FiLM (Feature-wise Linear Modulation)

```python
params = Linear(768 → 512)(cls_refined)   # 2×256
gamma  = params[:, :256]   # init bias = 1
beta   = params[:, 256:]   # init = 0
out    = gamma * feat + beta               # broadcast sobre H,W
```

- Modula **todos** los niveles de la pirámide con un prior de clase. Coste despreciable (es afín por canal).
- Init γ=1, β=0 → al arranque es la identidad (no perturba el feature map).
- Intuición: re-escala/desplaza canales según "qué clase estamos buscando", antes de mirar dónde.

### 6.2 Co-Attention bidireccional — `SketchCrossAttnLayer`

Versión vieja: unidireccional (imagen atiende a un sketch **estático**). Versión actual: **bidireccional** en dos pasos.

```python
kv_sketch = sketch_proj(sketch_patches)        # 768→256 : [B,49,256]

# Paso 1 — sketch ← imagen  (el sketch se adapta al estilo de ESTA página)
sketch_update, _ = sketch_attn_img(query=kv_sketch, key=x_seq, value=x_seq)  # x_seq=[B,HW,256]
kv_sketch_refined = norm_sketch(kv_sketch + sketch_update)

# Paso 2 — imagen ← sketch contextualizado  (afila las regiones que se parecen)
img_update, _ = img_attn_sketch(query=x_seq, key=kv_sketch_refined, value=kv_sketch_refined)
out = norm_img(x_seq + img_update)             # [B,HW,256]
```

- MHA de 256-dim, 8 cabezas, dropout 0.1, residual + LayerNorm en cada paso.
- **Init identity:** `out_proj` de ambas atenciones arranca en ceros → al cargar un checkpoint previo el output es idéntico hasta que aprende.
- Coste: el paso 2 es `O(HW · N_s)`. En P2 a 1333px hay ~73K ubicaciones → carísimo (ver §13, acá hay un bug).
- `sketch_back_proj` (Linear 256→768) está declarado pero **no se usa** en el forward → código muerto (ver §13).

`cross_attn_start_level` (config = 1) **debería** activar la co-attention recién desde P3, dejando P2 con solo FiLM. **Ojo: hoy no se aplica** (bug, §13).

---

## 7. FCOS Head

Cabeza FCOS con ramas conv **compartidas** entre niveles, pero **FiLM y CrossAttn separados por nivel** (`film_cls[i]`, `film_reg[i]`, `cross_attn_cls[i]`, `cross_attn_reg[i]`, `scales[i]`, i=0..4). Flujo por nivel y por rama (cls y reg):

```
feat [B,256,Hᵢ,Wᵢ]
  → FiLM(cls_refined)             [todos los niveles]
  → CoAttn(patches_refined)       [si i ≥ start_level]
  → 4× conv(3×3)-GN(32)-ReLU
  → predictor
```

| Predictor | Salida | Detalle |
|---|---|---|
| `cls_pred` | `[B,22,Hᵢ,Wᵢ]` | bias init = `sigmoid⁻¹(0.01) = -4.6` (arranca prediciendo "casi nada", clave para estabilizar focal loss) |
| `reg_pred` | `[B,4,Hᵢ,Wᵢ]` | distancias `(l,t,r,b)` × **Scale aprendible por nivel** → ReLU |
| `ctr_pred` | `[B,1,Hᵢ,Wᵢ]` | desde features de **regresión** (`centerness_on_reg=True`), no de clasificación |

**Score final:** `σ(cls) · σ(ctr)`. La centerness suprime predicciones lejos del centro del objeto.

**Por qué Scale por nivel:** cada nivel regresa distancias en un rango distinto; un escalar aprendible por nivel deja que la red ajuste la magnitud sin re-normalizar a mano.

---

## 8. Pipeline de pérdidas

```
L_total = λ_cls·L_focal + λ_bbox·L_GIoU + λ_ctr·L_BCE_ctr + λ_contrast·L_SupCon
λ_cls = λ_bbox = λ_ctr = 1.0   ·   λ_contrast = 0.1   (empezar 0.05, subir si mezcla clases)
```

### 8.1 Asignación de targets (`_assign_targets`)

Un punto `(x,y)` es **positivo** para un GT box si: (a) cae **dentro** del box, y (b) `max(l,t,r,b)` cae en el **rango de regresión** del nivel:

```
P2→(0,32)  P3→(32,64)  P4→(64,128)  P5→(128,256)  P6→(256,∞)
strides:  [4, 8, 16, 32, 64]
```

Si un punto cae en varios GT, se asigna al de **menor área** (regla FCOS estándar para resolver ambigüedad). Este "split por escala" es lo que hace que cada nivel se especialice en un tamaño de objeto.

### 8.2 Focal loss (clasificación)

`α=0.25, γ=2`. Baja el peso de los ejemplos fáciles (el fondo abundante). Se **normaliza por nº de positivos** del batch completo. El bias inicial −4.6 evita que al arranque el océano de negativos domine el gradiente.

### 8.3 GIoU loss (regresión)

Decodifica `(l,t,r,b)` + punto central a box `xyxy` y calcula GIoU. **Solo sobre puntos positivos.** GIoU (vs IoU plano) da gradiente útil incluso cuando las cajas no se solapan.

### 8.4 BCE centerness

```
centerness = sqrt( (min(l,r)/max(l,r)) · (min(t,b)/max(t,b)) )
```

Es 1 en el centro del objeto y →0 en los bordes. Se supervisa con BCE. Multiplicada al score en inferencia, baja las cajas mal centradas.

### 8.5 Supervised Contrastive Loss — `SketchContrastiveLoss`

```
L_SupCon = −1/|P(i)| · Σ_{p∈P(i)} log [ exp(cos(z_i,z_p)/τ) / Σ_{a∈A(i)} exp(cos(z_i,z_a)/τ) ]
```

- `z_i` = `cls_token` refinado del sketch i, **L2-normalizado**.
- `P(i)` = índices del batch de la **misma clase** que i (sin i). `A(i)` = todos menos i.
- `τ = 0.07`.
- Implementación numéricamente estable con `logsumexp` y máscara diagonal a `-inf`. Si el batch es monoclase (`pos_mask.sum()==0`) retorna `0.0` sin romper.
- **Para qué:** compacta los embeddings de sketches de la misma clase. Especialmente útil en la cola larga (clases con poquísimos ejemplos), donde el detector solo no aprende una representación discriminativa.

`forward_with_embeddings()` devuelve los `cls_tokens` extra **sin duplicar** el forward. Solo entran a la pérdida muestras con label válido (`>= 0`). Supuesto: cada muestra = (página, clase-query) tiene **una sola** clase activa (`t["labels"][0]`).

### 8.6 Class weights

Inverso de frecuencia con **raíz cuadrada**, normalizado por la media. La raíz amortigua: sin ella, marqeur (434) vs obj_42 (7) daría pesos absurdos.

---

## 9. Data augmentation

### 9.1 Sobre la página (`_augment`)

- **Multi-scale resize:** min_size aleatorio de `[640,720,800,900,1024]`, max_size 1333.
- **[AUG-2] Zoom-in forzado** (`p=0.3`): crop centrado en una instancia positiva (margen 15%) antes del resize. Garantiza que objetos chicos lleguen a P2/P3 con resolución.
- **[AUG-1] Copy-paste enriquecido:** pega crops con escala ×0.6–1.4 (≤40% de la imagen), flip H (0.5)/V (0.2), color jitter propio. **Pool prioritario:** `["marqeur","croix","pdp","S","T","petit_A"]`, p=0.7. *(Ojo: este pool son clases frecuentes/intermedias, no las más raras — ver §13.)*
- **[AUG-3] Hard-negative pages** (`p=0.2`): solo si la muestra ya no tiene positivos. Mezcla (alpha 15–35%) un fondo denso (>5 boxes) para enseñar "textura ≠ objeto".
- Flip horizontal global (0.5), color jitter global (0.8).

### 9.2 Query augmentation `[AUG-Q]` (solo train)

| Transf. | Prob | Parámetros |
|---|---|---|
| Color jitter | 0.6 | brillo±20%, contraste±30% |
| Rotación | 0.4 | ±8°, `fill=255` (fondo blanco, no negro) |
| Elastic | 0.3 | alpha=50, sigma=5 (requiere torchvision ≥0.12; si no, se omite silencioso) |
| Gaussian blur | 0.2 | radius 0.3–1.2 |

Objetivo: robustez a variación de trazo/inclinación/presión. *(Estos params no están en `config.py`; se toman con `cfg.get(key, default)` — pendiente exponerlos, ver §13.)*

### 9.3 Multi-query fusion (solo inferencia) `[MQ]`

N sketches de una clase se codifican por separado y sus embeddings se **promedian**:

```python
cls_proto     = mean([cls_1, ..., cls_N])
patches_proto = mean([patches_1, ..., patches_N])
```

Cancela ruido idiosincrático de un trazo. En train **no** se usa: `__getitem__` samplea un sketch al azar por epoch (lo cual ya es una forma de augmentación).

---

## 10. Post-procesado de inferencia

### 10.1 Adaptive score threshold `[NMS-2]`

```python
if n_candidatos <= 50:   threshold = 0.05
else:                    threshold = max(0.05, quantile(scores, 0.90))
```

Por imagen, sobre scores pre-NMS. En páginas simples usa el umbral base; en densas sube al top-10%, matando la cola de activaciones débiles.

### 10.2 Context-aware NMS `[NMS-1]`

Después del NMS estándar por clase. Marca un cluster como "textura repetitiva" si cumple **los tres**:

1. ≥ 4 detecciones con IoU mutua > 0.20,
2. `max(score) − min(score) < 0.15` (sin candidato dominante),
3. (2) es la señal: un objeto real tendría un pico de score claro.

En un cluster de textura, conserva **solo** la de mayor score. Complejidad `O(N²)`, pero N es chico post-NMS.

---

## 11. Estrategia de entrenamiento

### 11.1 Grupos de learning rate (`build_param_groups`)

| Grupo | Match por nombre | LR | Razón |
|---|---|---|---|
| `cross_attn` | `"cross_attn"` | `lr×4.0` | arranca de cero (out_proj=0) |
| `refinement` | `"refinement"` | `lr×4.0` | arranca near-identity |
| `film` | `"film"` | `lr×0.2` | ya entrenado, LR bajo |
| `other` | resto | `lr` | FPN, convs, level_norms, scales, predictores |

AdamW (`wd=1e-4`, `betas=(0.9,0.999)`), scheduler **cosine + warmup**:

```python
if epoch < warmup_epochs: factor = (epoch+1)/warmup_epochs
else: factor = 0.5*(1 + cos(π·t)),  t=(epoch-warmup)/(total-warmup)
lr = min_lr + (base_lr - min_lr)·factor
```

### 11.2 Entrenamiento en dos fases

- **Fase 1 (10–15 ep):** `--freeze_except crossattn`. *Intención:* entrenar co-attention + refinement en aislamiento sobre un checkpoint FiLM convergido, sin tocar lo demás. *(Realidad: refinement queda frozen — bug §13.)*
- **Fase 2 (hasta converger, 80 ep):** fine-tuning conjunto con LR diferencial (los grupos de arriba).

Racional: introducir los módulos nuevos de a poco estabiliza el entrenamiento y no destruye la capacidad de pattern-spotting heredada de iDoc.

### 11.3 Multi-GPU, AMP, misc

- `MultiGPUWrapper` unifica la interfaz para `DataParallel`; toda operación sobre parámetros (freeze, optimizer, checkpoints) se hace sobre el `FCOSDetector` base, no sobre el wrapper. `unwrap_model()` navega `DataParallel→.module→wrapper→.base_model`.
- **Gradient accumulation** (`--accum_steps N`): batch efectivo `4·N`; step y clip cada N iters.
- **Grad clipping** `max_norm=1.0` sobre el modelo base.
- **AMP** (`use_fp16=True`): `autocast("cuda")` + `GradScaler`, con el ViT siempre en fp32 (override).
- **Early stopping** `patience=1000` → efectivamente desactivado por defecto.
- **Semilla única** `seed=42` (una corrida; sin multi-seed → sin estimación de varianza).

---

## 12. Conteo de parámetros

| Módulo | Params | Estado |
|---|---|---|
| ViT-Base/16 | ~85.8M | **FROZEN** |
| level_norms (4× LN 768) | 6.144 | entrenable |
| refinement (2× Transformer) | ~6.0M | entrenable |
| FPN | ~3.2M | entrenable |
| FiLM (cls+reg, 5 niveles) | ~3.9M | entrenable |
| Co-Attn (cls+reg, 5 niveles) | ~5.2M | entrenable |
| convs cls+reg | ~1.2M | entrenable |
| predictores cls+reg+ctr | ~0.6M | entrenable |
| scales (5) | 5 | entrenable |
| **Total frozen** | **~85.8M** | |
| **Total entrenable** | **~20.1M** | |
| **Total** | **~105.9M** | |

Aproximado; varía con DINOv3 (register tokens) y `num_classes`.

---

## 13. Discrepancias código ↔ paper (LÉELO antes de defenderlo)

Tu profe es experto: si compara el texto con el código, estas son las grietas. Para cada una, o corriges el código y re-corres, o ajustas el texto.

1. **Co-attention en todos los niveles, no "desde un nivel intermedio".**
   El paper dice "bidirectional co-attention applied from an intermediate pyramid level onward" y la config tiene `cross_attn_start_level=1`. Pero `FCOSHead.__init__` recibe el argumento y **no lo guarda**; `forward_single_level` aplica co-attention en **todos** los niveles, incluido P2 (~73K ubicaciones a 1333px). Los resultados reportados salieron de la versión "todos los niveles".
   *Fix:* guardar `self.cross_attn_start_level` y envolver las llamadas en `if level_idx >= self.cross_attn_start_level`.

2. **El refinement encoder NO se entrena en Fase 1.**
   El paper dice que los módulos nuevos "are first optimized in isolation". Pero `apply_phase_freeze(..., 'crossattn')` solo descongela nombres con `"cross_attn"`; `refinement` queda frozen y recién aprende en Fase 2.
   *Fix:* `param.requires_grad = ("cross_attn" in name or "refinement" in name)`.

3. **Umbral de score: config 0.05 vs reportado 0.25.**
   `EVAL.score_threshold=0.05`, pero el paper/reporte reportan P/R/F1 a 0.25. Para mAP da igual (integra sobre umbrales); para P/R/F1 no. Declara explícito que 0.25 es el operating point elegido y que es el mismo para ambos backbones.

4. **`sketch_back_proj` es código muerto** (declarado, nunca llamado). Infla el conteo de "entrenables". Elimínalo antes de reportar nº de params.

5. **Pool de copy-paste vs justificación long-tail.** El paper vende copy-paste como remedio del long-tail, pero el pool prioritario son clases frecuentes/intermedias (marqeur, S, croix...), no las raras (obj_42, obj_31 con 7). Mecánicamente tiene sentido (necesitas crops de origen), pero acota la afirmación o ajusta el pool.

6. **Labels "Paso 1/Paso 2" invertidos** en los comentarios de `cross_attn.py` (la lógica está bien, los comentarios no). Cosmético, pero un revisor cuidadoso lo nota.

---

## 14. Preguntas tipo examen (con respuesta corta)

**P: Si el ViT es single-scale, ¿de dónde sale la pirámide?**
R: De extraer 4 bloques intermedios `[2,5,8,11]` y re-muestrearlos (up/downsample) a C2–C5; el FPN top-down los fusiona. No hay backbone jerárquico.

**P: ¿Por qué dos mecanismos de condicionamiento y no uno?**
R: FiLM es global y barato → prior de clase en todos los niveles ("qué"). Co-attention es local y caro → interacción espacial fina ("dónde"). Se complementan: FiLM orienta, co-attention afila.

**P: ¿Por qué inicializar FiLM y co-attention como identidad?**
R: Para añadir los módulos sobre un checkpoint ya convergido sin perturbar sus salidas; el entrenamiento las "enciende" gradualmente y se estabiliza.

**P: ¿Por qué GroupNorm y no BatchNorm en FPN/head?**
R: Batch chico (B=4); BN se vuelve ruidoso/inestable con pocos ejemplos por batch. GN no depende del batch.

**P: ¿Qué hace la centerness y por qué multiplicarla al score?**
R: Mide cuán centrado está un punto respecto al objeto (`sqrt` del producto de razones l/r y t/b). Multiplicada, hunde las cajas predichas en los bordes, que suelen ser imprecisas.

**P: ¿Por qué bias −4.6 en la cabeza de clasificación?**
R: `sigmoid(−4.6)≈0.01`; el modelo arranca prediciendo "casi todo es fondo", lo que evita que el desbalance positivo/negativo reviente la focal loss al inicio.

**P: ¿Por qué SupCon y no triplet?**
R: SupCon usa **todos** los positivos del batch a la vez y no requiere minería de pares/triples; más eficiente y estable en cola larga.

**P: ¿Por qué el ViT corre en fp32 dentro de AMP?**
R: Con RoPE, `Q·Kᵀ` puede crecer y dar overflow en fp16. Se aísla con `autocast(enabled=False)`.

**P: ¿Por qué DINOv3 (general) le gana a iDoc (domain-adaptado)?**
R: Hipótesis del paper: usado como extractor **frozen** para una cabeza entrenable, la amplitud representacional del pre-entrenamiento masivo transfiere mejor que la especialización estrecha de iDoc, sobre todo en clases raras. La ventaja se concentra justo donde hay poca supervisión.

**P: ¿Cuál es la mayor debilidad experimental?**
R: Val chico (55 páginas, 111 cajas), una sola semilla, 7/22 clases ausentes del split, y AP por clase con 1–3 GT (frágiles). Sin ablaciones empíricas todavía (la tabla §11 es de decisiones de diseño, no de resultados). Mitigación propuesta: cross-validation estratificada y ablaciones de SupCon/co-attention.
