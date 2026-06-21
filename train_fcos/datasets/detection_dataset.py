"""
datasets/detection_dataset.py

Dataset de detección + build_datasets para crear los splits train/val/test.

Mejoras respecto a la versión anterior:
  [AUG-1] Copy-paste enriquecido:
            - Escala aleatoria del crop (×0.6 – ×1.4) → crítico para objetos pequeños
            - Flip horizontal/vertical del crop
            - Jitter de color independiente al crop (variabilidad visual)
  [AUG-2] Zoom-in forzado sobre instancias pequeñas.
  [AUG-3] Hard-negative context pages.
  [AUG-Q] Query augmentation (NUEVO):
            Jitter de color, rotación leve y elastic distortion sobre el sketch.
            El modelo aprende que el mismo símbolo puede tener variaciones de trazo,
            inclinación y presión — especialmente importante para documentos históricos.
  [MQ]    Multi-query fusion en inferencia (NUEVO):
            load_multi_query() carga y promedia N sketches de la misma clase.
            El CLS token promediado es un prototipo más estable que un único sketch,
            especialmente para clases con alta variabilidad visual.
            Se usa en evaluate/predict, no en train (en train ya se samplea al azar).

El split se hace por page_path para evitar data leakage.
"""

import os
import json
import random
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# elastic distortion (disponible en torchvision >= 0.12)
try:
    from torchvision.transforms import ElasticTransform as _ElasticTransform
    HAS_ELASTIC = True
except ImportError:
    HAS_ELASTIC = False


# ─── Helpers de índice ────────────────────────────────────────────────────────

def build_class_index(classes: list) -> dict:
    sorted_ids = sorted(c["class_id"] for c in classes)
    return {cid: idx for idx, cid in enumerate(sorted_ids)}


def build_class_name_index(classes: list) -> dict:
    sorted_ids = sorted(c["class_id"] for c in classes)
    id_to_name = {c["class_id"]: c["class_name"] for c in classes}
    return {id_to_name[cid]: idx for idx, cid in enumerate(sorted_ids)}


def compute_class_weights(
    samples: list, class_index: dict, num_classes: int
) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for s in samples:
        idx = class_index[s["class_id"]]
        counts[idx] += len(s["boxes"])
    counts  = counts.clamp(min=1)
    weights = 1.0 / counts.sqrt()
    weights = weights / weights.mean()
    return weights


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HistoricalDocDetectionDataset(Dataset):
    """
    Dataset de detección condicionado por query sketch.

    Cada __getitem__ retorna:
        page_img:  [3, H, W]    tensor normalizado
        query_img: [3, 224, 224] tensor normalizado (sketch de la clase)
        boxes:     [M, 4] float  xyxy en coords de page_img escalada
        labels:    [M]    long
    """

    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        json_path:        str,
        image_root:       str,
        sample_ids:       list  = None,
        min_size:         int   = 800,
        max_size:         int   = 1333,
        query_size:       int   = 224,
        augment:          bool  = False,
        aug_cfg:          dict  = None,
        copy_paste_pool:  list  = None,
        # [AUG-3] pool de páginas densas sin la clase objetivo
        hard_neg_pool:    list  = None,
    ):
        super().__init__()
        self.image_root     = image_root
        self.min_size       = min_size
        self.max_size       = max_size
        self.query_size     = query_size
        self.augment        = augment
        self.aug_cfg        = aug_cfg or {}
        self.copy_paste_pool = copy_paste_pool
        self.hard_neg_pool  = hard_neg_pool or []

        with open(json_path, "r") as f:
            data = json.load(f)

        self.classes         = data["classes"]
        self.class_index     = build_class_index(self.classes)
        self.class_name_idx  = build_class_name_index(self.classes)
        self.num_classes     = len(self.class_index)

        self.query_pool = {
            cls["class_id"]: [q["query_path"] for q in cls.get("queries", [])]
            for cls in self.classes
        }

        all_samples = data["samples"]
        if sample_ids is not None:
            sid_set     = set(sample_ids)
            all_samples = [s for s in all_samples if s["sample_id"] in sid_set]

        self.samples = all_samples

        self.normalize = T.Normalize(mean=self.PIXEL_MEAN, std=self.PIXEL_STD)

        self.class_weights = compute_class_weights(
            all_samples, self.class_index, self.num_classes
        )

    # ─── I/O helpers ──────────────────────────────────────────────────────────

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(os.path.join(self.image_root, path)).convert("RGB")

    def _resize_image_and_boxes(
        self, img: Image.Image, boxes: np.ndarray, min_size: int, max_size: int
    ):
        W, H  = img.size
        scale = min_size / min(H, W)
        if scale * max(H, W) > max_size:
            scale = max_size / max(H, W)
        new_H, new_W = int(round(H * scale)), int(round(W * scale))
        img = img.resize((new_W, new_H), Image.BILINEAR)
        if len(boxes) > 0:
            boxes = boxes.copy().astype(float)
            boxes[:, [0, 2]] *= (new_W / W)
            boxes[:, [1, 3]] *= (new_H / H)
        return img, boxes, scale

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        return self.normalize(TF.to_tensor(img))

    # ─── [AUG-2] Zoom-in sobre instancias pequeñas ────────────────────────────

    def _zoom_in(self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        """
        Crop alrededor de una instancia positiva aleatoria y escala al min_size.
        Garantiza que los objetos pequeños lleguen en resolución usable a P2/P3.
        Sólo se aplica si hay al menos una instancia en la imagen.
        """
        if len(boxes) == 0:
            return img, boxes, labels

        cfg        = self.aug_cfg
        margin_rel = cfg.get("zoom_in_margin", 0.15)   # margen relativo alrededor del objeto
        W, H       = img.size

        # Elegir una instancia aleatoria como centro del zoom
        idx    = random.randint(0, len(boxes) - 1)
        x1, y1, x2, y2 = boxes[idx]
        bw, bh = x2 - x1, y2 - y1

        # Calcular tamaño del crop con margen
        margin_x = max(bw * margin_rel, 10)
        margin_y = max(bh * margin_rel, 10)

        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(W, x2 + margin_x)
        cy2 = min(H, y2 + margin_y)

        # Asegurar tamaño mínimo del crop
        if (cx2 - cx1) < 32 or (cy2 - cy1) < 32:
            return img, boxes, labels

        crop     = img.crop((cx1, cy1, cx2, cy2))
        cw, ch   = crop.size

        # Ajustar boxes al nuevo sistema de coordenadas
        new_boxes = boxes.copy().astype(float)
        new_boxes[:, [0, 2]] -= cx1
        new_boxes[:, [1, 3]] -= cy1
        # Clipar al área del crop
        new_boxes[:, 0] = np.clip(new_boxes[:, 0], 0, cw)
        new_boxes[:, 1] = np.clip(new_boxes[:, 1], 0, ch)
        new_boxes[:, 2] = np.clip(new_boxes[:, 2], 0, cw)
        new_boxes[:, 3] = np.clip(new_boxes[:, 3], 0, ch)

        # Filtrar boxes con área mínima
        areas = (new_boxes[:, 2] - new_boxes[:, 0]) * \
                (new_boxes[:, 3] - new_boxes[:, 1])
        valid = areas > 4
        new_boxes  = new_boxes[valid]
        new_labels = labels[valid]

        if len(new_boxes) == 0:
            return img, boxes, labels

        # Escalar el crop al tamaño estándar
        crop, new_boxes, _ = self._resize_image_and_boxes(
            crop, new_boxes, self.min_size, self.max_size
        )
        return crop, new_boxes.astype(np.float32), new_labels

    # ─── Augmentación principal ───────────────────────────────────────────────

    def _augment(self, img, boxes, labels):
        cfg = self.aug_cfg

        # [AUG-2] Zoom-in probabilístico (antes del resize global)
        if (len(boxes) > 0 and
                random.random() < cfg.get("zoom_in_prob", 0.3)):
            img, boxes, labels = self._zoom_in(img, boxes, labels)

        # Multi-scale resize
        multi_sizes = cfg.get("multi_scale_sizes", [self.min_size])
        min_s = random.choice(multi_sizes)
        img, boxes, _ = self._resize_image_and_boxes(img, boxes, min_s, self.max_size)

        # Flip horizontal
        if random.random() < cfg.get("flip_prob", 0.5):
            img = TF.hflip(img)
            if len(boxes) > 0:
                W = img.size[0]
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]

        # Color jitter sobre la imagen completa
        if random.random() < cfg.get("color_jitter_prob", 0.8):
            jitter = T.ColorJitter(
                brightness = cfg.get("brightness", 0.3),
                contrast   = cfg.get("contrast",   0.3),
                saturation = cfg.get("saturation", 0.2),
                hue        = cfg.get("hue",        0.05),
            )
            img = jitter(img)

        # [AUG-1] Copy-paste enriquecido
        if (self.copy_paste_pool is not None and
                random.random() < cfg.get("copy_paste_prob", 0.5)):
            img, boxes, labels = self._copy_paste(img, boxes, labels)

        # [AUG-3] Hard-negative context injection
        if (self.hard_neg_pool and
                random.random() < cfg.get("hard_neg_prob", 0.2) and
                len(boxes) == 0):
            # Solo cuando la muestra ya no tiene instancias positivas
            img = self._inject_hard_neg_background(img)

        return img, boxes, labels

    # ─── [AUG-1] Copy-paste enriquecido ──────────────────────────────────────

    def _copy_paste(self, dst_img, dst_boxes, dst_labels):
        cfg        = self.aug_cfg
        max_obj    = cfg.get("copy_paste_max_objects", 8)
        priority   = set(cfg.get("copy_paste_priority_classes", []))
        n_to_paste = random.randint(1, max_obj)

        W_dst, H_dst = dst_img.size
        dst_img      = dst_img.copy()
        new_boxes    = list(dst_boxes) if len(dst_boxes) > 0 else []
        new_labels   = list(dst_labels) if len(dst_labels) > 0 else []

        # Jitter de color para los crops (simula variabilidad visual)
        crop_jitter = T.ColorJitter(
            brightness = cfg.get("cp_brightness", 0.4),
            contrast   = cfg.get("cp_contrast",   0.4),
            saturation = cfg.get("cp_saturation", 0.3),
            hue        = cfg.get("cp_hue",        0.08),
        )

        for _ in range(n_to_paste):
            src_sample = self._sample_from_pool(priority)
            if src_sample is None:
                continue
            src_img_pil = self._load_image(src_sample["page_path"])
            if not src_sample["boxes"]:
                continue

            box_info    = random.choice(src_sample["boxes"])
            x1, y1, x2, y2 = [int(v) for v in box_info["bbox_xyxy"]]
            x1, x2 = max(0, x1), min(src_img_pil.width,  x2)
            y1, y2 = max(0, y1), min(src_img_pil.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = src_img_pil.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw < 5 or ch < 5:
                continue

            # ── [AUG-1a] Escala aleatoria del crop ────────────────────────
            scale_range = cfg.get("cp_scale_range", (0.6, 1.4))
            s = random.uniform(*scale_range)
            new_cw = max(4, int(cw * s))
            new_ch = max(4, int(ch * s))
            # Limitar para que no supere 40% de la imagen destino
            new_cw = min(new_cw, int(W_dst * 0.4))
            new_ch = min(new_ch, int(H_dst * 0.4))
            crop = crop.resize((new_cw, new_ch), Image.BILINEAR)
            cw, ch = new_cw, new_ch

            # ── [AUG-1b] Flip del crop ────────────────────────────────────
            if random.random() < cfg.get("cp_flip_prob", 0.5):
                crop = TF.hflip(crop)
            if random.random() < cfg.get("cp_vflip_prob", 0.2):
                crop = TF.vflip(crop)

            # ── [AUG-1c] Jitter de color del crop ────────────────────────
            if random.random() < cfg.get("cp_color_prob", 0.7):
                crop = crop_jitter(crop)

            max_px = max(0, W_dst - cw)
            max_py = max(0, H_dst - ch)
            if max_px == 0 or max_py == 0:
                continue

            px = random.randint(0, max_px)
            py = random.randint(0, max_py)
            dst_img.paste(crop, (px, py))
            new_boxes.append([px, py, px + cw, py + ch])
            new_labels.append(self.class_index[src_sample["class_id"]])

        if new_boxes:
            boxes_arr = np.array(new_boxes,  dtype=np.float32)
            lbls_arr  = np.array(new_labels, dtype=np.int64)
        else:
            boxes_arr = dst_boxes if len(dst_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            lbls_arr  = dst_labels if len(dst_labels) > 0 else np.zeros((0,),  dtype=np.int64)

        return dst_img, boxes_arr, lbls_arr

    # ─── [AUG-3] Hard-negative context injection ──────────────────────────────

    def _inject_hard_neg_background(self, img: Image.Image) -> Image.Image:
        """
        Sobreimprime un fondo denso (de hard_neg_pool) con opacidad parcial.
        Simula páginas con mucho detalle sin instancias de la clase objetivo.
        """
        if not self.hard_neg_pool:
            return img
        neg_sample = random.choice(self.hard_neg_pool)
        try:
            neg_img = self._load_image(neg_sample["page_path"])
        except Exception:
            return img
        neg_img = neg_img.resize(img.size, Image.BILINEAR)
        alpha   = random.uniform(0.15, 0.35)   # transparencia baja: el fondo es sutil
        blended = Image.blend(img.convert("RGBA"),
                              neg_img.convert("RGBA"), alpha)
        return blended.convert("RGB")

    def _sample_from_pool(self, priority_names: set):
        if not self.copy_paste_pool:
            return None
        priority_pool = [
            s for s in self.copy_paste_pool
            if s.get("class_name") in priority_names
        ]
        pool = (priority_pool
                if priority_pool and random.random() < 0.7
                else self.copy_paste_pool)
        return random.choice(pool)

    # ─── [AUG-Q] Query augmentation ──────────────────────────────────────────

    def _augment_query(self, query_img: Image.Image) -> Image.Image:
        """
        Augmentaciones aplicadas al sketch de query en tiempo de entrenamiento.

        Objetivo: que el modelo sea robusto a variaciones de trazo, inclinación
        y presión típicas de documentos históricos escritos a mano.

        Transformaciones aplicadas:
          1. Color jitter leve (brillo/contraste) — simula variaciones de tinta
          2. Rotación leve ±rot_deg — simula inclinación del trazo
          3. Elastic distortion — simula deformación del trazo (si disponible)
          4. Gaussian blur opcional — simula baja resolución de algunos sketches
        """
        cfg = self.aug_cfg

        # 1. Color jitter del sketch (más sutil que el de la página)
        if random.random() < cfg.get("query_jitter_prob", 0.6):
            jitter = T.ColorJitter(
                brightness = cfg.get("query_brightness", 0.2),
                contrast   = cfg.get("query_contrast",   0.3),
                saturation = cfg.get("query_saturation", 0.1),
                hue        = cfg.get("query_hue",        0.02),
            )
            query_img = jitter(query_img)

        # 2. Rotación leve
        if random.random() < cfg.get("query_rotate_prob", 0.4):
            max_deg = cfg.get("query_rotate_deg", 8)
            angle   = random.uniform(-max_deg, max_deg)
            query_img = TF.rotate(query_img, angle,
                                  fill=255)   # fondo blanco al rotar

        # 3. Elastic distortion (simula deformación de trazo caligráfico)
        if HAS_ELASTIC and random.random() < cfg.get("query_elastic_prob", 0.3):
            alpha  = cfg.get("query_elastic_alpha",  50.0)
            sigma  = cfg.get("query_elastic_sigma",   5.0)
            elastic = _ElasticTransform(alpha=alpha, sigma=sigma)
            query_img = elastic(query_img)

        # 4. Gaussian blur leve (simula baja resolución del sketch)
        if random.random() < cfg.get("query_blur_prob", 0.2):
            radius = random.uniform(0.3, 1.2)
            query_img = query_img.filter(
                __import__("PIL.ImageFilter", fromlist=["GaussianBlur"])
                .GaussianBlur(radius=radius)
            )

        return query_img

    # ─── [MQ] Multi-query loading ─────────────────────────────────────────────

    def load_multi_query(
        self,
        class_id:   str,
        n_queries:  int = 3,
        query_size: int = 224,
    ) -> list:
        """
        Carga hasta n_queries sketches de la clase dada y los devuelve como
        lista de tensores normalizados [3, query_size, query_size].

        Uso típico en inferencia/evaluación:
            tensors = dataset.load_multi_query(class_id, n_queries=3)
            # promediar embeddings fuera del dataset (en el modelo)

        En train no se usa este método — __getitem__ ya samplea al azar.
        """
        paths = self.query_pool.get(class_id, [])
        if not paths:
            return []

        selected = random.sample(paths, min(n_queries, len(paths)))
        tensors  = []
        for p in selected:
            try:
                img = self._load_image(p)
                img = ImageOps.pad(img, (query_size, query_size))
                tensors.append(self._to_tensor(img))
            except Exception:
                continue
        return tensors

    # ─── __getitem__ ─────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        page_img = self._load_image(sample["page_path"])

        boxes = np.array(
            [b["bbox_xyxy"] for b in sample["boxes"]], dtype=np.float32
        ) if sample["boxes"] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(
            [self.class_index[sample["class_id"]]] * len(sample["boxes"]),
            dtype=np.int64,
        )

        query_paths = self.query_pool.get(sample["class_id"], [])
        if query_paths:
            query_img = self._load_image(random.choice(query_paths))
        else:
            if len(boxes) > 0:
                b = boxes[0].astype(int)
                query_img = page_img.crop((b[0], b[1], b[2], b[3]))
            else:
                query_img = page_img

        if self.augment:
            page_img, boxes, labels = self._augment(page_img, boxes, labels)
            # [AUG-Q] augmentar el sketch en train
            query_img = self._augment_query(query_img)
        else:
            page_img, boxes, _ = self._resize_image_and_boxes(
                page_img, boxes, self.min_size, self.max_size
            )

        query_img = ImageOps.pad(query_img, (self.query_size, self.query_size))

        page_tensor  = self._to_tensor(page_img)
        query_tensor = self._to_tensor(query_img)

        return {
            "page_img":  page_tensor,
            "query_img": query_tensor,
            "boxes":     torch.from_numpy(boxes).float(),
            "labels":    torch.from_numpy(labels).long(),
            "sample_id": sample["sample_id"],
            "img_shape": (page_tensor.shape[-2], page_tensor.shape[-1]),
        }

    def __len__(self) -> int:
        return len(self.samples)


# ─── Collate ──────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """Pad page_imgs al máximo H y W del batch."""
    page_imgs  = [b["page_img"]  for b in batch]
    query_imgs = [b["query_img"] for b in batch]

    max_H = max(img.shape[-2] for img in page_imgs)
    max_W = max(img.shape[-1] for img in page_imgs)

    padded = torch.zeros(len(page_imgs), 3, max_H, max_W)
    for i, img in enumerate(page_imgs):
        padded[i, :, :img.shape[-2], :img.shape[-1]] = img

    return {
        "page_imgs":  padded,
        "query_imgs": torch.stack(query_imgs, dim=0),
        "targets":    [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
        "img_shapes": [b["img_shape"] for b in batch],
    }


# ─── build_datasets ───────────────────────────────────────────────────────────

def build_datasets(
    json_path:  str,
    image_root: str,
    cfg:        dict,
) -> tuple:
    """
    Construye (train_ds, val_ds, test_ds) haciendo split por page_path.

    Mejoras:
      - hard_neg_pool: páginas de train sin instancias de la clase query
        (una por cada clase) → entrenamiento adversarial contra FP en páginas densas.
    """
    import random as _random

    ds_cfg  = cfg["DATASET"]
    aug_cfg = cfg.get("AUGMENTATION", {})
    qe_cfg  = cfg.get("QUERY_ENCODER", {})

    with open(json_path, "r") as f:
        data = json.load(f)

    all_samples = data["samples"]

    # ── Split por page_path ───────────────────────────────────────────────────
    page_paths = sorted(set(s["page_path"] for s in all_samples))
    rng        = _random.Random(ds_cfg.get("seed", 42))
    rng.shuffle(page_paths)

    n       = len(page_paths)
    n_train = int(n * ds_cfg.get("train_ratio", 0.8))
    n_val   = int(n * ds_cfg.get("val_ratio",   0.1))

    train_pages = set(page_paths[:n_train])
    val_pages   = set(page_paths[n_train:n_train + n_val])
    test_pages  = set(page_paths[n_train + n_val:])

    train_ids = [s["sample_id"] for s in all_samples if s["page_path"] in train_pages]
    val_ids   = [s["sample_id"] for s in all_samples if s["page_path"] in val_pages]
    test_ids  = [s["sample_id"] for s in all_samples if s["page_path"] in test_pages]

    print(f"[build_datasets] Páginas  : {len(train_pages)} train / "
          f"{len(val_pages)} val / {len(test_pages)} test")
    print(f"[build_datasets] Muestras : {len(train_ids)} train / "
          f"{len(val_ids)} val / {len(test_ids)} test")

    # Copy-paste pool: solo muestras de train con al menos un box
    train_samples = [s for s in all_samples if s["page_path"] in train_pages]
    train_pool    = [s for s in train_samples if s["boxes"]]

    # ── [AUG-3] Hard-negative pool ────────────────────────────────────────────
    # Para cada clase, buscamos páginas de train que NO tengan esa clase.
    # Usamos un pool global: páginas densas (con muchos boxes de otras clases).
    # Criterio de "densa": más de 5 boxes en total (sumando todas las clases).
    page_to_samples = {}
    for s in train_samples:
        page_to_samples.setdefault(s["page_path"], []).append(s)

    # Páginas con alta densidad de objetos (potenciales hard negatives)
    dense_pages = [
        page for page, smpls in page_to_samples.items()
        if sum(len(s["boxes"]) for s in smpls) > 5
    ]
    hard_neg_pool = [
        {"page_path": p, "class_id": None}
        for p in dense_pages
    ]
    print(f"[build_datasets] Hard-neg pool: {len(hard_neg_pool)} páginas densas")

    common = dict(
        json_path  = json_path,
        image_root = image_root,
        min_size   = ds_cfg.get("min_size",  800),
        max_size   = ds_cfg.get("max_size", 1333),
        query_size = qe_cfg.get("size",      224),
    )

    train_ds = HistoricalDocDetectionDataset(
        **common,
        sample_ids       = train_ids,
        augment          = True,
        aug_cfg          = aug_cfg,
        copy_paste_pool  = train_pool,
        hard_neg_pool    = hard_neg_pool,
    )
    val_ds = HistoricalDocDetectionDataset(
        **common,
        sample_ids = val_ids,
        augment    = False,
    )
    test_ds = HistoricalDocDetectionDataset(
        **common,
        sample_ids = test_ids,
        augment    = False,
    )

    return train_ds, val_ds, test_ds
