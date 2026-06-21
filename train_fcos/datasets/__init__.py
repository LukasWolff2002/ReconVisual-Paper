"""
datasets/__init__.py
Construye los datasets de train/val/test con split por page_id.
"""

import json
import random
import sys
import os
from torch.utils.data import DataLoader
from .detection_dataset import HistoricalDocDetectionDataset, collate_fn


def build_datasets(json_path: str, image_root: str, config: dict):
    """
    Split por PAGE (no por sample) para evitar data leakage.
    Returns: (train_ds, val_ds, test_ds)
    """
    ds_cfg  = config["DATASET"]
    aug_cfg = config.get("AUGMENTATION", {})

    with open(json_path, "r") as f:
        data = json.load(f)

    samples = data["samples"]

    # Split por page_id
    page_ids = list({s["page_id"] for s in samples})
    rng      = random.Random(ds_cfg["seed"])
    rng.shuffle(page_ids)

    n       = len(page_ids)
    n_train = int(n * ds_cfg["train_ratio"])
    n_val   = int(n * ds_cfg["val_ratio"])

    train_pages = set(page_ids[:n_train])
    val_pages   = set(page_ids[n_train:n_train + n_val])
    test_pages  = set(page_ids[n_train + n_val:])

    train_ids = [s["sample_id"] for s in samples if s["page_id"] in train_pages]
    val_ids   = [s["sample_id"] for s in samples if s["page_id"] in val_pages]
    test_ids  = [s["sample_id"] for s in samples if s["page_id"] in test_pages]

    print(f"  Split por pages: {len(train_pages)} train / {len(val_pages)} val / {len(test_pages)} test")
    print(f"  Samples:         {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # Pool de copy-paste: solo muestras de train
    copy_paste_pool = [s for s in samples if s["sample_id"] in set(train_ids)]

    common_kwargs = dict(
        json_path   = json_path,
        image_root  = image_root,
        min_size    = ds_cfg["min_size"],
        max_size    = ds_cfg["max_size"],
    )

    train_ds = HistoricalDocDetectionDataset(
        **common_kwargs,
        sample_ids       = train_ids,
        augment          = True,
        aug_cfg          = aug_cfg,
        copy_paste_pool  = copy_paste_pool,
    )
    val_ds = HistoricalDocDetectionDataset(
        **common_kwargs,
        sample_ids = val_ids,
        augment    = False,
    )
    test_ds = HistoricalDocDetectionDataset(
        **common_kwargs,
        sample_ids = test_ids,
        augment    = False,
    )

    # Verificación: si los tamaños no coinciden, algo salió mal
    assert len(train_ds) == len(train_ids), \
        f"ERROR SPLIT: train_ds tiene {len(train_ds)} samples pero se esperaban {len(train_ids)}"
    assert len(val_ds) == len(val_ids), \
        f"ERROR SPLIT: val_ds tiene {len(val_ds)} samples pero se esperaban {len(val_ids)}"

    return train_ds, val_ds, test_ds