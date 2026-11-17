#!/usr/bin/env python3
"""Stage 1 (run once, on Kaggle): build the clients and cache frozen ResNet-18 features.

Usage:
    python fedbench_features.py --out feats [--root medmnist_data]

For each preset this writes feats/preset_A.npz and feats/preset_B.npz holding, for every client,
the 256-d features after the third residual stage (global-average-pooled) of the train, validation and
test images, their labels, and a JSON metadata record.

Partition (fixed, independent of the training seed, so every strategy and seed sees the same clients):
  * official MedMNIST train split, shuffled with PARTITION_SEED, cut into 3 equal contiguous parts;
    each part capped at 2,000 images; the first 15% of each part becomes that client's validation set;
  * official test split, shuffled, cut into 3 equal parts, each capped at 1,000 images.
Images are the native 64 x 64 MedMNIST+ files, converted to 3 channels and normalized with the
ImageNet mean and standard deviation. No augmentation is used, which is what makes caching valid.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision

PRESETS = {
    "A": ["pathmnist", "dermamnist", "bloodmnist", "octmnist", "tissuemnist"],
    "B": ["organamnist", "pneumoniamnist", "retinamnist", "breastmnist", "organcmnist"],
}
CLIENTS_PER_SPECIALTY = 3
TRAIN_CAP, TEST_CAP, VAL_FRAC = 2000, 1000, 0.15
PARTITION_SEED = 2026
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_split(flag, split, root):
    import medmnist
    from medmnist import INFO
    cls = getattr(medmnist, INFO[flag]["python_class"])
    ds = cls(split=split, download=True, size=64, root=root)
    imgs = np.asarray(ds.imgs)
    labels = np.asarray(ds.labels).reshape(-1).astype(np.int64)
    return imgs, labels, len(INFO[flag]["label"])


def to_tensor(imgs):
    x = imgs.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = np.repeat(x[..., None], 3, axis=-1)
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def make_backbone(device):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    body = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3,
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    return body.eval().to(device)


@torch.no_grad()
def extract(body, imgs, device, bs=512):
    out = []
    for k in range(0, len(imgs), bs):
        out.append(body(to_tensor(imgs[k:k + bs]).to(device)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="feats")
    ap.add_argument("--root", default="medmnist_data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    body = make_backbone(device)
    rng = np.random.default_rng(PARTITION_SEED)
    for preset, flags in PRESETS.items():
        t0 = time.time()
        arrays, meta = {}, {"preset": preset, "specialties": flags, "client_specialty": [], "n_classes": [],
                            "n_train": [], "n_val": [], "n_test": [], "partition_seed": PARTITION_SEED,
                            "train_cap": TRAIN_CAP, "test_cap": TEST_CAP, "val_frac": VAL_FRAC,
                            "image_size": 64, "feature": "resnet18_layer3_gap_256"}
        c = 0
        for s, flag in enumerate(flags):
            tr_x, tr_y, k = load_split(flag, "train", args.root)
            te_x, te_y, _ = load_split(flag, "test", args.root)
            tr_parts = np.array_split(rng.permutation(len(tr_y)), CLIENTS_PER_SPECIALTY)
            te_parts = np.array_split(rng.permutation(len(te_y)), CLIENTS_PER_SPECIALTY)
            for tp, sp in zip(tr_parts, te_parts):
                tp, sp = tp[:TRAIN_CAP], sp[:TEST_CAP]
                nva = int(round(VAL_FRAC * len(tp)))
                va, tr = tp[:nva], tp[nva:]
                for name, idx, X, Y in (("tr", tr, tr_x, tr_y), ("va", va, tr_x, tr_y), ("te", sp, te_x, te_y)):
                    arrays[f"X{name}_{c}"] = extract(body, X[idx], device)
                    arrays[f"y{name}_{c}"] = Y[idx]
                meta["client_specialty"].append(s)
                meta["n_classes"].append(int(k))
                meta["n_train"].append(int(len(tr)))
                meta["n_val"].append(int(len(va)))
                meta["n_test"].append(int(len(sp)))
                c += 1
            print(f"  preset {preset}: {flag} done ({k} classes)", flush=True)
        meta["n_clients"] = c
        np.savez_compressed(os.path.join(args.out, f"preset_{preset}.npz"), meta=json.dumps(meta), **arrays)
        print(f"preset {preset}: {c} clients, train {sum(meta['n_train'])}, val {sum(meta['n_val'])}, "
              f"test {sum(meta['n_test'])}, {time.time() - t0:.0f} s", flush=True)
        print("  test sizes:", meta["n_test"], flush=True)


if __name__ == "__main__":
    main()
