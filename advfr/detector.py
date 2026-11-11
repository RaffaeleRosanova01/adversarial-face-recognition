"""
Adversarial example detector: a binary CNN (genuine = 0, adversarial = 1)
placed in front of NN1 as a gate.

Dataset: adversarial examples sampled from 40 attack configurations of
different strength (FGSM, BIM, PGD, CW, DeepFool; untargeted and targeted
"fixed"), balanced 1:1 with clean images of the same 100 identities. The
split is identity-disjoint (70 / 20 / 10 subjects), so the detector is
tested on faces it has never seen.
"""

import copy
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from advfr import config
from advfr.data import IMAGE_EXTENSIONS

FAMILIES = ["FGSM", "BIM", "PGD", "CW", "DeepFool"]

# Adversarial configurations included in the detector dataset: folder and sub-folder in
# ADVERSARIAL_DIR, layout, number of sampled images. Strong, medium and weak settings of
# every attack are mixed, so the detector also sees partially successful perturbations.
ATTACK_CONFIGS = [
    ("FGSM", "eps_0_100", "nested", 200), ("FGSM", "eps_0_010", "nested", 200), ("FGSM", "eps_0_005", "nested", 200),
    ("FGSM_Targeted_Fixed", "eps_0_030", "nested", 200), ("FGSM_Targeted_Fixed", "eps_0_100", "nested", 200),
    ("FGSM_Targeted_Fixed", "eps_0_010", "nested", 200),
    ("PGD_nri3_iter5", "eps_0_100", "nested", 200), ("PGD_nri5_iter5", "eps_0_010", "nested", 200),
    ("PGD_nri1_iter5", "eps_0_005", "nested", 200),
    ("PGD_Targeted_Fixed_nri1_iter40", "eps_0_100", "nested", 200),
    ("PGD_Targeted_Fixed_nri3_iter5", "eps_0_030", "nested", 200),
    ("PGD_Targeted_Fixed_nri5_iter5", "eps_0_010", "nested", 200),
    ("BIM_iter10", "eps_0_100", "nested", 200), ("BIM_iter5", "eps_0_010", "nested", 200),
    ("BIM_iter20", "eps_0_005", "nested", 200),
    ("BIM_Targeted_Fixed_iter10", "eps_0_100", "nested", 200), ("BIM_Targeted_Fixed_iter5", "eps_0_030", "nested", 200),
    ("BIM_Targeted_Fixed_iter10", "eps_0_010", "nested", 200),
] + [("CW_Untargeted", f"iter_{it}_lr_{lr}_c_0_0", "flat", 100) for it, lr in [
    (1, "0_01"), (1, "0_1"), (1, "0_05"), (5, "0_001"), (5, "0_01"), (5, "0_1"), (5, "0_05"),
    (10, "0_001"), (10, "0_01"), (10, "0_1"), (20, "0_01"), (20, "0_1")]
] + [("DeepFool", f"iter_{it}_eps_1e-06", "flat", n) for it, n in [
    (1, 99), (5, 97), (50, 96), (100, 96), (20, 16), (200, 98), (3, 98)]
] + [("DeepFool", f"1000_iter_{it}_eps_1e-06", "nested", 200) for it in (1, 5, 10)]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Detector(nn.Module):
    """
    ImageNet-pretrained backbone with a 2-way head, fully fine-tuned.
    Input: float tensor [N, 3, 160, 160] in [0, 1]; ImageNet normalisation is internal.
    """

    def __init__(self, backbone="efficientnet", pretrained=True):
        super().__init__()
        if backbone == "efficientnet":
            self.backbone = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 2)
        elif backbone == "mobilenet":
            self.backbone = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
            self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, 2)
        else:
            raise ValueError(backbone)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.backbone(self.normalize(x))

    def score(self, x):
        """Probability of the adversarial class (differentiable)."""
        return F.softmax(self.forward(x), dim=1)[:, 1]


def load_detector(backbone="efficientnet", path=None, device="cpu"):
    model = Detector(backbone, pretrained=False)
    model.load_state_dict(torch.load(path or config.DETECTOR_WEIGHTS[backbone], map_location=device))
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def attack_family(filename):
    """Attack family of an adversarial file name ('<attack folder>__<config>__...')."""
    name = os.path.basename(filename).split("__")[0]
    return next((f for f in FAMILIES if name.startswith(f)), None)


def class_id_of(filename):
    """Subject ID from detector file names: orig|extra__<id>__<file>, <attack>__<cfg>__<id>[__<file>]."""
    parts = os.path.basename(filename).split("__")
    if parts[0] in ("orig", "extra"):
        return parts[1]
    return parts[2] if len(parts) >= 4 else os.path.splitext(parts[2])[0]


def build_detector_dataset(seed=config.SEED, out_dir=config.DETECTOR_DATA_DIR):
    """Copies the sampled adversarial examples and an equal number of clean images into out_dir."""
    from advfr.data import list_images

    rng = random.Random(seed)
    adv_dir, gen_dir = os.path.join(out_dir, "adversarial"), os.path.join(out_dir, "genuine")
    os.makedirs(adv_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    n_adv = 0
    for folder, sub, structure, n in ATTACK_CONFIGS:
        items = list_images(os.path.join(config.ADVERSARIAL_DIR, folder, sub), structure)
        for path, cid in rng.sample(items, min(n, len(items))):
            name = f"{folder}__{sub}__{cid}__{os.path.basename(path)}" if structure == "nested" else \
                f"{folder}__{sub}__{os.path.basename(path)}"
            shutil.copy2(path, os.path.join(adv_dir, name))
            n_adv += 1

    originals = list_images(config.DATA_DIR_1000, "nested")
    for path, cid in rng.sample(originals, min(1000, len(originals))):
        shutil.copy2(path, os.path.join(gen_dir, f"orig__{cid}__{os.path.basename(path)}"))
    extra = list_images(config.EXTRA_GENUINE_DIR, "nested")
    for path, cid in rng.sample(extra, min(max(0, n_adv - len(originals)), len(extra))):
        shutil.copy2(path, os.path.join(gen_dir, f"extra__{cid}__{os.path.basename(path)}"))


def identity_disjoint_split(data_dir=config.DETECTOR_DATA_DIR, subjects_csv=config.SUBJECTS_CSV,
                            ratios=(0.7, 0.2), seed=config.SEED, out_dir=config.SPLITS_DIR):
    """Splits the 100 identities 70/20/10 and assigns every image of a subject to the same split."""
    ids = sorted(pd.read_csv(subjects_csv)["Class_ID"].astype(str).str.strip())
    per_subject = {cid: [] for cid in ids}
    for label, sub in ((0, "genuine"), (1, "adversarial")):
        folder = os.path.join(data_dir, sub)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMAGE_EXTENSIONS) and class_id_of(f) in per_subject:
                per_subject[class_id_of(f)].append((os.path.join(folder, f), label))

    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train, n_val = int(len(ids) * ratios[0]), int(len(ids) * ratios[1])
    groups = {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}
    os.makedirs(out_dir, exist_ok=True)
    splits = {}
    for name, members in groups.items():
        files = [f for cid in members for f in per_subject[cid]]
        rng.shuffle(files)
        splits[name] = files
        df = pd.DataFrame(files, columns=["path", "label"])
        df["class_id"] = df["path"].map(class_id_of)
        df.to_csv(os.path.join(out_dir, f"detector_{name}_split.csv"), index=False)
    return splits


def load_split(name, splits_dir=config.SPLITS_DIR, data_dir=config.DETECTOR_DATA_DIR):
    """Reads a split CSV, re-rooting the paths in data_dir (the CSVs may contain Windows paths)."""
    df = pd.read_csv(os.path.join(splits_dir, f"detector_{name}_split.csv"))
    out = []
    for path, label in zip(df["path"], df["label"]):
        fname = path.replace("\\", "/").split("/")[-1]
        out.append((os.path.join(data_dir, "adversarial" if label == 1 else "genuine", fname), int(label)))
    return out


class DetectorDataset(Dataset):
    """Images as [0, 1] tensors at 160x160 (genuine JPEGs are resized, adversarial PNGs already are)."""

    def __init__(self, files):
        self.files = files
        self.transform = transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def weighted_sampler(files, boost_families=("CW", "DeepFool"), boost=3.0):
    """
    Oversamples the adversarial examples of the hardest families to detect: CW and
    DeepFool produce minimal, smooth perturbations and were initially poorly detected.
    """
    weights = [boost if label == 1 and attack_family(p) in boost_families else 1.0 for p, label in files]
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def scores_and_labels(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(model.score(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def classification_metrics(scores, labels, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    return {
        "accuracy": 100 * float(np.mean(pred == labels)),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "auroc": roc_auc_score(labels, scores),
        "genuine_fpr": 100 * float(np.mean(pred[labels == 0] == 1)),
    }


def eer_threshold(scores, labels, grid=np.linspace(0.01, 0.99, 200)):
    """
    Threshold where FAR (adversarial accepted) equals FRR (genuine rejected).
    Returns (threshold, eer %).
    """
    far = np.array([100 * np.mean(scores[labels == 1] < t) for t in grid])
    frr = np.array([100 * np.mean(scores[labels == 0] >= t) for t in grid])
    i = int(np.argmin(np.abs(far - frr)))
    return float(grid[i]), float((far[i] + frr[i]) / 2)


def train_detector(backbone, train_files, val_files, out_path, epochs=50, patience=20, lr=1e-4,
                   weight_decay=1e-4, batch_size=256, weighted=True, device="cpu"):
    """
    Full fine-tuning with Adam and cross-entropy; keeps the weights with the lowest
    validation loss and stops after `patience` epochs without improvement.

    weighted=True oversamples CW and DeepFool examples (weighted_sampler); with
    weighted=False the training split is read in the (already shuffled) order of its CSV.
    """
    model = Detector(backbone, pretrained=True).to(device)
    train_ds = DetectorDataset(train_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2,
                              sampler=weighted_sampler(train_files) if weighted else None, shuffle=False)
    val_loader = DataLoader(DetectorDataset(val_files), batch_size=batch_size, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best_loss, best_state, wait = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss, scores, labels = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                val_loss += criterion(logits, y.to(device)).item()
                scores.append(F.softmax(logits, 1)[:, 1].cpu())
                labels.append(y)
        val_loss /= len(val_loader)
        m = classification_metrics(torch.cat(scores).numpy(), torch.cat(labels).numpy())
        history.append({"epoch": epoch, "train_loss": train_loss / len(train_loader), "val_loss": val_loss, **m})
        print(f"Epoch {epoch:02d} | train loss {history[-1]['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val acc {m['accuracy']:.2f}% | AUROC {m['auroc']:.4f} | {time.time() - start:.0f}s")
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
            torch.save(best_state, out_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
