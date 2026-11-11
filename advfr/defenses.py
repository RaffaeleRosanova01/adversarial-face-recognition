"""
Defences evaluated on the detector test split (10 unseen identities):

1. Detector gate: images flagged by the detector are rejected; the others
   are passed to the face recognition network (NN1 or NN2).
2. JPEG compression (input transformation, ART JpegCompression) applied
   before NN1, at quality 10 / 50 / 80.
"""

import numpy as np
import pandas as pd
import torch
from art.defences.preprocessor import JpegCompression
from art.estimators.classification import PyTorchClassifier
from PIL import Image
from torch import nn

from advfr import config
from advfr.data import to_attack_tensor
from advfr.detector import FAMILIES, attack_family, class_id_of
from advfr.models import predict, preprocess_nn1, preprocess_nn2

JPEG_QUALITIES = [10, 50, 80]


def _load(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    x_det = torch.stack([to_attack_tensor(im) for im in imgs])
    return imgs, x_det


# ---------------------------------------------------------------------------
# Detector gate
# ---------------------------------------------------------------------------

def system_evaluation(detector, nn1, nn2, label_map, files, threshold=config.DETECTOR_THRESHOLD, batch_size=64):
    """
    Runs every test image through (a) the bare networks and (b) detector + network.

    Metrics, for NN1 and NN2:
      - genuine accuracy: correct identifications among genuine images (baseline) or
        among the genuine images accepted by the detector (defended)
      - system accuracy: (genuine accepted and correctly identified + adversarial
        rejected) / all images. Baseline: (genuine correct + adversarial still
        correctly identified) / all images.
      - conditional accuracy: accuracy on the images accepted by the detector
    plus the adversarial block rate and the genuine false-positive rate, per family.
    """
    device = next(detector.parameters()).device
    rows = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        paths = [p for p, _ in batch]
        imgs, x_det = _load(paths)
        with torch.no_grad():
            scores = detector.score(x_det.to(device)).cpu().numpy()
        cids = [class_id_of(p) for p in paths]
        ok1 = label_map.is_correct(predict(nn1, torch.stack([preprocess_nn1(im) for im in imgs])), cids)
        ok2 = label_map.is_correct(predict(nn2, torch.stack([preprocess_nn2(im) for im in imgs])), cids)
        for (path, label), s, c1, c2 in zip(batch, scores, ok1, ok2):
            rows.append({"path": path, "label": label, "family": attack_family(path) if label else "Genuine",
                         "score": float(s), "blocked": bool(s >= threshold), "nn1_correct": bool(c1),
                         "nn2_correct": bool(c2)})
    df = pd.DataFrame(rows)
    return df, summarize_system(df)


def summarize_system(df):
    gen, adv = df[df.label == 0], df[df.label == 1]
    passed = df[~df.blocked]
    out = {
        "adversarial_block_rate": 100 * adv.blocked.mean(),
        "genuine_false_positive_rate": 100 * gen.blocked.mean(),
    }
    for net in ("nn1", "nn2"):
        col = f"{net}_correct"
        out[f"{net}_baseline_genuine_acc"] = 100 * gen[col].mean()
        out[f"{net}_baseline_system_acc"] = 100 * df[col].mean()
        out[f"{net}_defended_genuine_acc"] = 100 * gen[~gen.blocked][col].mean()
        out[f"{net}_defended_system_acc"] = 100 * (
            (gen[col] & ~gen.blocked).sum() + adv.blocked.sum()) / len(df)
        out[f"{net}_conditional_acc"] = 100 * passed[col].mean()
    return out


def per_family_block_rate(df):
    adv = df[df.label == 1]
    rates = adv.groupby("family")["blocked"].mean().mul(100)
    return rates.reindex([f for f in FAMILIES if f in rates.index])


# ---------------------------------------------------------------------------
# JPEG compression
# ---------------------------------------------------------------------------

def jpeg_classifier(nn1, quality, device=None):
    """NN1 wrapped in ART with JPEG compression as a pre-processing defence."""
    device = device or next(nn1.parameters()).device
    return PyTorchClassifier(
        model=nn1,
        clip_values=(0.0, 1.0),
        loss=nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(3, config.IMAGE_SIZE, config.IMAGE_SIZE),
        nb_classes=config.NUM_CLASSES,
        preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])),
        preprocessing_defences=[JpegCompression(clip_values=(0.0, 1.0), quality=quality, apply_fit=False,
                                                apply_predict=True, channels_first=True)],
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


def jpeg_evaluation(classifier, label_map, files, batch_size=64):
    """NN1 accuracy on genuine images and per attack family (adversarial), for one classifier."""
    rows = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        x = np.stack([to_attack_tensor(Image.open(p).convert("RGB")).numpy() for p, _ in batch])
        pred = classifier.predict(x, batch_size=batch_size).argmax(1)
        ok = label_map.is_correct(pred, [class_id_of(p) for p, _ in batch])
        for (path, label), p, c in zip(batch, pred, ok):
            fam = attack_family(path) if label else "Genuine"
            targeted = label == 1 and "Targeted_Fixed" in path.replace("\\", "/").split("/")[-1].split("__")[0]
            rows.append({"family": fam, "targeted": targeted, "correct": bool(c),
                         "hit_target": bool(p == config.FIXED_TARGET_CLASS)})
    df = pd.DataFrame(rows)
    summary = {"Genuine": 100 * df[df.family == "Genuine"].correct.mean()}
    adv = df[df.family != "Genuine"]
    for fam in FAMILIES:
        sub = adv[adv.family == fam]
        if len(sub):
            summary[f"{fam} accuracy"] = 100 * sub.correct.mean()
    tgt = adv[adv.targeted]
    if len(tgt):
        summary["Targeted ASR"] = 100 * tgt.hit_target.mean()
    return summary
