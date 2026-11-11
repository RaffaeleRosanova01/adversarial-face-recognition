"""
Builds the detector dataset (if missing), the identity-disjoint split and trains a detector.

    python scripts/train_detector.py --backbone efficientnet
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advfr import config  # noqa: E402
from advfr.detector import (build_detector_dataset, eer_threshold, identity_disjoint_split, load_split,  # noqa: E402
                            scores_and_labels, train_detector, DetectorDataset)
from advfr.models import get_device  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "mobilenet"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--weighted", action="store_true", help="oversample CW / DeepFool examples")
args = parser.parse_args()

if not os.path.isdir(os.path.join(config.DETECTOR_DATA_DIR, "adversarial")):
    build_detector_dataset()
if not os.path.exists(os.path.join(config.SPLITS_DIR, "detector_train_split.csv")):
    identity_disjoint_split()
train, val = load_split("train"), load_split("val")

device = get_device()
os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
out = config.DETECTOR_WEIGHTS[args.backbone]
model, history = train_detector(args.backbone, train, val, out, epochs=args.epochs, patience=args.patience,
                                weighted=args.weighted, device=device)
os.makedirs(os.path.join(config.RESULTS_DIR, "training_results"), exist_ok=True)
history.to_csv(os.path.join(config.RESULTS_DIR, "training_results", f"detector_{args.backbone}_history.csv"),
               index=False)

scores, labels = scores_and_labels(model, DataLoader(DetectorDataset(val), batch_size=256), device)
threshold, eer = eer_threshold(scores, labels)
print(f"[INFO] weights: {out} | validation EER {eer:.2f}% at threshold {threshold:.3f}")
