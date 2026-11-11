"""
Evaluates the defences on the detector test split (10 unseen identities):
detector metrics, detector + NN1/NN2 system evaluation and JPEG compression.

    python scripts/evaluate_defenses.py --backbone efficientnet
"""

import argparse
import json
import os
import sys

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advfr import config  # noqa: E402
from advfr.data import LabelMap  # noqa: E402
from advfr.defenses import (JPEG_QUALITIES, jpeg_classifier, jpeg_evaluation, per_family_block_rate,  # noqa: E402
                            system_evaluation)
from advfr.detector import DetectorDataset, classification_metrics, load_detector, load_split, scores_and_labels  # noqa: E402,E501
from advfr.models import art_classifier, get_device, load_nn1, load_nn2  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "mobilenet"])
parser.add_argument("--threshold", type=float, default=config.DETECTOR_THRESHOLD)
parser.add_argument("--skip-jpeg", action="store_true")
args = parser.parse_args()

device = get_device()
test = load_split("test")
out_dir = os.path.join(config.RESULTS_DIR, "Defenses")
os.makedirs(out_dir, exist_ok=True)

detector = load_detector(args.backbone, device=device)
scores, labels = scores_and_labels(detector, DataLoader(DetectorDataset(test), batch_size=128), device)
metrics = classification_metrics(scores, labels, args.threshold)
print("Detector test metrics:", json.dumps({k: round(float(v), 4) for k, v in metrics.items()}))

label_map = LabelMap()
nn1, nn2 = load_nn1(device), load_nn2(device=device)
per_image, summary = system_evaluation(detector, nn1, nn2, label_map, test, args.threshold)
per_image.to_csv(os.path.join(out_dir, f"system_eval_{args.backbone}_per_image.csv"), index=False)
print("System evaluation:", json.dumps({k: round(float(v), 2) for k, v in summary.items()}, indent=1))
print("Block rate per attack family:\n", per_family_block_rate(per_image).round(2))

if not args.skip_jpeg:
    rows = [{"defence": "none", **jpeg_evaluation(art_classifier(nn1), label_map, test)}]
    for q in JPEG_QUALITIES:
        rows.append({"defence": f"JPEG q={q}", **jpeg_evaluation(jpeg_classifier(nn1, q), label_map, test)})
    df = pd.DataFrame(rows).round(2)
    df.to_csv(os.path.join(out_dir, "jpeg_defense_testset.csv"), index=False)
    print(df.to_string(index=False))
