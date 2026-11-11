"""
Adaptive PGD against detector + NN1 on the 10 identities of the detector test split
(100 clean images never seen by the detector).

    python scripts/adaptive_attack.py --steps 40 --eps 0.01 0.03 0.1 --lams 0 1
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advfr import config  # noqa: E402
from advfr.adaptive import run_adaptive_experiment, true_class_indices  # noqa: E402
from advfr.data import LabelMap, list_images, load_batch  # noqa: E402
from advfr.detector import load_detector  # noqa: E402
from advfr.models import get_device, load_nn1  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "mobilenet"])
parser.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.03, 0.1])
parser.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0])
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--limit", type=int, default=None, help="use only the first N images (quick test)")
parser.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "Adaptive_Attack", "adaptive_pgd.csv"))
args = parser.parse_args()

torch.manual_seed(config.SEED)
device = get_device()
test_ids = set(pd.read_csv(os.path.join(config.SPLITS_DIR, "detector_test_split.csv"))["class_id"])
items = [it for it in list_images(config.DATA_DIR_1000, "nested") if it[1] in test_ids][:args.limit]
label_map = LabelMap()
y = torch.tensor(true_class_indices(label_map, [c for _, c in items]))
assert (y >= 0).all(), "some test identities are not NN1 classes"
x = torch.from_numpy(load_batch([p for p, _ in items]))
print(f"[INFO] {len(items)} images of {len(test_ids)} test identities, device {device}")

nn1 = load_nn1(device)
detector = load_detector(args.backbone, device=device)
for p in list(nn1.parameters()) + list(detector.parameters()):
    p.requires_grad_(False)

rows = run_adaptive_experiment(nn1, detector, x, y, args.eps, args.lams, args.steps)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"[INFO] saved {args.out}")
