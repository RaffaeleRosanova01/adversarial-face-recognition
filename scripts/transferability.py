"""
Transferability of the stored adversarial examples (crafted on NN1) to NN2.

    python scripts/transferability.py FGSM                        # untargeted, nested folders
    python scripts/transferability.py PGD_Targeted_Fixed_nri1_iter40 --target 2026
    python scripts/transferability.py DeepFool --flat
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advfr import config  # noqa: E402
from advfr.data import LabelMap, list_images  # noqa: E402
from advfr.models import load_nn2  # noqa: E402
from advfr.transferability import evaluate_folder_on_nn2, nn2_reference_accuracy  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("folder", help="attack folder inside adversarial_data/")
parser.add_argument("--flat", action="store_true", help="CW / DeepFool layout (one image per subject)")
parser.add_argument("--target", type=int, default=None, help="target class of a targeted attack")
args = parser.parse_args()

label_map = LabelMap()
nn2 = load_nn2()
structure = "flat" if args.flat else "nested"
reference = nn2_reference_accuracy(
    nn2, label_map, list_images(config.DATA_DIR_100 if args.flat else config.DATA_DIR_1000, structure))
print(f"NN2 reference accuracy (clean images, 160x160 PNG): {reference:.2f}%")

root = os.path.join(config.ADVERSARIAL_DIR, args.folder)
rows = []
for sub in sorted(os.listdir(root)):
    res = evaluate_folder_on_nn2(nn2, label_map, os.path.join(root, sub), structure, args.target)
    rows.append({"attack_name": args.folder, "config": sub, "nn2_reference_accuracy": round(reference, 2), **res})
    print(rows[-1])
out_dir = os.path.join(config.RESULTS_DIR, f"{args.folder}_Transferability")
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{args.folder}_transferability_results.csv"), index=False)
