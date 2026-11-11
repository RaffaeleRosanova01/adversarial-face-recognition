"""
Builds the test images from VGGFace2 (metadata CSV + training archive):
data/selected_100_subjects.csv, data/dataset_100_subjects, data/dataset_1_per_person, data/extra_genuine.

    python scripts/prepare_data.py --meta identity_meta.csv --archive vggface2_train.tar.gz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advfr.data import prepare_test_sets  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--meta", required=True, help="VGGFace2 identity_meta.csv")
parser.add_argument("--archive", required=True, help="VGGFace2 training archive (.tar / .tar.gz)")
args = parser.parse_args()
prepare_test_sets(args.meta, args.archive)
print("[INFO] test sets written to data/")
