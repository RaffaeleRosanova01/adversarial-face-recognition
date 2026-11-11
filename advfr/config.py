"""Paths and experiment constants (all paths are relative to the repository root)."""

import os

DATA_DIR = "data"
DATA_DIR_1000 = os.path.join(DATA_DIR, "dataset_100_subjects")   # 100 subjects x 10 images (nested folders)
DATA_DIR_100 = os.path.join(DATA_DIR, "dataset_1_per_person")    # 1 image per subject (flat), for CW and DeepFool
SUBJECTS_CSV = os.path.join(DATA_DIR, "selected_100_subjects.csv")
EXTRA_GENUINE_DIR = os.path.join(DATA_DIR, "extra_genuine")      # additional clean images for the detector dataset

WEIGHTS_DIR = "weights"
NN2_WEIGHTS = os.path.join(WEIGHTS_DIR, "resnet50_ft_weight.pkl")
LABELS_FILE = os.path.join(WEIGHTS_DIR, "rcmalli_vggface_labels_v2.npy")
LABELS_URL = "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy"
DETECTOR_WEIGHTS = {
    "mobilenet": os.path.join(WEIGHTS_DIR, "detector_mobilenet_v3.pth"),
    "efficientnet": os.path.join(WEIGHTS_DIR, "detector_efficientnet_b0.pth"),
}

ADVERSARIAL_DIR = "adversarial_data"     # generated adversarial examples (PNG, 160x160)
DETECTOR_DATA_DIR = "detector_dataset"   # genuine/ and adversarial/ images of the detector
SPLITS_DIR = "dataset_splits"            # identity-disjoint detector splits (CSV)
RESULTS_DIR = "results"

NUM_CLASSES = 8631          # VGGFace2 training identities
IMAGE_SIZE = 160            # NN1 input size; attacks operate in [0, 1] at this resolution
EPSILON_MAX = 0.1           # L-inf budget: 10% of the pixel range
FIXED_TARGET_CLASS = 2026   # target identity of the "fixed" error-specific attacks
SEED = 42

EPSILONS = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]

# Operating threshold of the EfficientNet-B0 detector (EER point on the validation set)
DETECTOR_THRESHOLD = 0.11
