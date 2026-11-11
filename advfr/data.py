"""
Test-set construction from VGGFace2, image loading and label handling.

Ground truth is matched by identity *name*: the predicted class index is
mapped to a name with the official VGGFace2 label list (rcmalli) and
compared with the name of the subject in `selected_100_subjects.csv`.
"""

import os
import random
import shutil
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from advfr import config

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Attack input: resize + [0, 1] tensor. Normalisation is done by the model wrappers.
to_attack_tensor = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])


def normalize_name(name):
    """Canonical form of an identity name, for comparison with the official labels."""
    return str(name).strip().lower().replace(" ", "_").replace(",_", "_").replace(",", "_")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def load_label_names(path=config.LABELS_FILE):
    """Names of the 8631 VGGFace2 classes, in model output order (downloaded if missing)."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(config.LABELS_URL, path)
    labels = np.load(path, allow_pickle=True)
    out = []
    for raw in labels:
        if isinstance(raw, np.ndarray):
            raw = raw.flatten()[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        out.append(normalize_name(str(raw).replace("b'", "").replace("'", "").replace("[", "").replace("]", "")))
    return out


class LabelMap:
    """Maps class indices and subject IDs to comparable identity names."""

    def __init__(self, subjects_csv=config.SUBJECTS_CSV, labels_file=config.LABELS_FILE):
        self.class_names = load_label_names(labels_file)
        subjects = pd.read_csv(subjects_csv)
        self.id_to_name = {
            str(r.Class_ID).strip(): normalize_name(r.Name) for r in subjects.itertuples()
        }

    def is_correct(self, pred_indices, class_ids):
        """Boolean array: prediction i matches the identity of subject class_ids[i]."""
        return np.array([
            self.class_names[int(p)] == self.id_to_name.get(str(c).strip(), "unknown")
            for p, c in zip(pred_indices, class_ids)
        ])


# ---------------------------------------------------------------------------
# Datasets on disk
# ---------------------------------------------------------------------------

def list_images(root, structure="nested"):
    """
    Lists (path, class_id) pairs.
      nested: root/<class_id>/<image>      (dataset_100_subjects, adversarial FGSM/BIM/PGD)
      flat:   root/<class_id>.<ext>        (dataset_1_per_person, adversarial CW/DeepFool)
    """
    items = []
    if structure == "nested":
        for class_id in sorted(os.listdir(root)):
            folder = os.path.join(root, class_id)
            if os.path.isdir(folder):
                items += [(os.path.join(folder, f), class_id.strip())
                          for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTENSIONS)]
    else:
        items = [(os.path.join(root, f), os.path.splitext(f)[0].strip())
                 for f in sorted(os.listdir(root)) if f.lower().endswith(IMAGE_EXTENSIONS)]
    return items


def load_batch(paths):
    """Loads images as a float32 array [N, 3, 160, 160] in [0, 1]."""
    return np.stack([to_attack_tensor(Image.open(p).convert("RGB")).numpy() for p in paths])


def save_png(x, path):
    """
    Saves an image in [0, 1] (C, H, W) as PNG.
    Values are truncated to uint8 as in the original experiments, so the
    stored files match the adversarial dataset used for the defences.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.transpose((x * 255).astype(np.uint8), (1, 2, 0)), mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Test-set preparation from the VGGFace2 archive
# ---------------------------------------------------------------------------

def prepare_test_sets(meta_csv, archive, images_per_subject=10, extra_per_subject=20, seed=config.SEED):
    """
    Builds the three image sets from the VGGFace2 metadata and archive:
      1. 100 subjects of the VGGFace2 training split with >= 10 images,
         sampled with a fixed seed -> selected_100_subjects.csv
      2. dataset_100_subjects: the first `images_per_subject` images of each subject
      3. dataset_1_per_person: one random image per subject (CW and DeepFool are too slow for 1000)
      4. extra_genuine: up to `extra_per_subject` further images per subject, disjoint
         from (2), used as additional clean samples for the detector dataset
    """
    df = pd.read_csv(meta_csv, on_bad_lines="skip", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if "Sample_Num" not in df.columns:
        df.columns = ["Class_ID", "Name", "Sample_Num", "Flag", "Gender"]
    df["Sample_Num"] = pd.to_numeric(df["Sample_Num"], errors="coerce")
    df["Flag"] = pd.to_numeric(df["Flag"], errors="coerce")
    candidates = df[(df["Sample_Num"] >= images_per_subject) & (df["Flag"] == 1)].copy()
    candidates["Class_ID"] = candidates["Class_ID"].astype(str).str.strip()
    subjects = candidates.sample(n=100, random_state=seed)
    subjects["Name"] = subjects["Name"].str.replace(",_", "_", regex=False).str.replace(",", "_", regex=False)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    subjects.to_csv(config.SUBJECTS_CSV, index=False)

    counts = {cid: 0 for cid in subjects["Class_ID"]}
    extra = {cid: 0 for cid in subjects["Class_ID"]}
    needed = len(counts) * (images_per_subject + extra_per_subject)
    done = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if done >= needed:
                break
            parts = member.name.split("/")
            if not member.isfile() or len(parts) < 2 or not parts[-1].lower().endswith(IMAGE_EXTENSIONS):
                continue
            cid = parts[-2].strip()
            if cid not in counts:
                continue
            if counts[cid] < images_per_subject:
                dest = os.path.join(config.DATA_DIR_1000, cid)
                counts[cid] += 1
            elif extra[cid] < extra_per_subject:
                dest = os.path.join(config.EXTRA_GENUINE_DIR, cid)
                extra[cid] += 1
            else:
                continue
            os.makedirs(dest, exist_ok=True)
            with tar.extractfile(member) as src, open(os.path.join(dest, parts[-1]), "wb") as out:
                shutil.copyfileobj(src, out)
            done += 1

    rng = random.Random(seed)
    os.makedirs(config.DATA_DIR_100, exist_ok=True)
    for cid in sorted(os.listdir(config.DATA_DIR_1000)):
        images = sorted(os.listdir(os.path.join(config.DATA_DIR_1000, cid)))
        chosen = rng.choice(images)
        shutil.copy2(os.path.join(config.DATA_DIR_1000, cid, chosen),
                     os.path.join(config.DATA_DIR_100, cid + os.path.splitext(chosen)[1]))
