"""Clean accuracy of NN1 and NN2 on the 1000 test images (and NN2 on the 160x160 PNG reference)."""

import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advfr import config  # noqa: E402
from advfr.data import LabelMap, list_images  # noqa: E402
from advfr.models import get_device, load_nn1, load_nn2, predict, preprocess_nn1  # noqa: E402
from advfr.transferability import nn2_predictions, nn2_reference_accuracy  # noqa: E402

device = get_device()
label_map = LabelMap()
items = list_images(config.DATA_DIR_1000, "nested")
cids = [c for _, c in items]

nn1 = load_nn1(device)
pred = [predict(nn1, torch.stack([preprocess_nn1(Image.open(p).convert("RGB")) for p, _ in items[i:i + 64]]))
        for i in range(0, len(items), 64)]
import numpy as np  # noqa: E402
acc1 = 100 * label_map.is_correct(np.concatenate(pred), cids).mean()

nn2 = load_nn2(device=device)
acc2 = 100 * label_map.is_correct(nn2_predictions(nn2, [p for p, _ in items]), cids).mean()
ref2 = nn2_reference_accuracy(nn2, label_map, items)

print(f"NN1 clean accuracy:                         {acc1:.2f}%")
print(f"NN2 clean accuracy (original images):       {acc2:.2f}%")
print(f"NN2 clean accuracy (160x160 PNG reference): {ref2:.2f}%")
