"""
Transferability: adversarial examples crafted on NN1 (white-box) are fed to NN2.

The adversarial images are stored at 160x160, the input size of NN1. NN2 works
at 224x224, so they are upsampled. To separate the effect of the perturbation
from the loss of resolution, NN2 is also evaluated on the *clean* images
saved through the same pipeline (resize to 160, uint8 PNG): this is the
reference accuracy against which the transfer of the attack is measured.
"""

import numpy as np
import torch
from PIL import Image

from advfr.data import list_images, to_attack_tensor
from advfr.models import predict, preprocess_nn2


def nn2_predictions(nn2, paths, batch_size=64):
    out = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([preprocess_nn2(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
        out.append(predict(nn2, x, batch_size))
    return np.concatenate(out)


def nn2_reference_accuracy(nn2, label_map, items, batch_size=64):
    """NN2 accuracy on clean images reduced to 160x160 uint8, like the adversarial files (no perturbation)."""
    correct = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        imgs = []
        for path, _ in batch:
            x = to_attack_tensor(Image.open(path).convert("RGB")).numpy()
            imgs.append(Image.fromarray(np.transpose((x * 255).astype(np.uint8), (1, 2, 0))))
        pred = predict(nn2, torch.stack([preprocess_nn2(im) for im in imgs]), batch_size)
        correct += int(label_map.is_correct(pred, [c for _, c in batch]).sum())
    return 100 * correct / len(items)


def evaluate_folder_on_nn2(nn2, label_map, folder, structure="nested", target=None):
    """
    Accuracy of NN2 on a folder of adversarial examples (untargeted), or the
    rate at which NN2 predicts `target` (targeted transfer).
    """
    items = list_images(folder, structure)
    pred = nn2_predictions(nn2, [p for p, _ in items])
    if target is None:
        return {"nn2_accuracy": round(100 * label_map.is_correct(pred, [c for _, c in items]).mean(), 2),
                "total_images": len(items)}
    return {"nn2_target_rate": round(100 * np.mean(pred == target), 2), "total_images": len(items)}
