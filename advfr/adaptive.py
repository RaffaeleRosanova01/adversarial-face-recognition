"""
Adaptive attack against the defended system (detector + NN1).

The detector was trained and evaluated on adversarial examples that ignore
it. An attacker who knows the defence can include it in the objective
(Carlini & Wagner, 2017; Tramer et al., 2020). Here untargeted PGD maximises

    CE(NN1(x), y) + lam * log(1 - D(x))

i.e. it pushes NN1 away from the true identity while keeping the detector
score D(x) low. lam = 0 is the standard (non-adaptive) PGD.
"""

import numpy as np
import torch
import torch.nn.functional as F

from advfr import config


def true_class_indices(label_map, class_ids):
    """NN1 output index of each subject (-1 if the identity is not among the 8631 classes)."""
    index = {name: i for i, name in enumerate(label_map.class_names)}
    return [index.get(label_map.id_to_name.get(c, ""), -1) for c in class_ids]


def adaptive_pgd(nn1, detector, x, y, eps=config.EPSILON_MAX, steps=40, step_size=None, lam=1.0,
                 random_init=True, seed=config.SEED):
    """
    L-inf PGD on images in [0, 1]. Both networks must be on the same device.
    Returns the adversarial batch.
    """
    device = x.device
    step_size = step_size or 2.5 * eps / steps
    g = torch.Generator(device="cpu").manual_seed(seed)
    delta = (torch.rand(x.shape, generator=g) * 2 - 1).to(device) * eps if random_init else torch.zeros_like(x)
    x_adv = (x + delta).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = nn1((x_adv - 0.5) / 0.5)
        loss = F.cross_entropy(logits, y)
        if lam > 0:
            p_adv = F.softmax(detector(x_adv), dim=1)[:, 1]
            loss = loss + lam * torch.log(1 - p_adv + 1e-12).mean()
        grad, = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


@torch.no_grad()
def evaluate_system(nn1, detector, x, y, threshold=config.DETECTOR_THRESHOLD):
    """
    Per-image outcome on the defended system:
      detected        - rejected by the detector
      misclassified   - NN1 does not return the true identity
      bypass          - not detected AND misclassified (attack succeeds end-to-end)
    """
    pred = nn1((x - 0.5) / 0.5).argmax(1)
    score = detector.score(x)
    detected = score >= threshold
    mis = pred != y
    return {
        "detection_rate": 100 * detected.float().mean().item(),
        "nn1_accuracy": 100 * (~mis).float().mean().item(),
        "bypass_rate": 100 * (mis & ~detected).float().mean().item(),
        "mean_score": score.mean().item(),
    }


def run_adaptive_experiment(nn1, detector, x, y, eps_list, lams=(0.0, 1.0), steps=40, batch_size=25,
                            threshold=config.DETECTOR_THRESHOLD):
    """Security evaluation of the defended system for every (lambda, epsilon) pair."""
    rows = []
    device = next(nn1.parameters()).device
    clean = evaluate_system(nn1, detector, x.to(device), y.to(device), threshold)
    rows.append({"lambda": None, "epsilon": 0.0, **{k: round(v, 4) for k, v in clean.items()}})
    for lam in lams:
        for eps in eps_list:
            outs = []
            for i in range(0, len(x), batch_size):
                xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
                outs.append(adaptive_pgd(nn1, detector, xb, yb, eps=eps, steps=steps, lam=lam))
            # 8-bit image, as it would be stored and sent: the clean images are on the 1/255 grid,
            # so truncating the perturbation towards zero keeps it within the budget
            delta = torch.cat(outs).cpu() - x
            x_adv = (x + torch.trunc(delta * 255 + 1e-4 * torch.sign(delta)) / 255).clamp(0, 1).to(device)
            res = evaluate_system(nn1, detector, x_adv, y.to(device), threshold)
            res["max_linf"] = (x_adv.cpu() - x).abs().amax(dim=(1, 2, 3)).max().item()
            rows.append({"lambda": lam, "epsilon": eps, **{k: round(v, 4) for k, v in res.items()}})
            print(f"lam={lam:<5} eps={eps:<6} detected {res['detection_rate']:5.1f}% | "
                  f"NN1 acc {res['nn1_accuracy']:5.1f}% | bypass {res['bypass_rate']:5.1f}%")
    return rows


def as_numpy_images(x):
    """[N, 3, H, W] in [0, 1] -> uint8 HWC images (what would be stored on disk)."""
    return (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
