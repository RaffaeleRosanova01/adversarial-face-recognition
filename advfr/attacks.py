"""
Adversarial attacks (ART) against NN1 and their evaluation.

Error-generic (untargeted) attacks are measured by the accuracy of NN1 on
the adversarial images; error-specific (targeted) attacks by the attack
success rate (ASR), i.e. the fraction of images classified as the target.

Budget: every attack is limited to an L-inf perturbation of at most 0.1 in
[0, 1]. FGSM, BIM and PGD enforce it by construction; Carlini & Wagner and
DeepFool do not take a budget, so the L-inf norm of each example is checked
afterwards: an example above the budget is an attack failure and the clean
image is evaluated in its place (fixed denominator, Croce et al. 2020).
"""

import os
import time

import numpy as np
from art.attacks.evasion import (BasicIterativeMethod, CarliniLInfMethod, DeepFool, FastGradientMethod,
                                 ProjectedGradientDescentPyTorch)

from advfr import config
from advfr.data import load_batch, save_png


class CarliniLInfMethodFixed(CarliniLInfMethod):
    """
    Carlini & Wagner L-inf attack with a corrected success check for targeted mode.

    In ART (verified on 1.15 - 1.20.1) `CarliniLInfMethod.generate` accepts a
    candidate when `argmax(prediction) != argmax(y)` regardless of `targeted`.
    In targeted mode `y` holds the *target* class, so the check keeps only
    candidates that did NOT reach the target and the targeted attack can never
    succeed (0% ASR). This subclass applies `==` in targeted mode; the
    untargeted behaviour is unchanged.
    """

    def _is_success(self, x_adv_batch, y_batch):
        pred = np.argmax(self.estimator.predict(x_adv_batch), axis=1)
        target = np.argmax(y_batch, axis=1)
        return (pred == target) if self.targeted else (pred != target)

    def generate(self, x, y=None, **kwargs):
        from art.utils import check_and_transform_label_format, get_labels_np_array

        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.astype(np.float32)
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        clip_min, clip_max = self.estimator.clip_values

        # Same search as ART: decreasing tau, increasing constant c (only the success check differs)
        for i in range(x.shape[0]):
            sample_done, tau, best = False, 1.0, 1.0
            while tau > 1.0 / 256.0 and not sample_done:
                sample_done = True
                const = self.initial_const
                while const < self.largest_const:
                    x_batch, y_batch = x[[i]], y[[i]]
                    candidate = self._generate_single(x_batch, y_batch, clip_min, clip_max, const=const, tau=tau)
                    delta = np.max(np.abs(candidate - x[i]))
                    if self._is_success(candidate, y_batch)[0] and delta < best:
                        x_adv[i], best, sample_done = candidate, delta, False
                    const *= self.const_factor
                tau = min(tau, np.max(np.abs(x_adv[i] - x[i]))) * self.decrease_factor
        return x_adv


def build_attack(name, classifier, targeted=False, **params):
    """
    name: fgsm | bim | pgd | cw | deepfool
    params: eps, max_iter, num_random_init, learning_rate, confidence, epsilon (DeepFool overshoot)
    For BIM and PGD the step size is eps / max_iter, as in the experiments.
    """
    if name == "fgsm":
        return FastGradientMethod(estimator=classifier, eps=params["eps"], targeted=targeted)
    if name == "bim":
        return BasicIterativeMethod(estimator=classifier, eps=params["eps"], eps_step=params["eps"] / params["max_iter"],
                                    max_iter=params["max_iter"], targeted=targeted, verbose=False)
    if name == "pgd":
        return ProjectedGradientDescentPyTorch(estimator=classifier, eps=params["eps"],
                                               eps_step=params["eps"] / params["max_iter"], max_iter=params["max_iter"],
                                               num_random_init=params.get("num_random_init", 0), targeted=targeted,
                                               verbose=False)
    if name == "cw":
        return CarliniLInfMethodFixed(classifier=classifier, targeted=targeted, max_iter=params["max_iter"],
                                      learning_rate=params["learning_rate"], confidence=params.get("confidence", 0.0),
                                      verbose=False)
    if name == "deepfool":
        if targeted:
            raise ValueError("DeepFool is an error-generic attack only.")
        return DeepFool(classifier=classifier, max_iter=params["max_iter"], epsilon=params.get("epsilon", 1e-6),
                        verbose=False)
    raise ValueError(f"Unknown attack: {name}")


# Attacks without an explicit L-inf budget: the budget is checked afterwards
UNBOUNDED_ATTACKS = {"cw", "deepfool"}
# Attacks generated one image at a time (as in the experiments, for memory and to avoid
# coupling between samples in ART's internal search)
PER_SAMPLE_ATTACKS = {"cw", "deepfool"}


def target_indices(classifier, x, mode):
    """Target class per image: a fixed identity, or the least likely class for the clean image."""
    if mode == "fixed":
        return np.full(len(x), config.FIXED_TARGET_CLASS)
    if mode == "least_likely":
        return np.argmin(classifier.predict(x), axis=1)
    raise ValueError(mode)


def run_attack(classifier, label_map, items, attack, params, targeted=False, target_mode="fixed",
               batch_size=64, save_dir=None, structure="nested"):
    """
    Generates adversarial examples for `items` [(path, class_id)] and evaluates NN1.

    Returns a dict with: accuracy (untargeted) or attack_success_rate (targeted),
    the counts, the number of examples over the L-inf budget and the mean L-inf
    norm of the valid examples. If save_dir is given the valid examples are saved
    as PNG (save_dir/<class_id>/<name>.png, or save_dir/<class_id>.png when flat).
    """
    start = time.time()
    hits, over_budget, linf = 0, 0, []
    step = 1 if attack in PER_SAMPLE_ATTACKS else batch_size
    for i in range(0, len(items), step):
        batch = items[i:i + step]
        x = load_batch([p for p, _ in batch])
        class_ids = [c for _, c in batch]
        attacker = build_attack(attack, classifier, targeted=targeted, **params)
        if targeted:
            targets = target_indices(classifier, x, target_mode)
            x_adv = attacker.generate(x=x, y=np.eye(config.NUM_CLASSES)[targets])
        else:
            x_adv = attacker.generate(x=x)

        norms = np.abs(x_adv - x).reshape(len(x), -1).max(axis=1)
        valid = norms <= config.EPSILON_MAX + 1e-5 if attack in UNBOUNDED_ATTACKS else np.ones(len(x), bool)
        over_budget += int((~valid).sum())
        linf += norms[valid].tolist()
        x_eval = np.where(valid[:, None, None, None], np.clip(x_adv, 0.0, 1.0), x)  # failed -> clean image

        pred = np.argmax(classifier.predict(x_eval), axis=1)
        if targeted:
            hits += int(np.sum(pred == targets))
        else:
            hits += int(label_map.is_correct(pred, class_ids).sum())

        if save_dir:
            for k, (path, cid) in enumerate(batch):
                if not valid[k]:
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(save_dir, f"{cid}.png") if structure == "flat" else \
                    os.path.join(save_dir, cid, f"{stem}.png")
                save_png(x_eval[k], out)

    n = len(items)
    result = {"total_images": n, "over_budget": over_budget,
              "mean_linf": round(float(np.mean(linf)), 4) if linf else 0.0,
              "time_elapsed_s": round(time.time() - start, 2)}
    if targeted:
        result.update(successful_attacks=hits, attack_success_rate=round(100 * hits / n, 2))
    else:
        result.update(correct_predictions=hits, accuracy=round(100 * hits / n, 2))
    return result
