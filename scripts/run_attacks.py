"""
Security evaluation of NN1: runs one attack over a grid of parameters and saves
results/<name>/<name>_results.csv, the security evaluation curves and (optionally)
the adversarial examples in adversarial_data/.

Examples:
    python scripts/run_attacks.py fgsm
    python scripts/run_attacks.py bim --max-iter 1 5 10 20
    python scripts/run_attacks.py pgd --max-iter 5 --random-init 1 3 5 --targeted fixed
    python scripts/run_attacks.py cw --max-iter 1 5 10 20 --lr 0.01 0.1
    python scripts/run_attacks.py deepfool --max-iter 1 3 5 10 20 50 100 200
"""

import argparse
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advfr import config  # noqa: E402
from advfr.attacks import PER_SAMPLE_ATTACKS, run_attack  # noqa: E402
from advfr.data import LabelMap, list_images  # noqa: E402
from advfr.models import art_classifier, load_nn1  # noqa: E402
from advfr.plots import security_evaluation_curve  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("attack", choices=["fgsm", "bim", "pgd", "cw", "deepfool"])
parser.add_argument("--targeted", choices=["fixed", "least_likely"], default=None)
parser.add_argument("--eps", type=float, nargs="+", default=config.EPSILONS)
parser.add_argument("--max-iter", type=int, nargs="+", default=[10])
parser.add_argument("--random-init", type=int, nargs="+", default=[1])
parser.add_argument("--lr", type=float, nargs="+", default=[0.01])
parser.add_argument("--confidence", type=float, default=0.0)
parser.add_argument("--subset", choices=["1000", "100"], default=None,
                    help="test images (default: 1000 for gradient attacks, 100 for CW/DeepFool)")
parser.add_argument("--save", action="store_true", help="store the adversarial examples")
parser.add_argument("--clean-accuracy", type=float, default=85.1, help="NN1 clean accuracy (curve origin)")
parser.add_argument("--limit", type=int, default=None, help="use only the first N images (quick test)")
args = parser.parse_args()

subset = args.subset or ("100" if args.attack in PER_SAMPLE_ATTACKS else "1000")
structure = "flat" if subset == "100" else "nested"
items = list_images(config.DATA_DIR_100 if subset == "100" else config.DATA_DIR_1000, structure)[:args.limit]

name = args.attack.upper().replace("DEEPFOOL", "DeepFool")
if args.targeted:
    name += "_Targeted_" + ("Fixed" if args.targeted == "fixed" else "LeastLikely")
elif args.attack == "cw":
    name += "_Untargeted"

label_map = LabelMap()
classifier = art_classifier(load_nn1())

if args.attack == "fgsm":
    grid = [{"eps": e} for e in args.eps]
elif args.attack == "bim":
    grid = [{"eps": e, "max_iter": it} for it, e in itertools.product(args.max_iter, args.eps)]
elif args.attack == "pgd":
    grid = [{"eps": e, "max_iter": it, "num_random_init": r}
            for r, it, e in itertools.product(args.random_init, args.max_iter, args.eps)]
elif args.attack == "cw":
    grid = [{"max_iter": it, "learning_rate": lr, "confidence": args.confidence}
            for it, lr in itertools.product(args.max_iter, args.lr)]
else:
    grid = [{"max_iter": it, "epsilon": 1e-6} for it in args.max_iter]


def config_folder(p):
    tag = lambda v: str(v).replace(".", "_")  # noqa: E731
    if args.attack == "fgsm":
        return os.path.join(name, f"eps_{p['eps']:.3f}".replace(".", "_"))
    if args.attack == "bim":
        return os.path.join(f"{name}_iter{p['max_iter']}", f"eps_{p['eps']:.3f}".replace(".", "_"))
    if args.attack == "pgd":
        return os.path.join(f"{name}_nri{p['num_random_init']}_iter{p['max_iter']}",
                            f"eps_{p['eps']:.3f}".replace(".", "_"))
    if args.attack == "cw":
        return os.path.join(name, f"iter_{p['max_iter']}_lr_{tag(p['learning_rate'])}_c_{tag(p['confidence'])}")
    return os.path.join(name, f"iter_{p['max_iter']}_eps_1e-06")


rows = []
for params in grid:
    save_dir = os.path.join(config.ADVERSARIAL_DIR, config_folder(params)) if args.save else None
    res = run_attack(classifier, label_map, items, args.attack, params, targeted=bool(args.targeted),
                     target_mode=args.targeted or "fixed", save_dir=save_dir, structure=structure)
    rows.append({"attack_name": name, **params, **res})
    metric = "attack_success_rate" if args.targeted else "accuracy"
    print(f"{name} {params} -> {metric} {res[metric]}% (over budget: {res['over_budget']})")

out_dir = os.path.join(config.RESULTS_DIR, name)
os.makedirs(out_dir, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, f"{name}_results.csv"), index=False)

if "eps" in df:
    metric = "attack_success_rate" if args.targeted else "accuracy"
    group = [c for c in ("max_iter", "num_random_init") if c in df]
    curves = {"": df} if not group else {
        ", ".join(f"{g}={v}" for g, v in zip(group, k if isinstance(k, tuple) else (k,))): d
        for k, d in df.groupby(group)}
    curves = {k or name: (d["eps"].tolist(), d[metric].tolist()) for k, d in curves.items()}
    security_evaluation_curve(curves, os.path.join(out_dir, f"{name}_SEC.png"), title=name,
                              baseline=args.clean_accuracy, metric="asr" if args.targeted else "accuracy")
print(f"[INFO] results in {out_dir}")
