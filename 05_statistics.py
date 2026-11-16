"""Step 5: statistics (Section III-F of the paper). No training; reads the saved confusion matrices.

1. Every result against chance: chance = the expected accuracy of predictions independent of the labels, from each
   confusion matrix; a permutation test shuffling true labels against the fixed predictions within each fold (10,000
   shuffles); a one-sample t-test of fold accuracies against fold chance; 95% intervals (Wilson over trials, t over folds);
   Holm correction across all results.
2. Paired fold-wise t-tests on identical folds: EEGNet vs CSP + LDA (seed 42), Holm-corrected.
Holm-adjusted p-values depend on the family of tests; this release tests only the results in the conference paper.

3. What EEGNet's 5-class accuracy consists of: rest recall, and limb and real/imagined accuracy among movement trials.

Output: <out>/stats_vs_chance.csv, <out>/stats_paired.csv, <out>/eegnet_breakdown.json. A few minutes on CPU.
"""
import glob, json, os, re

import numpy as np
import pandas as pd
from scipy import stats

from common import SEED, args, holm

N_PERM = 10_000
RNG = np.random.default_rng(SEED)


def expand(C):
    C = np.asarray(C, int); k = len(C)
    return np.repeat(np.arange(k), C.sum(1)), np.concatenate([np.repeat(np.arange(k), C[i]) for i in range(k)])


def chance_acc(confs):
    exp = n = 0.0
    for C in confs:
        C = np.asarray(C, float); m = C.sum(); exp += (C.sum(1) * C.sum(0)).sum() / m; n += m
    return exp / n


def perm_p(confs, n_perm=N_PERM, chunk=500):
    pairs = [expand(C) for C in confs]
    obs = sum(int((yt == yp).sum()) for yt, yp in pairs)
    null = np.zeros(n_perm, int)
    for s in range(0, n_perm, chunk):
        m = min(chunk, n_perm - s)
        for yt, yp in pairs:
            null[s:s + m] += (RNG.permuted(np.tile(yt, (m, 1)), axis=1) == yp).sum(1)
    return (1 + int((null >= obs).sum())) / (1 + n_perm)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def t_ci(x):
    x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


if __name__ == "__main__":
    a = args(__doc__.splitlines()[0])
    REC = {}
    for path in glob.glob(f"{a.out}/protocols/P?_*_fold*.json"):
        m = re.match(r"(P\d)_(.+)_fold(\d+)$", os.path.basename(path)[:-5])
        REC.setdefault(("5class", m.group(2).replace("_", " "), m.group(1)), {})[int(m.group(3))] = json.load(open(path))["confusion"]
    pat = re.compile(r"cross_(\w+?)_(1s|4s)_(csp|eegnet)(?:_seed(\d+))?_fold(\d+)$")
    for path in glob.glob(f"{a.out}/reference/cross_*.json"):
        m = pat.match(os.path.basename(path)[:-5])
        model = "CSP + LDA" if m.group(3) == "csp" else f"EEGNet seed {m.group(4)}"
        REC.setdefault((m.group(1), model, m.group(2)), {})[int(m.group(5))] = json.load(open(path))["confusion"]
    REC = {k: [v[i] for i in sorted(v)] for k, v in REC.items()}

    rows = []
    for (task, model, setting), confs in sorted(REC.items()):
        n = int(sum(np.sum(C) for C in confs)); k = int(sum(np.trace(C) for C in confs))
        acc_f = [np.trace(C) / np.sum(C) for C in confs]; ch_f = [chance_acc([C]) for C in confs]
        lo, hi = wilson(k, n); flo, fhi = t_ci(acc_f)
        rows.append({"task": task, "model": model, "setting": setting, "folds": len(confs), "test trials": n,
                     "accuracy": k / n, "chance": chance_acc(confs), "ci_trials_low": lo, "ci_trials_high": hi,
                     "ci_folds_low": flo, "ci_folds_high": fhi, "p_permutation": perm_p(confs),
                     "p_fold_t": float(stats.ttest_1samp(np.subtract(acc_f, ch_f), 0.0, alternative="greater").pvalue)})
    VS = pd.DataFrame(rows)
    VS["p_permutation_holm"] = holm(VS["p_permutation"]); VS["p_fold_t_holm"] = holm(VS["p_fold_t"])
    VS.to_csv(f"{a.out}/stats_vs_chance.csv", index=False)
    print(VS[["task", "model", "setting", "accuracy", "chance", "ci_folds_low", "ci_folds_high", "p_permutation_holm",
              "p_fold_t_holm"]].round(4).to_string(index=False))

    rows = []
    for task in ("5class", "real_vs_imag", "left_vs_right"):
        for length in ("1s", "4s"):
            e, c = REC.get((task, "EEGNet seed 42", length)), REC.get((task, "CSP + LDA", length))
            if e is None or c is None:
                continue
            fa = lambda confs: np.array([np.trace(C) / np.sum(C) for C in confs])
            d = fa(e) - fa(c); lo, hi = t_ci(d)
            rows.append({"comparison": "EEGNet vs CSP + LDA", "task": task, "length": length, "difference_pts": 100 * d.mean(),
                         "ci_low_pts": 100 * lo, "ci_high_pts": 100 * hi, "folds_first_higher": int((d > 0).sum()),
                         "p_paired_t": float(stats.ttest_rel(fa(e), fa(c)).pvalue)})
    PAIR = pd.DataFrame(rows); PAIR["p_holm"] = holm(PAIR["p_paired_t"])
    PAIR.to_csv(f"{a.out}/stats_paired.csv", index=False)
    print("\n" + PAIR.round(4).to_string(index=False))

    # What EEGNet's 5-class accuracy consists of (classes: 0 rest, 1 real fist, 2 imagined fist, 3 real feet, 4 imagined feet)
    C = np.sum(REC.get(("5class", "EEGNet seed 42", "4s"), [np.zeros((5, 5))]), axis=0)
    if C.sum():
        mv = C[1:, 1:]                                   # movement trials predicted as some movement
        limb = {1: 0, 2: 0, 3: 1, 4: 1}; real = {1: 0, 2: 1, 3: 0, 4: 1}
        same = lambda f: sum(mv[i - 1, j - 1] for i in range(1, 5) for j in range(1, 5) if f[i] == f[j]) / mv.sum()
        b = {"rest_recall": C[0, 0] / C[0].sum(), "movement_as_movement": mv.sum() / C[1:].sum(),
             "limb_correct_among_movement": same(limb), "real_vs_imagined_correct_among_movement": same(real)}
        json.dump(b, open(f"{a.out}/eegnet_breakdown.json", "w"), indent=2)
        print("\nEEGNet 5 classes, 4 s, seed 42:", {k: round(float(v), 3) for k, v in b.items()})
