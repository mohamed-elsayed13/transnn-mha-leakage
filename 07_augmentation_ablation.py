"""Step 7: does the leak-free result depend on the oversampling choice? (Sec. IV-B of the paper)

P2 and P3 are repeated with only the training data of each fold changed; folds, validation split, scaling, the frozen
feature selections, model and seed stay as in 02_protocols.py. Test folds hold whole, real recordings only.
  smote_3x       the authors' setting (3 x the largest class), the reference
  smote_auto     SMOTE only to balance the classes
  none           real recordings only
  none_weighted  real recordings only, with class weights
  crops          real recordings plus random crops of the same training runs (half a run each, at a random start;
                 4 per task run, 6 per baseline run, so the classes are balanced with real signal). ICA cleaning is
                 done once on the whole run, then the 258 features are computed on each crop.

Output: <data>/crops/S###.npz, <out>/augmentation/*.json, <out>/augmentation_summary.json, <out>/augmentation_paired.csv.
About 10 hours to build the crop features (ICA on every run, 3-10 min per subject), then about 1 hour of
training on a T4 GPU; resumable after a disconnect.
"""
import importlib, json, os, time

import mne
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.signal import iirnotch, filtfilt
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import SEED, args, cached, check_ram, edf_path, free, holm, load_feature_table

F1 = importlib.import_module("01_features")
P = importlib.import_module("02_protocols")
CONDITIONS = ["smote_3x", "smote_auto", "none", "none_weighted", "crops"]
CROP_FRAC, N_CROPS_TASK, N_CROPS_REST = 0.5, 4, 6


def clean(signals):
    """The authors' cleaning (as in 01_features.py), without the feature step."""
    x = F1.bandpass(signals, 0.5, 40)
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    ica = FastICA(n_components=x.shape[0], max_iter=1000, tol=0.001, random_state=0)
    comps = ica.fit_transform(x.T).T
    comps[np.where(kurtosis(comps, axis=1) > 10)[0], :] = 0
    x = ica.inverse_transform(comps.T).T
    b, a = iirnotch(w0=50.0, Q=30.0, fs=F1.FS)
    return filtfilt(b, a, x, axis=1)


def features(x):
    """The 258 features of 01_features.py on an already cleaned signal."""
    from scipy.integrate import trapezoid
    from scipy.signal import welch
    from scipy.stats import skew

    def band_power(sig, band):
        f, Pw = welch(sig, fs=F1.FS, nperseg=128)
        return trapezoid(Pw[(f >= band[0]) & (f <= band[1])])
    return np.concatenate([x.mean(1), x.var(1), skew(x, axis=1), kurtosis(x, axis=1),
                           [band_power(x[0], (8, 12)), band_power(x[0], (12, 30))]])


def subject_crops(data_dir, s, runs):
    path = f"{data_dir}/crops/S{s:03d}.npz"
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    F, R = [], []
    for run in runs:
        x = clean(mne.io.read_raw_edf(edf_path(data_dir, s, run), preload=True).get_data() * 1e6)
        n = x.shape[1]; L = int(CROP_FRAC * n)
        rng = np.random.default_rng([SEED, s, run])
        for start in rng.integers(0, n - L + 1, N_CROPS_REST if run in (1, 2) else N_CROPS_TASK):
            F.append(features(x[:, start:start + L])); R.append(run)
    d = {"crops": np.array(F), "crop_run": np.array(R)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **d)
    return d


if __name__ == "__main__":
    a = args(__doc__.splitlines()[0])
    mne.set_log_level("ERROR")
    R = f"{a.out}/augmentation"; os.makedirs(R, exist_ok=True)
    table = load_feature_table(f"{a.data}/features")
    X_real, y_real, groups, _ = P.load_data(a.data)
    runs_real = table["run"].to_numpy()
    X_aug, y_aug = P.smote_3x(X_real, y_real)
    FOLDS = P.splits(a.splits, X_real, y_real, groups, y_aug)

    t0 = time.time(); CROP_X = {}
    for s in sorted(set(groups)):
        d = subject_crops(a.data, s, sorted(set(runs_real[groups == s])))
        for run in np.unique(d["crop_run"]):
            CROP_X[(s, int(run))] = d["crops"][d["crop_run"] == run]
        print(f"S{s:03d} crops ready | {time.time() - t0:5.0f} s", flush=True)

    def training_set(cond, fit, sc):
        X, y = sc.transform(X_real[fit]), y_real[fit]
        if cond == "smote_3x":
            X, y = P.smote_3x(X, y)
        elif cond == "smote_auto":
            X, y = SMOTE(random_state=SEED).fit_resample(X, y)
        elif cond == "crops":
            extra = [(CROP_X[(g, r)], lab) for g, r, lab in zip(groups[fit], runs_real[fit], y) if (g, r) in CROP_X]
            X = np.vstack([X] + [sc.transform(c) for c, _ in extra])
            y = np.concatenate([y] + [np.full(len(c), lab) for c, lab in extra])
        cw = None
        if cond == "none_weighted":
            cnt = np.bincount(y, minlength=5); cw = {c: len(y) / (5 * cnt[c]) for c in range(5)}
        return X, y, cw

    RES = {}
    for proto in ("P2", "P3"):
        for cond in CONDITIONS:
            for k, (tr, te) in enumerate(FOLDS[proto]):
                def run():
                    fit, val = train_test_split(tr, test_size=0.1, stratify=y_real[tr], random_state=SEED)
                    sc = StandardScaler().fit(X_real[fit])
                    top = json.load(open(f"{a.splits}/{proto}_selection_fold{k}.json"))
                    Xtr, ytr, cw = training_set(cond, fit, sc)
                    model = P.build_their_model(len(top))
                    import keras
                    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
                          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)]
                    model.fit(Xtr[:, top][:, None, :], ytr, validation_data=(sc.transform(X_real[val])[:, top][:, None, :], y_real[val]),
                              epochs=110, batch_size=128, class_weight=cw, callbacks=cb, verbose=0)
                    pred = model.predict(sc.transform(X_real[te])[:, top][:, None, :], verbose=0).argmax(1)
                    r = {"accuracy": float(accuracy_score(y_real[te], pred)), "macro_f1": float(f1_score(y_real[te], pred, average="macro")),
                         "confusion": confusion_matrix(y_real[te], pred, labels=range(5)).tolist(), "n_train": int(len(ytr))}
                    free(); return r
                RES.setdefault((proto, cond), []).append(cached(f"{R}/{proto}_{cond}_fold{k}.json", run))
                check_ram()
            acc = [r["accuracy"] for r in RES[(proto, cond)]]
            print(f"{proto} {cond:<14} accuracy {np.mean(acc):.3f} +/- {np.std(acc):.3f} | {time.time() - t0:5.0f} s", flush=True)

    fa = lambda p, c: np.array([r["accuracy"] for r in RES[(p, c)]])
    rows = []
    for proto in ("P2", "P3"):
        for cond in CONDITIONS[1:]:
            d = fa(proto, cond) - fa(proto, "smote_3x"); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"protocol": proto, "condition vs smote_3x": cond, "difference": d.mean(), "ci_low": d.mean() - h,
                         "ci_high": d.mean() + h, "p": float(stats.ttest_rel(fa(proto, cond), fa(proto, "smote_3x")).pvalue)})
    comp = pd.DataFrame(rows); comp["p_holm"] = holm(comp["p"])
    comp.to_csv(f"{a.out}/augmentation_paired.csv", index=False)
    json.dump({f"{p}/{c}": {"accuracy_mean": float(fa(p, c).mean()), "accuracy_sd": float(fa(p, c).std())} for p, c in RES},
              open(f"{a.out}/augmentation_summary.json", "w"), indent=2)
    print("\n" + comp.round(4).to_string(index=False))
