"""Step 2: the published pipeline, then removing one leak at a time (Fig. 2 of the paper).

P0  the authors' code: SMOTE (3 x the largest class) on all data, standardization and feature selection on all data,
    6 random folds, the test fold used as validation data for early stopping.
P1  P0 with a separate validation set (10% of the training part) for early stopping.
P2  folds over the real recordings only; validation split, scaling, SMOTE and feature selection inside each training fold.
P3  P2 with 6 subject-wise folds (new people in every test fold).
KNN (k = 5) and logistic regression run on the same folds and features.

Folds and selected features are written to <splits>/ on the first run and read back afterwards, so reruns and other
users get the same splits. Output: <out>/protocols/*.json (one per protocol, model and fold) and <out>/protocols_summary.json.
About 1.5-2 hours on a Colab T4 GPU (recursive feature elimination is the slow part); resumable. If RAM runs
high, the script stops: restart and run it again.
"""
import json, os, time

import keras
import numpy as np
from imblearn.over_sampling import SMOTE
from keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import CLASSES5, FEATURE_COLS, SEED, args, cached, check_ram, free, load_feature_table


def smote_3x(X, y):
    """The authors' Data Augmentation/data_augmentation.py: every class to 3 x the largest class, k = 5, random_state 42."""
    cnt = np.bincount(y)
    return SMOTE(sampling_strategy={int(c): int(3 * cnt.max()) for c in np.unique(y)}, random_state=SEED).fit_resample(X, y)


def their_selection(X, y, k=70):
    """Their TransNN-MHA.py: SMOTE again (default strategy), then the raw scores of five methods summed; top 70."""
    Xr, yr = SMOTE(random_state=SEED).fit_resample(X, y)
    scores = np.zeros(Xr.shape[1])
    scores += mutual_info_classif(Xr, yr, random_state=SEED)
    scores += np.nan_to_num(f_classif(Xr, yr)[0])
    scores += RFE(LogisticRegression(max_iter=500), n_features_to_select=30).fit(Xr, yr).support_.astype(float)
    l1 = OneVsRestClassifier(LogisticRegression(penalty="l1", solver="liblinear", max_iter=500)).fit(Xr, yr)
    scores += np.abs(l1.estimators_[0].coef_[0])                 # first class vs the rest, as their code
    scores += RandomForestClassifier(random_state=SEED).fit(Xr, yr).feature_importances_
    return [int(i) for i in np.argsort(scores / 5)[-k:]]


def build_their_model(n_features, n_classes=5, seed=SEED):
    """6 encoder layers (32 heads x 64, dropout 0.2) on a sequence of length 1, as in their TransNN-MHA.py (23.2M parameters)."""
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(1, n_features)); x = inp
    for _ in range(6):
        a = layers.MultiHeadAttention(num_heads=32, key_dim=64, dropout=0.2)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([a, x]))
        x = layers.Dense(512, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(layers.Dense(512, activation="relu")(x))
    model = keras.Model(inp, layers.Dense(n_classes, activation="softmax")(x))
    model.compile(optimizer=keras.optimizers.Adam(3e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_their_model(Xtr, ytr, Xval, yval, history=False):
    model = build_their_model(Xtr.shape[-1])
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)]
    h = model.fit(Xtr[:, None, :], ytr, validation_data=(Xval[:, None, :], yval), epochs=110, batch_size=128,
                  callbacks=cb, verbose=0)
    return (model, {k: [float(v) for v in vals] for k, vals in h.history.items()}) if history else model


def scores_of(y_true, pred, extra=None):
    r = {"accuracy": float(accuracy_score(y_true, pred)), "macro_f1": float(f1_score(y_true, pred, average="macro")),
         "confusion": confusion_matrix(y_true, pred, labels=range(5)).tolist()}
    r.update(extra or {})
    return r


def load_data(data_dir):
    data = load_feature_table(f"{data_dir}/features")
    enc = LabelEncoder().fit(CLASSES5)                       # alphabetical class order, as their label encoder
    return (data[FEATURE_COLS].to_numpy(float), enc.transform(data["Target"]), data["subject"].to_numpy(),
            list(enc.classes_))


def splits(split_dir, X_real, y_real, groups, y_aug):
    """Folds: computed once with seed 42 and frozen in <splits>/folds.json."""
    def make():
        f_aug = list(StratifiedKFold(6, shuffle=True, random_state=SEED).split(np.zeros(len(y_aug)), y_aug))
        f_real = list(StratifiedKFold(6, shuffle=True, random_state=SEED).split(X_real, y_real))
        subj = np.random.default_rng(SEED).permutation(np.unique(groups))
        f_subj = [(np.where(~np.isin(groups, g))[0], np.where(np.isin(groups, g))[0]) for g in np.array_split(subj, 6)]
        as_list = lambda fs: [{"train": tr.tolist(), "test": te.tolist()} for tr, te in fs]
        return {"P0_P1": as_list(f_aug), "P2": as_list(f_real), "P3": as_list(f_subj)}
    F = cached(f"{split_dir}/folds.json", make)
    return {k: [(np.array(f["train"]), np.array(f["test"])) for f in v] for k, v in F.items()}


if __name__ == "__main__":
    a = args(__doc__.splitlines()[0])
    R = f"{a.out}/protocols"; os.makedirs(R, exist_ok=True)
    X_real, y_real, groups, _ = load_data(a.data)
    X_aug, y_aug = smote_3x(X_real, y_real)
    print(f"Real recordings: {len(y_real)} {np.bincount(y_real).tolist()} | after SMOTE on all data: {len(y_aug)} "
          f"({100 * (1 - len(y_real) / len(y_aug)):.1f}% synthetic)")
    FOLDS = splits(a.splits, X_real, y_real, groups, y_aug)
    for tr, te in FOLDS["P3"]:
        assert not set(groups[tr]) & set(groups[te])
    RES = {}

    def record(proto, model, k, res):
        RES.setdefault((proto, model), []).append(res)
        print(f"{proto} fold {k} {model:<20} accuracy {res['accuracy']:.3f} | macro-F1 {res['macro_f1']:.3f}", flush=True)

    def baselines(proto, k, Xtr, ytr, Xte, yte):
        for name, clf in (("KNN", KNeighborsClassifier(5)), ("Logistic regression", LogisticRegression(max_iter=3000))):
            record(proto, name, k, cached(f"{R}/{proto}_{name.replace(' ', '_')}_fold{k}.json",
                                          lambda clf=clf: scores_of(yte, clf.fit(Xtr, ytr).predict(Xte))))

    # P0 and P1: SMOTE, scaling and selection on all data
    X_aug_s = StandardScaler().fit_transform(X_aug)
    top = cached(f"{a.splits}/selection_alldata.json", lambda: their_selection(X_aug_s, y_aug))
    X_sel = X_aug_s[:, top]
    for k, (tr, te) in enumerate(FOLDS["P0_P1"]):
        def p0():
            m = train_their_model(X_sel[tr], y_aug[tr], X_sel[te], y_aug[te])          # test fold as validation
            r = scores_of(y_aug[te], m.predict(X_sel[te][:, None, :], verbose=0).argmax(1)); free(); return r
        record("P0", "TransNN-MHA", k, cached(f"{R}/P0_TransNN-MHA_fold{k}.json", p0))
        baselines("P0", k, X_sel[tr], y_aug[tr], X_sel[te], y_aug[te])

        def p1():
            fit, val = train_test_split(tr, test_size=0.1, stratify=y_aug[tr], random_state=SEED)
            m = train_their_model(X_sel[fit], y_aug[fit], X_sel[val], y_aug[val])
            r = scores_of(y_aug[te], m.predict(X_sel[te][:, None, :], verbose=0).argmax(1)); free(); return r
        record("P1", "TransNN-MHA", k, cached(f"{R}/P1_TransNN-MHA_fold{k}.json", p1))
        check_ram()

    # P2 and P3: everything learned inside the training fold; test folds hold real recordings only
    for proto in ("P2", "P3"):
        t0 = time.time()
        for k, (tr, te) in enumerate(FOLDS[proto]):
            fit, val = train_test_split(tr, test_size=0.1, stratify=y_real[tr], random_state=SEED)
            sc = StandardScaler().fit(X_real[fit])
            Xf, yf = smote_3x(sc.transform(X_real[fit]), y_real[fit])
            top = cached(f"{a.splits}/{proto}_selection_fold{k}.json", lambda: their_selection(Xf, yf))
            Xte, Xval = sc.transform(X_real[te])[:, top], sc.transform(X_real[val])[:, top]

            def inside():
                m = train_their_model(Xf[:, top], yf, Xval, y_real[val])
                r = scores_of(y_real[te], m.predict(Xte[:, None, :], verbose=0).argmax(1),
                              {"train_accuracy": float(accuracy_score(yf, m.predict(Xf[:, top][:, None, :], verbose=0).argmax(1)))})
                free(); return r
            record(proto, "TransNN-MHA", k, cached(f"{R}/{proto}_TransNN-MHA_fold{k}.json", inside))
            baselines(proto, k, Xf[:, top], yf, Xte, y_real[te])
            check_ram()
        print(f"{proto} done in {(time.time() - t0) / 60:.0f} min\n")

    summary = {f"{p}/{m}": {"accuracy_mean": float(np.mean([r["accuracy"] for r in rs])),
                            "accuracy_std": float(np.std([r["accuracy"] for r in rs])),
                            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in rs]))} for (p, m), rs in RES.items()}
    summary["counts"] = {"real": int(len(y_real)), "after_smote": int(len(y_aug)), "real_per_class": np.bincount(y_real).tolist()}
    json.dump(summary, open(f"{a.out}/protocols_summary.json", "w"), indent=2)
    print("Paper: 92.39% +/- 1.34% (reported). Ours:")
    for p in ("P0", "P1", "P2", "P3"):
        s = summary[f"{p}/TransNN-MHA"]; print(f"  {p}: {100 * s['accuracy_mean']:.1f}% +/- {100 * s['accuracy_std']:.1f}")
