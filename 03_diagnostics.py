"""Step 3: what the leak-free model learns (Figs. 3 and 4 of the paper, per-class numbers in Section IV).

Part A (no training): pooled confusion matrices of P0 and P2 and per-class precision, recall and F1 for every protocol,
from the saved fold results of 02_protocols.py.
Part B: fold 0 of P0 and P2 retrained with the training history kept (same fold, seed, settings and frozen feature
selection), giving the learning curves, and ROC and precision-recall curves (micro-averaged as in the authors' code,
plus per-class ROC AUC).

Output: <out>/confusion_pooled.json, <out>/per_class.csv, <out>/curves_fold0.json. About 5 minutes on a T4 GPU.
GPU training is not bit-for-bit repeatable: the fold-0 accuracy can differ by a few points from 02_protocols.py.
"""
import glob, importlib, json, os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import SEED, args

P = importlib.import_module("02_protocols")

if __name__ == "__main__":
    a = args(__doc__.splitlines()[0])
    X_real, y_real, groups, CLASS_NAMES = P.load_data(a.data)

    # Part A: pooled confusion matrices and per-class metrics
    runs = {}
    for path in sorted(glob.glob(f"{a.out}/protocols/P*_fold*.json")):
        proto, rest = os.path.basename(path)[:-5].split("_", 1)
        runs.setdefault((proto, rest.rsplit("_fold", 1)[0].replace("_", " ")), []).append(json.load(open(path))["confusion"])
    pooled = {f"{p}/{m}": np.sum(c, axis=0).tolist() for (p, m), c in runs.items()}
    json.dump({"classes": CLASS_NAMES, "pooled": pooled}, open(f"{a.out}/confusion_pooled.json", "w"), indent=1)
    rows = []
    for key, cm in pooled.items():
        cm = np.asarray(cm, float); tp = np.diag(cm)
        prec = np.divide(tp, cm.sum(0), out=np.zeros(5), where=cm.sum(0) > 0)
        rec = np.divide(tp, cm.sum(1), out=np.zeros(5), where=cm.sum(1) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(5), where=(prec + rec) > 0)
        for c in range(5):
            rows.append({"protocol/model": key, "class": CLASS_NAMES[c], "precision": prec[c], "recall": rec[c], "f1": f1[c]})
    pd.DataFrame(rows).to_csv(f"{a.out}/per_class.csv", index=False)
    print(pd.DataFrame(rows).query("`protocol/model`.str.contains('TransNN')").round(3).to_string(index=False))

    # Part B: fold 0 of P0 and P2, with history
    X_aug, y_aug = P.smote_3x(X_real, y_real)
    FOLDS = P.splits(a.splits, X_real, y_real, groups, y_aug)
    X_aug_s = StandardScaler().fit_transform(X_aug)
    X_sel = X_aug_s[:, json.load(open(f"{a.splits}/selection_alldata.json"))]
    tr, te = FOLDS["P0_P1"][0]
    m0, h0 = P.train_their_model(X_sel[tr], y_aug[tr], X_sel[te], y_aug[te], history=True)
    out = {"classes": CLASS_NAMES, "fold": 0, "seed": SEED}
    proba = {"P0": (m0.predict(X_sel[te][:, None, :], verbose=0), y_aug[te], h0)}
    P.free()

    tr, te = FOLDS["P2"][0]
    fit, val = train_test_split(tr, test_size=0.1, stratify=y_real[tr], random_state=SEED)
    sc = StandardScaler().fit(X_real[fit])
    Xf, yf = P.smote_3x(sc.transform(X_real[fit]), y_real[fit])
    top = json.load(open(f"{a.splits}/P2_selection_fold0.json"))
    m2, h2 = P.train_their_model(Xf[:, top], yf, sc.transform(X_real[val])[:, top], y_real[val], history=True)
    proba["P2"] = (m2.predict(sc.transform(X_real[te])[:, top][:, None, :], verbose=0), y_real[te], h2)

    for tag, (pr, y, h) in proba.items():
        onehot = np.eye(5)[y]
        fpr, tpr, _ = roc_curve(onehot.ravel(), pr.ravel())
        prec, rec, _ = precision_recall_curve(onehot.ravel(), pr.ravel())
        per = {n: float(roc_auc_score(onehot[:, c], pr[:, c])) for c, n in enumerate(CLASS_NAMES)}
        out[tag] = {"history": h, "accuracy": float(accuracy_score(y, pr.argmax(1))),
                    "roc_micro": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))},
                    "roc_macro_auc": float(roc_auc_score(y, pr, multi_class="ovr", average="macro")),
                    "pr_micro": {"precision": prec.tolist(), "recall": rec.tolist(), "auc": float(auc(rec, prec))},
                    "roc_auc_per_class": per}
        print(f"{tag} fold 0: accuracy {out[tag]['accuracy']:.3f} | ROC AUC micro {out[tag]['roc_micro']['auc']:.3f} | "
              f"PR AUC micro {out[tag]['pr_micro']['auc']:.3f} | per class {({k: round(v, 3) for k, v in per.items()})}")
    json.dump(out, open(f"{a.out}/curves_fold0.json", "w"))
