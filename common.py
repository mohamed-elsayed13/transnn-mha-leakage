"""Shared settings and helpers for the conference release.

Every script takes --data (where the PhysioNet EDF files and cached features go) and --out (results).
On Colab, point both at a Google Drive folder so long runs survive a disconnect; every step is cached and resumable.
"""
import argparse, gc, json, os, time

import numpy as np
import pandas as pd

SEED = 42
CLASSES5 = ["Rest", "Real Right Fist", "Imaginary Right Fist", "Real Both Feet", "Imaginary Both Feet"]
FEATURE_COLS = ([f"Mean_Channel_{i+1}" for i in range(64)] + [f"Variance_Channel_{i+1}" for i in range(64)] +
                [f"Skewness_Channel_{i+1}" for i in range(64)] + [f"Kurtosis_Channel_{i+1}" for i in range(64)] +
                ["Alpha_Power", "Beta_Power"])

def args(description, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default="data", help="folder for EDF files and cached features (default: ./data)")
    p.add_argument("--out", default="results", help="folder for results (default: ./results)")
    p.add_argument("--splits", default="splits", help="folder with the frozen folds and feature selections")
    for a, kw in (extra or {}).items():
        p.add_argument(a, **kw)
    a = p.parse_args()
    for d in (a.data, a.out, a.splits):
        os.makedirs(d, exist_ok=True)
    return a

def cached(path, fn):
    """Load a finished result, or compute and save it. Makes every script resumable."""
    if os.path.exists(path):
        return json.load(open(path))
    res = fn()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f)
    return res

def check_ram(limit_gb=10.0):
    """TensorFlow keeps some memory after each training. Stop cleanly before the session crashes; rerun to continue."""
    try:
        import psutil
    except ImportError:
        return
    used = psutil.Process().memory_info().rss / 1e9
    if used > limit_gb:
        raise SystemExit(f"RAM {used:.1f} GB: restart the Python session and run the script again; "
                         "finished results are cached and are not recomputed.")

def edf_path(data_dir, subject, run):
    """Download one PhysioNet EEGMMIDB recording if needed and return its path."""
    from mne.datasets import eegbci
    for attempt in range(5):
        try:
            return eegbci.load_data(subject, [run], path=data_dir, update_path=False)[0]
        except (OSError, ConnectionError) as e:
            print(f"S{subject:03d}R{run:02d}: {type(e).__name__}, retrying in {5 * (attempt + 1)} s")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not get S{subject:03d}R{run:02d}")

def load_feature_table(feat_dir):
    """The authors' 5-class table: whole-run features, their duplicate removal, rest + the four T2 classes."""
    import glob
    rows = []
    for path in sorted(glob.glob(f"{feat_dir}/S*.npz")):
        s = int(os.path.basename(path)[1:4])
        d = np.load(path, allow_pickle=True)
        rows += [(s, r, l, *f) for f, l, r in zip(d["F"], d["labels"], d["runs"])]
    t = pd.DataFrame(rows, columns=["subject", "run", "Target"] + FEATURE_COLS)
    t = t.drop_duplicates(subset=FEATURE_COLS + ["Target"]).reset_index(drop=True)   # as in their code
    return t[t["Target"].isin(CLASSES5)].reset_index(drop=True)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p)
    adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def free():
    import keras
    keras.backend.clear_session(); gc.collect()
