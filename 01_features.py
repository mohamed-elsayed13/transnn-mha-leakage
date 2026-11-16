"""Step 1: the authors' feature extraction (DSP/Final_DSP.py of github.com/madibabaiasl/EEGIntentPaper), re-implemented.

For each of the 14 runs of each of the 109 subjects, on the WHOLE recording: band-pass 0.5-40 Hz (5th-order Butterworth),
z-score per channel, FastICA with 64 components (components with kurtosis > 10 removed), 50 Hz notch, then 258 features:
mean, variance, skewness and kurtosis of each channel, plus alpha (8-12 Hz) and beta (12-30 Hz) power of channel 1.
Their label mapping is kept exactly (runs 1-2 = Rest; in task runs T1 gives the left/both-fists label, anything else the
right-fist/both-feet label), as is their assumption of 160 Hz for every subject.

Output: <data>/features/S###.npz (one per subject, resumable). About 1 hour on Colab (ICA on 1,526 recordings).
"""
import os, time

import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import kurtosis, skew
from sklearn.decomposition import FastICA

from common import args, edf_path, load_feature_table, CLASSES5

FS = 160


def bandpass(x, lo, hi, order=5):
    b, a = butter(order, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    return filtfilt(b, a, x, axis=1)


def their_features(signals):
    x = bandpass(signals, 0.5, 40)
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    ica = FastICA(n_components=x.shape[0], max_iter=1000, tol=0.001, random_state=0)
    comps = ica.fit_transform(x.T).T
    comps[np.where(kurtosis(comps, axis=1) > 10)[0], :] = 0
    x = ica.inverse_transform(comps.T).T
    b, a = iirnotch(w0=50.0, Q=30.0, fs=FS)
    x = filtfilt(b, a, x, axis=1)

    def band_power(sig, band):
        f, P = welch(sig, fs=FS, nperseg=128)
        return trapezoid(P[(f >= band[0]) & (f <= band[1])])
    return np.concatenate([x.mean(1), x.var(1), skew(x, axis=1), kurtosis(x, axis=1),
                           [band_power(x[0], (8, 12)), band_power(x[0], (12, 30))]])


def map_labels(event_type, run):
    if run in (1, 2): return "Rest"
    if run in (3, 7, 11): return "Real Left Fist" if event_type == 1 else "Real Right Fist"
    if run in (4, 8, 12): return "Imaginary Left Fist" if event_type == 1 else "Imaginary Right Fist"
    if run in (5, 9, 13): return "Real Both Fists" if event_type == 1 else "Real Both Feet"
    if run in (6, 10, 14): return "Imaginary Both Fists" if event_type == 1 else "Imaginary Both Feet"
    return "Unknown"


def subject_rows(data_dir, feat_dir, s):
    path = f"{feat_dir}/S{s:03d}.npz"
    if os.path.exists(path):
        return
    F, labels, runs = [], [], []
    for run in range(1, 15):
        raw = mne.io.read_raw_edf(edf_path(data_dir, s, run), preload=True)
        feats = their_features(raw.get_data() * 1e6)        # one feature vector per whole run
        seen = []
        for d in raw.annotations.description:               # every event type of the run gets a row with the same features
            et = {"T0": 0, "T1": 1, "T2": 2}.get(d, -1)
            if et in seen:
                continue
            seen.append(et)
            F.append(feats); labels.append(map_labels(et, run)); runs.append(run)
    np.savez_compressed(path, F=np.array(F), labels=np.array(labels), runs=np.array(runs))


if __name__ == "__main__":
    a = args(__doc__.splitlines()[0])
    mne.set_log_level("ERROR")
    feat_dir = f"{a.data}/features"; os.makedirs(feat_dir, exist_ok=True)
    t0 = time.time()
    for s in range(1, 110):
        subject_rows(a.data, feat_dir, s)
        print(f"S{s:03d} done | {time.time() - t0:5.0f} s", flush=True)
    data = load_feature_table(feat_dir)
    counts = data["Target"].value_counts()
    print("\nReal recordings per class:", counts.reindex(CLASSES5).to_dict(), "| total", len(data))
    print(f"SMOTE to 3 x the largest class gives {5 * 3 * counts.max()} samples (paper: 4,905)")
