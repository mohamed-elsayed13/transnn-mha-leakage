"""Step 4: leak-free reference decoders on unseen subjects (Fig. 5, Table IV and Section IV-C of the paper).

CSP + LDA (8 components, Ledoit-Wolf shrinkage, 8-30 Hz) and EEGNet-8,2 (Lawhern et al., 2018; 1-35 Hz), on cue-locked
trials of the 106 subjects recorded at 160 Hz, resampled to 80 Hz, as 4-second trials and their first second.
Tasks: the paper's 5 classes (rest from T0, balanced to one movement class; chance 20%), real vs imagined movement and
imagined left vs right fist (chance 50%). Evaluation: 5 subject-wise folds; CSP filters, LDA, input scaling and network
weights are fitted on training subjects only; EEGNet stops early on 10% of the training subjects.
EEGNet is trained with seed 42 for every task and length, and with seeds 7 and 123 on the 4-second trials.

Output: <data>/trials/S###.npz (cached trials), <out>/reference/*.json, <out>/reference_summary.json.
About 3-4 hours on a Colab T4 GPU; resumable. If RAM runs high, the script stops: restart and run it again.
"""
import gc, json, os, time

import keras
import mne
import numpy as np
from keras import constraints, layers
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from common import SEED, args, cached, check_ram, edf_path

SUBJECTS = [s for s in range(1, 110) if s not in (88, 92, 100)]       # three subjects recorded at 128 Hz are skipped
TASK_RUNS = list(range(3, 15))
FS, TRIAL_S = 80, 4.0
REAL_FIST, IMAG_FIST, REAL_FEET, IMAG_FEET = [3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]
BANDS = {"mi": (8.0, 30.0), "bb": (1.0, 35.0)}
TASKS = {"5class": 5, "real_vs_imag": 2, "left_vs_right": 2}
LENGTHS = {"1s": int(1 * FS), "4s": int(TRIAL_S * FS)}
SEEDS_4S = [42, 7, 123]
DATA = {}


def subject_trials(data_dir, s):
    path = f"{data_dir}/trials/S{s:03d}.npz"
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    out = {"mi": [], "bb": [], "run": [], "event": []}
    n = int(TRIAL_S * FS)
    for run in TASK_RUNS:
        raw = mne.io.read_raw_edf(edf_path(data_dir, s, run), preload=True)
        mne.datasets.eegbci.standardize(raw)
        versions = {k: raw.copy().filter(lo, hi, method="iir", iir_params=dict(order=4, ftype="butter")).resample(FS).get_data() * 1e6
                    for k, (lo, hi) in BANDS.items()}
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            start = int(round(onset * FS))
            if desc not in ("T0", "T1", "T2") or start + n > versions["mi"].shape[1]:
                continue
            for k in BANDS:
                out[k].append(versions[k][:, start:start + n].astype("float32"))
            out["run"].append(run); out["event"].append(int(desc[1]))
    d = {"mi": np.stack(out["mi"]), "bb": np.stack(out["bb"]), "run": np.array(out["run"]), "event": np.array(out["event"])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **d)
    return d


def prune(d, rest_rng):
    """Keep only the trials the tasks use; draw the 5-class rest trials (one generator over subjects in order)."""
    run, ev = d["run"], d["event"]
    t2 = np.where(np.isin(run, REAL_FIST + IMAG_FIST + REAL_FEET + IMAG_FEET) & (ev == 2))[0]
    t1 = np.where(np.isin(run, IMAG_FIST) & (ev == 1))[0]
    rest_all = np.where(ev == 0)[0]
    rest = rest_rng.choice(rest_all, min(int(round(len(t2) / 4)), len(rest_all)), replace=False)
    keep = np.unique(np.concatenate([t2, t1, rest]))
    out = {k: d[k][keep] for k in ("mi", "bb", "run", "event")}
    out["rest_order"] = np.searchsorted(keep, rest)
    return out


def task_trials(d, task):
    run, ev = d["run"], d["event"]
    idx, lab = [], []
    if task == "5class":
        for runs, label in ((REAL_FIST, 1), (IMAG_FIST, 2), (REAL_FEET, 3), (IMAG_FEET, 4)):
            sel = np.where(np.isin(run, runs) & (ev == 2))[0]; idx += list(sel); lab += [label] * len(sel)
        idx += list(d["rest_order"]); lab += [0] * len(d["rest_order"])
    elif task == "real_vs_imag":
        for runs, label in ((REAL_FIST + REAL_FEET, 0), (IMAG_FIST + IMAG_FEET, 1)):
            sel = np.where(np.isin(run, runs) & (ev == 2))[0]; idx += list(sel); lab += [label] * len(sel)
    else:
        for e, label in ((1, 0), (2, 1)):
            sel = np.where(np.isin(run, IMAG_FIST) & (ev == e))[0]; idx += list(sel); lab += [label] * len(sel)
    return np.array(idx, dtype=int), np.array(lab)


def index(task):
    S, I, y = [], [], []
    for s in sorted(DATA):
        idx, lab = task_trials(DATA[s], task)
        S.append(np.full(len(idx), s)); I.append(idx); y.append(lab)
    return np.concatenate(S), np.concatenate(I), np.concatenate(y)


def gather(key, S, I, length):
    out = np.empty((len(S), N_CH, LENGTHS[length]), dtype="float32")
    for s in np.unique(S):
        m = S == s
        out[m] = DATA[s][key][I[m], :, :LENGTHS[length]]
    return out


def eegnet(n_classes, chans, samples, F1=8, D=2, F2=16, kern=40, dropout=0.5, seed=SEED):
    """EEGNet-8,2 adapted to 80 Hz (temporal kernel 40 samples = 0.5 s)."""
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(chans, samples, 1))
    x = layers.Conv2D(F1, (1, kern), padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=constraints.max_norm(1.0))(x)
    x = layers.Activation("elu")(layers.BatchNormalization()(x))
    x = layers.Dropout(dropout)(layers.AveragePooling2D((1, 4))(x))
    x = layers.SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(x)
    x = layers.Activation("elu")(layers.BatchNormalization()(x))
    x = layers.Flatten()(layers.Dropout(dropout)(layers.AveragePooling2D((1, 8))(x)))
    out = layers.Dense(n_classes, activation="softmax", kernel_constraint=constraints.max_norm(0.25))(x)
    m = keras.Model(inp, out, name="EEGNet")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


class Batches(keras.utils.PyDataset):
    """Feeds EEGNet 64 scaled trials at a time, read directly from the cached trials (no full copy in memory)."""
    def __init__(self, S, I, y, length, mu, sd, shuffle=False, seed=SEED, batch=64):
        super().__init__()
        self.S, self.I, self.y, self.length, self.mu, self.sd, self.batch = S, I, y, length, mu, sd, batch
        self.order, self.shuffle, self.rng = np.arange(len(S)), shuffle, np.random.default_rng(seed)
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.S) / self.batch))

    def __getitem__(self, i):
        b = self.order[i * self.batch:(i + 1) * self.batch]
        x = ((gather("bb", self.S[b], self.I[b], self.length) - self.mu) / self.sd)[..., None]
        return (x, self.y[b]) if self.y is not None else x

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def channel_stats(S, I, length):
    n, s1, s2 = 0, 0.0, 0.0
    for p in np.unique(S):
        x = DATA[p]["bb"][I[S == p], :, :LENGTHS[length]].astype("float64")
        s1 = s1 + x.sum(axis=(0, 2)); s2 = s2 + (x ** 2).sum(axis=(0, 2)); n += x.shape[0] * x.shape[2]
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
    return mu.astype("float32")[None, :, None], sd.astype("float32")[None, :, None]


def scores(y_true, pred, n_classes):
    return {"accuracy": float(accuracy_score(y_true, pred)), "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "confusion": confusion_matrix(y_true, pred, labels=range(n_classes)).tolist(), "n_test": int(len(y_true))}


def fit_csp(S, I, y, tr, te, length):
    clf = make_pipeline(CSP(n_components=8, reg="ledoit_wolf", log=True), LinearDiscriminantAnalysis())
    clf.fit(gather("mi", S[tr], I[tr], length), y[tr])
    pred = clf.predict(gather("mi", S[te], I[te], length)); gc.collect()
    return pred


def fit_eegnet(S, I, y, fit, va, te, length, n_classes, seed):
    mu, sd = channel_stats(S[fit], I[fit], length)
    m = eegnet(n_classes, N_CH, LENGTHS[length], seed=seed)
    m.fit(Batches(S[fit], I[fit], y[fit], length, mu, sd, shuffle=True, seed=seed),
          validation_data=Batches(S[va], I[va], y[va], length, mu, sd), epochs=100, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    pred = m.predict(Batches(S[te], I[te], None, length, mu, sd), verbose=0).argmax(1)
    keras.backend.clear_session(); gc.collect()
    return pred


if __name__ == "__main__":
    a = args(__doc__.splitlines()[0])
    mne.set_log_level("ERROR")
    R = f"{a.out}/reference"; os.makedirs(R, exist_ok=True)
    t0 = time.time()
    rest_rng = np.random.default_rng(SEED)
    for s in SUBJECTS:
        DATA[s] = prune(subject_trials(a.data, s), rest_rng)
        if s % 10 == 0:
            print(f"S{s:03d} loaded | {time.time() - t0:5.0f} s", flush=True)
    N_CH = next(iter(DATA.values()))["mi"].shape[1]
    people = np.random.default_rng(SEED).permutation(sorted(DATA))
    FOLDS = cached(f"{a.splits}/reference_folds.json", lambda: [f.tolist() for f in np.array_split(people, 5)])

    RES = {}
    for task, n_cls in TASKS.items():
        S, I, y = index(task)
        for length in LENGTHS:
            runs = [("CSP + LDA", None)] + [("EEGNet", s) for s in (SEEDS_4S if length == "4s" else [42])]
            for model, seed in runs:
                for k, test_people in enumerate(FOLDS):
                    te = np.isin(S, test_people); tr = ~te
                    if model == "CSP + LDA":
                        r = cached(f"{R}/cross_{task}_{length}_csp_fold{k}.json",
                                   lambda: scores(y[te], fit_csp(S, I, y, tr, te, length), n_cls))
                    else:
                        def run():
                            train_people = np.unique(S[tr])
                            val = np.random.default_rng(seed + k).choice(train_people, max(1, len(train_people) // 10), replace=False)
                            va = np.isin(S, val); fit = tr & ~va
                            return scores(y[te], fit_eegnet(S, I, y, fit, va, te, length, n_cls, seed), n_cls)
                        r = cached(f"{R}/cross_{task}_{length}_eegnet_seed{seed}_fold{k}.json", run)
                    RES.setdefault((task, length, model, seed), []).append(r)
                    check_ram()
                acc = [r["accuracy"] for r in RES[(task, length, model, seed)]]
                print(f"{task:<14} {length} {model:<10} seed {seed} accuracy {np.mean(acc):.3f} +/- {np.std(acc):.3f} "
                      f"| {time.time() - t0:5.0f} s", flush=True)

    summary = {f"{t}/{l}/{m}" + (f"/seed{s}" if s else ""): {
        "accuracy_pooled": sum(np.trace(r["confusion"]) for r in v) / sum(r["n_test"] for r in v),
        "accuracy_fold_mean": float(np.mean([r["accuracy"] for r in v]))} for (t, l, m, s), v in RES.items()}
    for task in TASKS:
        accs = [summary[f"{task}/4s/EEGNet/seed{s}"]["accuracy_pooled"] for s in SEEDS_4S]
        summary[f"{task}/4s/EEGNet/seed_mean"] = {"mean": float(np.mean(accs)), "min": float(min(accs)), "max": float(max(accs))}
        print(f"{task}: EEGNet 4 s over seeds {SEEDS_4S}: mean {np.mean(accs):.3f} (range {min(accs):.3f}-{max(accs):.3f})")
    json.dump(summary, open(f"{a.out}/reference_summary.json", "w"), indent=2, default=float)
