# Oversampling before cross-validation inflates EEG motor-intent decoding: a reproduction of TransNN-MHA

Code for the paper

> M. Elsayed, S. Mustafa and S. Mohsen, "Oversampling Before Cross-Validation Inflates EEG Motor-Intent Decoding:
> A Reproduction of TransNN-MHA," IEEE-ICCA 2026 (under review).

It reproduces the 92.39% five-class accuracy reported for TransNN-MHA (Sultan et al., *IEEE Access*, 2025,
doi:10.1109/ACCESS.2025.3567908) with a re-implementation of the authors' released pipeline, removes its leaks one at a
time, and compares the leak-free result with two established decoders (CSP + LDA, EEGNet) on unseen subjects.

## The scripts

Run them in order. Each one caches its results and can be restarted where it stopped.

| Script | What it does | Time (Colab T4) |
|---|---|---|
| `01_features.py` | Downloads PhysioNet EEGMMIDB; the authors' whole-run features (ICA cleaning, 258 statistics per run) | ~1 h |
| `02_protocols.py` | P0 (published code) → P1 → P2 → P3, TransNN-MHA, KNN and logistic regression; freezes folds and feature selections in `splits/` | 1.5–2 h |
| `03_diagnostics.py` | Pooled confusion matrices, per-class metrics; fold-0 learning, ROC and PR curves | ~5 min |
| `04_reference_decoders.py` | CSP + LDA and EEGNet, 5 subject-wise folds, three tasks, 1-s and 4-s trials, EEGNet seeds 42/7/123 | 3–4 h |
| `05_statistics.py` | Chance from confusion matrices, permutation and fold-level tests, Holm correction, paired tests | minutes, CPU |
| `06_figures.py` | Figs. 1–5 of the paper from the saved results | seconds |
| `07_augmentation_ablation.py` | P2 and P3 with other oversampling choices and real-signal crop augmentation | ~10 h (crop features), then ~1 h |

```bash
pip install -r requirements.txt
python 01_features.py --data data --out results
python 02_protocols.py --data data --out results
python 03_diagnostics.py --data data --out results
python 04_reference_decoders.py --data data --out results
python 05_statistics.py --out results
python 06_figures.py --out results
python 07_augmentation_ablation.py --data data --out results   # optional, Sec. IV-B check
```

On Google Colab, open `run_on_colab.ipynb`: it mounts Google Drive and keeps data and results there, so a disconnect
loses nothing.

## Where each number in the paper comes from

| Paper | Value | Script | Output |
|---|---|---|---|
| Sec. III-B: recordings, synthetic share | 1,526 real; 4,905 after SMOTE (68.9% synthetic) | 01, 02 | console; `protocols_summary.json` → `counts` |
| Abstract, Sec. IV-A: published code | 90.3% ± 1.5% (paper: 92.39% ± 1.34%) | 02 | `protocols_summary.json` → `P0/TransNN-MHA` |
| Fig. 2, Sec. IV-B: one leak at a time | P1 86.8%, P2 27.5% (macro-F1 0.259), P3 29.2% | 02 | `protocols_summary.json` |
| Sec. IV-B: baselines | KNN 75.9% → 27.7%, logistic regression 39.7% → 26.0% | 02 | `protocols_summary.json` |
| Sec. IV-B: oversampling choice | 26.5–29.2% with no oversampling, class weights, balancing-only SMOTE or crops (all paired p > 0.24) | 07 | `augmentation_summary.json`, `augmentation_paired.csv` |
| Fig. 3, Sec. IV-B: confusion matrices, per class | 981 test samples per class in P0; P2 F1 0.19–0.30 (movement), 0.45 (rest) | 03 | `confusion_pooled.json`, `per_class.csv` |
| Fig. 4, Sec. IV-B: curves, AUCs | ROC AUC 0.973 → 0.639, PR AUC 0.930 → 0.313; per-class ROC AUC | 03 | `curves_fold0.json` |
| Fig. 5, Sec. IV-C: reference decoders | EEGNet 43.4% (seeds 43.0–44.3%), 39.6% at 1 s; CSP + LDA 22.0% | 04, 05 | `reference_summary.json`, `stats_vs_chance.csv` |
| Sec. IV-C: EEGNet vs CSP + LDA | +22.3 points (95% CI 20.7–23.8), Holm p < 0.0001 | 05 | `stats_paired.csv` |
| Sec. IV-C: above chance | all leak-free results, permutation p ≤ 0.0028 (Holm) | 05 | `stats_vs_chance.csv` |
| Sec. IV-C: what EEGNet's accuracy consists of | rest 56%, movement as movement 90%, limb 74%, real vs imagined 62% | 05 | `eegnet_breakdown.json` |
| Sec. V-B, Table IV: implementation check | EEGNet imagined left vs right fist 77.4% (three-seed mean) | 04 | `reference_summary.json` → `left_vs_right/4s/EEGNet/seed_mean` |
| Sec. V-D: training variability | EEGNet seed spread 0.7–3.5 points | 04 | `reference_summary.json` → `*/seed_mean` |

## Reproducibility notes

- **Splits are frozen.** `02_protocols.py` and `04_reference_decoders.py` write the folds and the selected features to
  `splits/` on their first run (seed 42) and read them back afterwards. The files in this repository are the ones
  behind the paper; delete them only if you want new random splits.
- **GPU training is not bit-for-bit repeatable.** Expect fold accuracies to move by a few points between runs (we saw
  87.5% and 89.7% for the same P0 fold); means over folds move much less. The three EEGNet seeds measure this variation.
- **Library versions.** The authors did not pin their versions. The results in the paper were produced on Google
  Colab with the versions in `requirements.txt`; SMOTE's synthetic samples depend on the imbalanced-learn version.
- **Holm-corrected p-values** depend on which tests form the family. `05_statistics.py` corrects over the results in
  this paper; the conclusions do not depend on this choice.
- **Data.** PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0 (Schalk et al., 2004; Goldberger et al., 2000),
  https://physionet.org/content/eegmmidb/1.0.0/, downloaded automatically through MNE-Python. Three subjects recorded
  at 128 Hz (88, 92, 100) are excluded from the reference-decoder analysis; the authors' pipeline keeps them, as their code does.

## Relation to the authors' code

The TransNN-MHA pipeline (feature extraction, SMOTE, feature selection, model and training settings) is re-implemented
here from the authors' public repository, https://github.com/madibabaiasl/EEGIntentPaper, which is licensed for
non-commercial use with citation. No file of that repository is included here. Please cite their paper when you use
this code.

## Citation

See `CITATION.cff`. If you use this code, please cite the paper above and Sultan et al. (2025).

## License

MIT (see `LICENSE`).
