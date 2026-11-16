# Frozen splits

The scripts write these files on their first run and read them back afterwards:

- `folds.json`: the 6 folds of P0/P1 (over the SMOTE-augmented samples), P2 (random) and P3 (subject-wise)
- `selection_alldata.json`: the 70 features the authors' method selects on all data (P0, P1)
- `P2_selection_fold{0-5}.json`, `P3_selection_fold{0-5}.json`: the 70 features selected inside each training fold
- `reference_folds.json`: the test subjects of the 5 subject-wise folds of the reference decoders

They are the splits behind the paper. Delete them only to draw new random splits.
