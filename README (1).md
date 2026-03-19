# construction-classifier

A decision-tree classifier that predicts **verb construction categories** (e.g. causative, non-causative, anticausative) from syntactic and POS features extracted per verb token.

---

## What it does

| Step | Output |
|------|--------|
| Loads an Excel workbook | reads one sheet of verb-token annotations |
| Encodes syntactic features | label-encodes categorical columns, fills missing values |
| Trains a decision tree | `sklearn` `DecisionTreeClassifier` with balanced class weights |
| Evaluates on held-out test split | accuracy + full classification report |
| Exports a tree diagram | `tree_diagram.png` — visual of the learned rules |
| Saves predictions | a copy of the input workbook with a `predicted_construction` column appended |

---

## Input format

The script expects an **Excel workbook** (`.xlsx`) with a sheet named `childes_labile` that contains at minimum:

| Column | Description |
|--------|-------------|
| `labile` | **Target** — construction category label (e.g. `causative`, `non_causative`, `anticausative`) |
| `obj` | Object type (`noun`, `pro`, …) |
| `subj` | Subject type (`pron`, `nom`, …) |
| `passive` | Passive auxiliary (`werden`, `sein`, `other`) |
| `sv_order` | Subject–verb order (`sv`, `vs`) |
| `oblique` | Oblique argument type |
| `refl` | Reflexive marker flag |
| `pos_cat` | Broad POS category (`V`, `N`, `J`, `R`) |
| `cp` | Complementiser-phrase flag |
| `iobj` | Indirect object flag |
| `modpp_prep` | Modifier PP / preposition flag |
| `right_N` | Right-adjacent noun flag |

Additional columns are preserved but not used as features.

---

## Usage

```bash
# basic
python construction_classifier.py --input data.xlsx

# with custom options
python construction_classifier.py \
    --input  data.xlsx \
    --output results.xlsx \
    --max_depth 5 \
    --test_size 0.2 \
    --seed 42 \
    --tree_png my_tree.png
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Path to the input `.xlsx` workbook |
| `--output` | `<input>_predictions.xlsx` | Path for the output workbook |
| `--max_depth` | `6` | Maximum tree depth (`0` = unlimited) |
| `--test_size` | `0.2` | Fraction of data held out for evaluation |
| `--seed` | `42` | Random seed for reproducibility |
| `--tree_png` | `tree_diagram.png` | Output path for the tree diagram |

---

## Installation

```bash
pip install pandas openpyxl scikit-learn matplotlib
```

Python 3.10+ recommended.

---

## Output example

```
Target distribution (labile):
causative        24673
non_causative    16886
anticausative       11
…

Split  →  train: 33,256  |  test: 8,315

Accuracy : 0.9134  (91.34%)

Classification report:
              precision  recall  f1-score  support
  anticausative   0.61    0.55     0.58       9
  causative       0.92    0.95     0.93    4935
  non_causative   0.90    0.87     0.88    3371
  …

Feature importances (top 10):
obj          0.4821
subj         0.2103
passive      0.1187
…

Tree diagram saved → tree_diagram.png
Predictions saved  → data_predictions.xlsx
```

---

## Extending the classifier

- **Swap in a Random Forest**: replace `DecisionTreeClassifier` with `RandomForestClassifier` (same API).
- **Add features**: append column names to `SYNTACTIC_FEATURES` at the top of the script.
- **Change the target**: update `TARGET_COL` to any categorical column in your sheet.
- **Cross-validation**: replace the single train/test split with `sklearn.model_selection.cross_val_score`.

---

## License

MIT
