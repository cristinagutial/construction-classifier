"""
construction_classifier.py
───────────────────────────────────────────────────────────────────────────────
Decision-tree classifier for verb construction categories.

Input  : an Excel workbook (.xlsx) with a sheet named 'childes_labile'
         containing syntactic/POS features per verb token.
Output : • console — classification report + accuracy
         • tree_diagram.png — visualised decision tree
         • predictions appended to a copy of the input workbook

Target variable   : `labile`   (causative | non_causative | anticausative | …)
Feature set       : syntactic / POS columns (obj, subj, passive, sv_order,
                    oblique, refl, pos_cat, cp, iobj, modpp_prep, right_N)

Usage
─────
    python construction_classifier.py --input data.xlsx
    python construction_classifier.py --input data.xlsx --output results.xlsx
                                      --max_depth 5 --test_size 0.2

Requirements
────────────
    pip install pandas openpyxl scikit-learn matplotlib
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────

SHEET_NAME = "childes_labile"
TARGET_COL = "labile"

SYNTACTIC_FEATURES = [
    "obj",          # object type (noun, pro, …)
    "subj",         # subject type (pron, nom, …)
    "passive",      # passive auxiliary (werden, sein, other)
    "sv_order",     # subject–verb order (sv, vs)
    "oblique",      # oblique argument presence / type
    "refl",         # reflexive marker (numeric flag)
    "pos_cat",      # broad POS category (V, N, J, R)
    "cp",           # complementiser-phrase flag
    "iobj",         # indirect object flag
    "modpp_prep",   # modifier PP / preposition flag
    "right_N",      # right-adjacent noun flag
]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    return df


def encode_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Label-encode all string/categorical features; fill NaNs with sentinels."""
    out = df[feature_cols].copy()
    for col in out.columns:
        # treat object, string, and StringDtype as categorical
        is_str = (
            out[col].dtype == object
            or pd.api.types.is_string_dtype(out[col])
        )
        if is_str:
            out[col] = out[col].fillna("_missing_").astype(str)
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col])
        else:
            out[col] = out[col].fillna(-1)
    return out


def build_model(
    X_train, y_train,
    max_depth: int | None,
    random_state: int,
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def save_tree_diagram(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    out_path: str = "tree_diagram.png",
    max_depth_display: int = 4,
) -> None:
    fig_width = max(20, len(feature_names) * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth_display,
        ax=ax,
        fontsize=9,
    )
    ax.set_title("Decision Tree — Verb Construction Categories", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tree diagram saved → {out_path}")


def save_predictions(
    source_path: str,
    df_original: pd.DataFrame,
    predictions: pd.Series,
    out_path: str,
) -> None:
    df_out = df_original.copy()
    df_out["predicted_construction"] = predictions.values
    df_out.to_excel(out_path, sheet_name=SHEET_NAME, index=False)
    print(f"  Predictions saved  → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decision-tree classifier for verb construction categories."
    )
    p.add_argument(
        "--input", required=True,
        help="Path to the input .xlsx workbook.",
    )
    p.add_argument(
        "--output", default=None,
        help="Path for the output .xlsx (default: <input>_predictions.xlsx).",
    )
    p.add_argument(
        "--max_depth", type=int, default=6,
        help="Maximum tree depth (default: 6). Use 0 for unlimited.",
    )
    p.add_argument(
        "--test_size", type=float, default=0.2,
        help="Proportion of data for the test split (default: 0.2).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--tree_png", default="tree_diagram.png",
        help="Output path for the tree diagram PNG (default: tree_diagram.png).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_depth = args.max_depth if args.max_depth > 0 else None

    # ── load ──────────────────────────────────────────────────────────────────
    df = load_data(args.input)

    # ── validate columns ──────────────────────────────────────────────────────
    missing_cols = [c for c in SYNTACTIC_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: Required columns not found in sheet: {missing_cols}")

    # keep only rows where the target label is present
    df_model = df[df[TARGET_COL].notna()].copy()
    print(f"\nTarget distribution ({TARGET_COL}):")
    print(df_model[TARGET_COL].value_counts().to_string())

    # stratified split requires ≥2 members per class; warn and drop singletons
    class_counts = df_model[TARGET_COL].value_counts()
    singletons = class_counts[class_counts < 2].index.tolist()
    if singletons:
        print(f"\n  NOTE: Dropping {len(singletons)} singleton class(es) from split: {singletons}")
        df_model = df_model[~df_model[TARGET_COL].isin(singletons)]

    # ── features & target ─────────────────────────────────────────────────────
    available_features = [c for c in SYNTACTIC_FEATURES if c in df_model.columns]
    X = encode_features(df_model, available_features)
    y = df_model[TARGET_COL]

    # ── train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )
    print(f"\nSplit  →  train: {len(X_train):,}  |  test: {len(X_test):,}")

    # ── train ─────────────────────────────────────────────────────────────────
    print("\nTraining decision tree …")
    clf = build_model(X_train, y_train, max_depth=max_depth, random_state=args.seed)

    # ── evaluate ──────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # feature importances
    importances = pd.Series(clf.feature_importances_, index=available_features)
    print("Feature importances (top 10):")
    print(importances.sort_values(ascending=False).head(10).to_string())

    # ── tree diagram ──────────────────────────────────────────────────────────
    print("\nGenerating tree diagram …")
    save_tree_diagram(
        clf,
        feature_names=available_features,
        class_names=sorted(y.unique().tolist()),
        out_path=args.tree_png,
    )

    # ── predict on full dataset & save ───────────────────────────────────────
    print("\nGenerating predictions for full dataset …")
    X_full = encode_features(df_model, available_features)
    full_preds = pd.Series(clf.predict(X_full), index=df_model.index)

    out_path = args.output or str(
        Path(args.input).stem + "_predictions.xlsx"
    )
    save_predictions(args.input, df_model, full_preds, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
