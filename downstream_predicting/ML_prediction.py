#!/usr/bin/env python3#!/usr/bin/env python3
"""
AF embedding experiment (CV + GridSearch) — single-file, GitHub-ready.

What it does
------------
- Loads a de-identified baseline CSV and:
  * an optional Supervised embedding CSV 
  * optional one or more Unsupervised embedding CSVs 
- Merges each embedding with baseline (index-aligned merge, no PHI).
- Runs nested CV: inner GridSearchCV (refit=roc_auc) inside an outer StratifiedKFold.
- Writes per-dataset, per-model metrics (AUC mean/std, F1 mean/std) to a results CSV.

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# ------------------------- Logging & I/O helpers ------------------------- #

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def init_results_csv(path: str | Path) -> None:
    """
    Initialize results CSV with header if it does not already exist.
    """
    path = Path(path)
    ensure_parent(path)
    header = ["dataset", "model", "auc_mean", "auc_std", "f1_mean", "f1_std"]
    if not path.exists():
        path.write_text(",".join(header) + "\n")


def append_result_csv(path: str | Path, dataset: str, model: str,
                      auc_mean: float, auc_std: float,
                      f1_mean: float, f1_std: float) -> None:
    """
    Append a single result row, formatting to 3 decimals as requested.
    """
    with Path(path).open("a", newline="") as f:
        f.write(f"{dataset},{model},{auc_mean:.3f},{auc_std:.3f},{f1_mean:.3f},{f1_std:.3f}\n")


# ------------------------------- Data ----------------------------------- #

ECHO_DROP = [
    "LV EF EUH2", "IVS d, 2D", "LVPW d, 2D", "LAVI", "LA diameter",
    "VentricularRate", "Smoker"
]


def load_baseline(baseline_file: str) -> pd.DataFrame:
    """
    Load & coerce baseline. Assumes a 'target' column exists.
    Drops some echo-related columns to match earlier behavior.
    Keeps EMPI_NBR and drops the first 10 columns after that if shape allows,
    mirroring prior slicing (EMPI_NBR + cols from index >= 11).
    """
    logging.info("Loading baseline: %s", baseline_file)
    dat = pd.read_csv(baseline_file)

    # Optional: drop a site-specific column if present
    if "SHF" in dat.columns:
        dat = dat.drop(columns=["SHF"])

    # Mirror original logic: keep EMPI_NBR + columns from col 11 onward (0-based)
    if "EMPI_NBR" in dat.columns and dat.shape[1] >= 12:
        dat = pd.concat([dat[["EMPI_NBR"]], dat.iloc[:, 11:]], axis=1)

    if "target" not in dat.columns:
        raise ValueError("Baseline file must contain a 'target' column.")
    dat["target"] = pd.to_numeric(dat["target"], errors="coerce").astype(int)

    # Coerce known numeric columns if present
    int_like = [
        "gender_Male", "age", "Obesity", "HTN", "diabetes", "CHF", "AEN", "CAM", "DYS", "CAD",
        "PVD_vascular", "CKD", "AVD", "MVD", "TVD", "PVD_valve", "MI", "AP", "COPD",
        "hyperthyroidism", "PRInterval", "QRSDuration", "QTInterval", "PAxis", "RAxis",
        "TAxis", "PVC", "PAC", "R-block", "L-block", "LA enlargement", "AV block",
        "left axis deviation", "ischemia", "infarct", "LVH", "Tach", "ST&T", "Brady"
    ]
    float_like = [
        "Height", "Weight", "Total_cholesterol", "Non_HDL_cholesterol", "White_blood_count",
        "Red_blood_count", "Hemoglobin", "Creatinine", "Glucose", "Triglyceride",
        "Platelet_count", "QTCorrected"
    ]
    for c in int_like + float_like:
        if c in dat.columns:
            dat[c] = pd.to_numeric(dat[c], errors="coerce")

    # Drop echo columns if present
    dat = dat.drop(columns=[c for c in ECHO_DROP if c in dat.columns], errors="ignore")

    return dat


def merge_index_aligned(baseline: pd.DataFrame, emb_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Merge baseline with an embedding DF by row index (safe, explicit).
    Adds a temporary 'row_id' to both; drops it afterward.
    """
    b = baseline.reset_index(drop=False).rename(columns={"index": "row_id"})
    e = emb_df.reset_index(drop=False).rename(columns={"index": "row_id"})
    merged = b.merge(e, on="row_id", how="inner")
    if "row_id" in merged.columns:
        merged = merged.drop(columns=["row_id"])
    logging.info("Merged %s | shape=%s", label, merged.shape)
    return merged


# --------------------------- Pipelines & Grids -------------------------- #

def make_pipelines_and_grids(seed: int = 42):
    # Pipelines
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("classifier", LogisticRegression(max_iter=100, random_state=seed))
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("classifier", RandomForestClassifier(random_state=seed))
    ])
    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("classifier", GradientBoostingClassifier(random_state=seed))
    ])

    # Param grids
    # Note: lbfgs does not support L1; split into two valid sub-grids.
    lr_grid = [
        {"classifier__solver": ["lbfgs"],
         "classifier__penalty": ["l2"],
         "classifier__class_weight": [None, "balanced"],
         "classifier__C": [0.01, 0.1, 1, 10, 100]},
        {"classifier__solver": ["liblinear"],
         "classifier__penalty": ["l1", "l2"],
         "classifier__class_weight": [None, "balanced"],
         "classifier__C": [0.01, 0.1, 1, 10, 100]},
    ]
    rf_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__class_weight": [None, "balanced"],
    }
    gb_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.001, 0.01, 0.1],
        "classifier__max_depth": [3, 5, 7],
    }

    return (lr, rf, gb), (lr_grid, rf_grid, gb_grid)


# ------------------------------- Main CV -------------------------------- #

def run_experiments(
    baseline_csv: str,
    supervised_csv: str | None,
    unsupervised_csvs: list[str] | None,
    results_csv: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 42,
    unsup_name: str = "Unsupervised",
    sup_name: str = "Supervised",
) -> None:
    """
    Build datasets, run nested CV (GridSearchCV inside outer CV), and write results.
    """
    # init results file
    init_results_csv(results_csv)

    # load baseline
    baseline_df = load_baseline(baseline_csv)

    # Optional: mirror earlier mid-column removal to avoid duplicates
    if baseline_df.shape[1] >= 26:
        baseline_df = pd.concat([baseline_df.iloc[:, :8], baseline_df.iloc[:, 25:]], axis=1)
    logging.info("Baseline columns #: %d", baseline_df.shape[1])

    # collect datasets with friendly labels
    datasets: dict[str, pd.DataFrame] = {}

    if supervised_csv and Path(supervised_csv).exists():
        emb = pd.read_csv(supervised_csv)
        label = sup_name  # "Supervised"
        datasets[label] = merge_index_aligned(baseline_df, emb, label)

    if unsupervised_csvs:
        for i, path in enumerate(unsupervised_csvs, start=1):
            if path and Path(path).exists():
                emb = pd.read_csv(path)
                # If multiple unsupervised sources are provided, suffix with index
                label = unsup_name if len(unsupervised_csvs) == 1 else f"{unsup_name} ({i})"
                datasets[label] = merge_index_aligned(baseline_df, emb, label)

    if not datasets:
        raise SystemExit("No embedding CSVs were found. Provide --supervised and/or at least one --unsupervised file.")

    # CV setups
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    scoring = {"roc_auc": "roc_auc"}  # for GridSearchCV refit
    outer_scoring = {"AUC": "roc_auc", "F1": "f1"}  # for cross_validate reports

    (lr_pipe, rf_pipe, gb_pipe), (lr_grid, rf_grid, gb_grid) = make_pipelines_and_grids(seed=seed)

    # Estimators for nested CV (GridSearchCV inside cross_validate)
    lr_gs = GridSearchCV(lr_pipe, lr_grid, scoring=scoring, refit="roc_auc", cv=inner_cv, n_jobs=-1)
    rf_gs = GridSearchCV(rf_pipe, rf_grid, scoring=scoring, refit="roc_auc", cv=inner_cv, n_jobs=-1)
    gb_gs = GridSearchCV(gb_pipe, gb_grid, scoring=scoring, refit="roc_auc", cv=inner_cv, n_jobs=-1)

    for ds_name, df in datasets.items():
        logging.info("Dataset: %s | shape=%s", ds_name, df.shape)
        if "target" not in df.columns:
            raise ValueError(f"Dataset '{ds_name}' is missing the required 'target' column.")
        # EMPI_NBR is optional; drop if present
        X = df.drop(columns=[c for c in ["target", "EMPI_NBR"] if c in df.columns])
        y = df["target"]

        # Logistic Regression
        lr_scores = cross_validate(
            lr_gs, X, y, scoring=outer_scoring, cv=outer_cv,
            return_train_score=False, n_jobs=-1
        )
        append_result_csv(
            results_csv, ds_name, "LR",
            np.mean(lr_scores["test_AUC"]), np.std(lr_scores["test_AUC"]),
            np.mean(lr_scores["test_F1"]),  np.std(lr_scores["test_F1"])
        )

        # Random Forest
        rf_scores = cross_validate(
            rf_gs, X, y, scoring=outer_scoring, cv=outer_cv,
            return_train_score=False, n_jobs=-1
        )
        append_result_csv(
            results_csv, ds_name, "RF",
            np.mean(rf_scores["test_AUC"]), np.std(rf_scores["test_AUC"]),
            np.mean(rf_scores["test_F1"]),  np.std(rf_scores["test_F1"])
        )

        # Gradient Boosting
        gb_scores = cross_validate(
            gb_gs, X, y, scoring=outer_scoring, cv=outer_cv,
            return_train_score=False, n_jobs=-1
        )
        append_result_csv(
            results_csv, ds_name, "GB",
            np.mean(gb_scores["test_AUC"]), np.std(gb_scores["test_AUC"]),
            np.mean(gb_scores["test_F1"]),  np.std(gb_scores["test_F1"])
        )

    logging.info("Done. Results saved to %s", results_csv)


# --------------------------------- CLI ---------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--baseline", required=True, help="Path to baseline CSV (de-identified).")
    p.add_argument("--supervised", help="Path to a Supervised embedding CSV.")
    p.add_argument("--unsupervised", nargs="+", help="Path(s) to one or more Unsupervised embedding CSVs.")
    p.add_argument("--results", required=True, help="Path to output results CSV.")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unsup-name", default="Unsupervised",
                   help="Base label for unsupervised rows (e.g., 'Unsupervised pretraining').")
    p.add_argument("--sup-name", default="Supervised",
                   help="Label for the supervised row.")
    return p


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    run_experiments(
        baseline_csv=args.baseline,
        supervised_csv=args.supervised,
        unsupervised_csvs=args.unsupervised,
        results_csv=args.results,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
        unsup_name=args.unsup_name,
        sup_name=args.sup_name,
    )


if __name__ == "__main__":
    main()
