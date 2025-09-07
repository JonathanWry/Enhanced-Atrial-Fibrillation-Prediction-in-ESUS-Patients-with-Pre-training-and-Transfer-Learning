#!/usr/bin/env python3
"""
Cohort Experiment (single-file, GitHub-ready)

- Trains LR / RF / GB on a main cohort with optional Supervised and/or Unsupervised embeddings.
- Saves the best models (full pipelines with any preprocessing).
- Runs external validation on an external cohort and saves ROC figures.
- No PHI; all paths provided via CLI arguments.

Usage (examples)

# Train on the main cohort (one supervised embedding)
python External_validation.py \
  --train-main \
  --main-baseline data/main/baseline.csv \
  --embed-supervised data/main/supervised.csv \
  --models-dir outputs/main/models \
  --figs-dir outputs/main/figs

# Train on the main cohort (two unsupervised embeddings)
python External_validation.py \
  --train-main \
  --main-baseline data/main/baseline.csv \
  --embed-unsupervised data/main/unsup_a.csv data/main/unsup_b.csv \
  --models-dir outputs/main/models \
  --figs-dir outputs/main/figs

# Predict on an external cohort using previously saved models
python External_validation.py \
  --predict-external \
  --external-baseline data/external/baseline.csv \
  --external-embed-unsupervised data/external/unsup_a.npy data/external/unsup_b.npy \
  --models-dir outputs/main/models \
  --figs-dir outputs/external/figs

You can also do both in one run by passing both --train-main and --predict-external.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle
import warnings

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ----------------------------- Utilities ----------------------------- #

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def ensure_dirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")


# ----------------------------- Data Loading/Cleaning ----------------------------- #

ECHO_DROP = [
    "LV EF EUH2", "IVS d, 2D", "LVPW d, 2D", "LAVI",
    "LA diameter", "VentricularRate", "Smoker"
]

def load_baseline(baseline_file: str, drop_specific: bool = True) -> pd.DataFrame:
    """
    Load and lightly clean a baseline dataframe; tolerant to missing columns.
    """
    logging.info("Loading baseline: %s", baseline_file)
    dat = pd.read_csv(baseline_file)

    # Optional: drop known site-specific column if present
    if drop_specific and "SHF" in dat.columns:
        dat.drop(columns=["SHF"], inplace=True, errors="ignore")

    # Mirror prior behavior: keep EMPI_NBR and drop early columns if present
    if drop_specific and "EMPI_NBR" in dat.columns and dat.shape[1] >= 12:
        dat = pd.concat([dat[["EMPI_NBR"]], dat.iloc[:, 11:]], axis=1)

    if "target" in dat.columns:
        dat["target"] = pd.to_numeric(dat["target"], errors="coerce").astype(int)
    else:
        raise ValueError("Baseline file must contain a 'target' column.")

    # Coerce common numeric columns if present
    int_cols = [
        "gender_Male", "age", "Obesity", "HTN", "diabetes", "CHF", "AEN", "CAM",
        "DYS", "CAD", "PVD_vascular", "CKD", "AVD", "MVD", "TVD", "PVD_valve", "MI",
        "AP", "COPD", "hyperthyroidism", "PRInterval", "QRSDuration", "QTInterval",
        "PAxis", "RAxis", "TAxis", "PVC", "PAC", "R-block", "L-block", "LA enlargement",
        "AV block", "left axis deviation", "ischemia", "infarct", "LVH", "Tach", "ST&T", "Brady"
    ]
    float_cols = [
        "Height", "Weight", "Total_cholesterol", "Non_HDL_cholesterol",
        "White_blood_count", "Red_blood_count", "Hemoglobin", "Creatinine",
        "Glucose", "Triglyceride", "Platelet_count", "QTCorrected"
    ]
    for c in int_cols + float_cols:
        if c in dat.columns:
            dat[c] = pd.to_numeric(dat[c], errors="coerce")

    # Drop echo-related cols if present
    dat.drop(columns=[c for c in ECHO_DROP if c in dat.columns], inplace=True, errors="ignore")

    return dat


def merge_on_empinbr(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Merge on EMPI_NBR, dropping 'target' from right if present."""
    drop_cols = [c for c in ["target"] if c in other.columns]
    return base.merge(other.drop(columns=drop_cols, errors="ignore"), on="EMPI_NBR", how="inner")


def concat_by_row(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Row-wise concat (aligns on index)."""
    return pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def df_from_npy(npy_path: str, n_cols: int | None = None) -> pd.DataFrame:
    arr = np.load(npy_path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if n_cols is None:
        n_cols = arr.shape[1]
    return pd.DataFrame(arr, columns=[f"v{i}" for i in range(1, n_cols + 1)])


def load_embedding_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".npy"):
        return df_from_npy(path)
    return pd.read_csv(path)


def merge_flex(base: pd.DataFrame, emb: pd.DataFrame) -> pd.DataFrame:
    """
    Merge baseline with embedding:
    - If 'EMPI_NBR' exists in embedding, merge on EMPI_NBR.
    - Otherwise, align by row order (concat).
    """
    if "EMPI_NBR" in emb.columns and "EMPI_NBR" in base.columns:
        return merge_on_empinbr(base, emb)
    return concat_by_row(base, emb)


# ----------------------------- Pipelines/Grids ----------------------------- #

def make_pipelines():
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("oversample", RandomOverSampler(random_state=42)),
        ("classifier", LogisticRegression(max_iter=500))
    ])

    rf_pipeline = Pipeline([
        ("oversample", RandomOverSampler(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=100))
    ])

    gb_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("oversample", RandomOverSampler(random_state=42)),
        ("classifier", GradientBoostingClassifier(random_state=100))
    ])

    return lr_pipeline, rf_pipeline, gb_pipeline


def param_grids():
    lr_param_grid = [
        {  # lbfgs supports L2 only
            "classifier__solver": ["lbfgs"],
            "classifier__penalty": ["l2"],
            "classifier__class_weight": ["balanced"],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
        },
        {  # liblinear supports L1 and L2
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l1", "l2"],
            "classifier__class_weight": ["balanced"],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
        }
    ]

    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 200, 500],
        "classifier__max_depth": [None, 10, 20, 50],
        "classifier__class_weight": ["balanced"]
    }

    gb_param_grid = {
        "classifier__n_estimators": [50, 100, 200, 500],
        "classifier__learning_rate": [0.001, 0.01, 0.1],
        "classifier__max_depth": [3, 5, 7]
    }

    return lr_param_grid, rf_param_grid, gb_param_grid


# ----------------------------- Training ----------------------------- #

def fit_best_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_dir: str,
    seed: int = 100
) -> Dict[str, str]:
    """
    Runs GridSearchCV for LR/RF/GB and saves the best pipelines to disk.
    Returns dict of model_name -> path.
    NOTE: This saves ONE set of models (LR/RF/GB) into models_dir.
    If you train multiple dataset variants, run this script separately for each
    to avoid overwriting (or change the models_dir per run).
    """
    ensure_dirs(models_dir)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {"roc_auc": "roc_auc"}

    lr, rf, gb = make_pipelines()
    lr_grid, rf_grid, gb_grid = param_grids()

    def grid_fit_save(name: str, pipeline: Pipeline, grid) -> str:
        gs = GridSearchCV(pipeline, grid, scoring=scoring, refit="roc_auc", cv=inner_cv, n_jobs=-1)
        gs.fit(X, y)
        logging.info("Best %s params: %s", name, gs.best_params_)
        best = gs.best_estimator_
        out_path = os.path.join(models_dir, f"{name}.pkl")
        save_pickle(best, out_path)
        return out_path

    paths = {
        "LR": grid_fit_save("LR", lr, lr_grid),
        "RF": grid_fit_save("RF", rf, rf_grid),
        "GB": grid_fit_save("GB", gb, gb_grid),
    }
    logging.info("Saved models: %s", paths)
    return paths


# ----------------------------- Evaluation / Prediction ----------------------------- #

def plot_overlay_roc_curves(df_name: str, filename: str,
                            true_y_list: List[np.ndarray],
                            prob_y_list: List[np.ndarray],
                            labels: List[str]) -> None:
    """Plot multiple ROC curves on one figure and save."""
    plt.figure(figsize=(8, 6))
    for y_true, y_prob, lbl in zip(true_y_list, prob_y_list, labels):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lbl} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title(f"ROC Curves: {df_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def predict_and_eval_single(model, X: pd.DataFrame, y: pd.Series, tag: str) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Predict using a fitted pipeline `model`. Returns (auc, acc, f1, y_true, y_prob).
    Threshold fixed at 0.5 for F1.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int) #tuning for usage case
    auc_val = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    logging.info("%s | AUC=%.3f ACC=%.3f F1=%.3f | positives=%d / n=%d | max_prob=%.3f",
                 tag, auc_val, acc, f1, int(y_pred.sum()), len(y), float(y_prob.max()))
    return auc_val, acc, f1, y.to_numpy(), y_prob


# ----------------------------- Orchestration ----------------------------- #

def train_main_cohort(
    main_baseline_csv: str,
    embed_supervised_csv: str | None,
    embed_unsupervised_csvs: List[str] | None,
    models_dir: str,
    seed: int = 100
) -> Dict[str, Dict[str, str]]:
    """
    Builds main-cohort training datasets and trains the best models.
    NOTE: This function ultimately saves ONE set of LR/RF/GB models to models_dir.
    If you pass multiple embeddings, consider running separate invocations with different models_dir.
    """
    base = load_baseline(main_baseline_csv, drop_specific=True)

    datasets: Dict[str, pd.DataFrame] = {}

    if embed_supervised_csv:
        emb = load_embedding_any(embed_supervised_csv)
        datasets["Supervised"] = merge_flex(base, emb)

    if embed_unsupervised_csvs:
        for i, p in enumerate(embed_unsupervised_csvs, start=1):
            emb = load_embedding_any(p)
            label = "Unsupervised" if len(embed_unsupervised_csvs) == 1 else f"Unsupervised_{i}"
            datasets[label] = merge_flex(base, emb)

    if not datasets:
        raise SystemExit("No embeddings provided. Use --embed-supervised and/or --embed-unsupervised.")

    # Train on the FIRST dataset provided (to avoid overwriting).
    # If you wish to train multiple, run the script multiple times with different --models-dir.
    first_label = next(iter(datasets))
    df = datasets[first_label]
    logging.info("Training on dataset: %s | shape=%s", first_label, df.shape)

    X = df.drop(columns=[c for c in ["target", "EMPI_NBR"] if c in df.columns])
    y = df["target"]
    paths = fit_best_models(X, y, models_dir=models_dir, seed=seed)
    return {first_label: paths}


def predict_external_cohort(
    models_dir: str,
    figs_dir: str,
    external_baseline_csv: str,
    external_embed_supervised_csv: str | None,
    external_embed_unsupervised_paths: List[str] | None
) -> None:
    """
    Builds external-cohort datasets and evaluates with previously saved models.
    Saves ROC figures to figs_dir.
    """
    base = load_baseline(external_baseline_csv, drop_specific=False)

    datasets: Dict[str, pd.DataFrame] = {}

    if external_embed_supervised_csv:
        emb = load_embedding_any(external_embed_supervised_csv)
        datasets["external_supervised"] = merge_flex(base, emb)

    if external_embed_unsupervised_paths:
        for i, p in enumerate(external_embed_unsupervised_paths, start=1):
            emb = load_embedding_any(p)
            label = f"external_unsupervised_{i}" if len(external_embed_unsupervised_paths) > 1 else "external_unsupervised"
            datasets[label] = merge_flex(base, emb)

    if not datasets:
        raise SystemExit("No external embeddings provided. Use --external-embed-supervised and/or --external-embed-unsupervised.")

    # Load models (single set saved in models_dir)
    model_paths = {
        "LR": os.path.join(models_dir, "LR.pkl"),
        "RF": os.path.join(models_dir, "RF.pkl"),
        "GB": os.path.join(models_dir, "GB.pkl"),
    }
    models = {k: load_pickle(v) for k, v in model_paths.items() if os.path.exists(v)}
    if not models:
        raise FileNotFoundError(f"No models found in {models_dir}. Train first with --train-main.")

    for name, df in datasets.items():
        logging.info("Predicting dataset: %s | shape=%s", name, df.shape)
        X = df.drop(columns=[c for c in ["target", "EMPI_NBR"] if c in df.columns])
        y = df["target"]

        labels, y_true_list, y_prob_list = [], [], []
        for label, model in models.items():
            auc_val, acc, f1, y_true, y_prob = predict_and_eval_single(model, X, y, f"{name} | {label}")
            labels.append(label)
            y_true_list.append(y_true)
            y_prob_list.append(y_prob)

        fig_path = os.path.join(figs_dir, f"{slugify(name)}.png")
        plot_overlay_roc_curves(name, fig_path, y_true_list, y_prob_list, labels)
        logging.info("Saved ROC: %s", fig_path)


# ----------------------------- CLI ----------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cohort experiment (single file).")

    # mode switches
    p.add_argument("--train-main", action="store_true", help="Train models on the main cohort.")
    p.add_argument("--predict-external", action="store_true", help="Run external validation on an external cohort.")

    # common
    p.add_argument("--models-dir", required=True, help="Directory to save/load models.")
    p.add_argument("--figs-dir", required=True, help="Directory to save figures.")
    p.add_argument("--seed", type=int, default=100)

    # Main cohort inputs
    p.add_argument("--main-baseline", help="Main cohort baseline CSV.")
    p.add_argument("--embed-supervised", help="Supervised embedding CSV for the main cohort (optional).")
    p.add_argument("--embed-unsupervised", nargs="+", help="Unsupervised embedding file(s) for the main cohort (CSV or NPY).")

    # External cohort inputs
    p.add_argument("--external-baseline", help="External cohort baseline CSV.")
    p.add_argument("--external-embed-supervised", help="Supervised embedding CSV for the external cohort (optional).")
    p.add_argument("--external-embed-unsupervised", nargs="+", help="Unsupervised embedding file(s) for the external cohort (CSV or NPY).")
    return p


def main():
    setup_logging()
    args = build_parser().parse_args()

    # Validate minimal args for chosen modes
    if not args.train_main and not args.predict_external:
        raise SystemExit("Nothing to do. Add --train-main and/or --predict-external.")

    ensure_dirs(args.models_dir, args.figs_dir)

    if args.train_main:
        if not args.main_baseline:
            raise SystemExit("--train-main requires --main-baseline and at least one embedding (--embed-supervised and/or --embed-unsupervised).")
        train_main_cohort(
            main_baseline_csv=args.main_baseline,
            embed_supervised_csv=args.embed_supervised,
            embed_unsupervised_csvs=args.embed_unsupervised,
            models_dir=args.models_dir,
            seed=args.seed,
        )

    if args.predict_external:
        if not args.external_baseline:
            raise SystemExit("--predict-external requires --external-baseline and external embeddings (--external-embed-supervised and/or --external-embed-unsupervised).")
        predict_external_cohort(
            models_dir=args.models_dir,
            figs_dir=args.figs_dir,
            external_baseline_csv=args.external_baseline,
            external_embed_supervised_csv=args.external_embed_supervised,
            external_embed_unsupervised_paths=args.external_embed_unsupervised,
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
