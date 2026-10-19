"""Narrow UNSW-NB15 real-data feasibility, NOT reproduction of AMCF paper results.

Requires the actual UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv.
No synthetic fallback, no SOC incident-level claims, no physical actuation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPECTED_TRAIN, EXPECTED_TEST = 175_341, 82_332
CATEGORICAL = ("proto", "service", "state")
DROP_COLS = ("id", "label", "attack_cat")


def load_release(train_csv: Path, test_csv: Path, *, allow_partial: bool = False):
    for file in (train_csv, test_csv):
        if not file.is_file():
            raise FileNotFoundError(
                f"Missing genuine UNSW-NB15 CSV: {file}. Download the official "
                "training/testing release; no demonstration data are substituted."
            )
    train, test = pd.read_csv(train_csv), pd.read_csv(test_csv)
    for label, frame, expected in (("train", train, EXPECTED_TRAIN),
                                    ("test", test, EXPECTED_TEST)):
        needed = {"label", *CATEGORICAL}
        missing = needed.difference(frame.columns)
        if missing:
            raise ValueError(f"{label}: missing columns {sorted(missing)}")
        if not allow_partial and len(frame) != expected:
            raise ValueError(f"{label}: expected {expected:,} rows in official release, "
                             f"found {len(frame):,}. Use --allow-partial ONLY for smoke tests.")
        labels = pd.to_numeric(frame["label"], errors="raise")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError(f"{label}: labels must be exactly binary 0/1")
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{label}: both classes are needed")
    if set(train.columns) != set(test.columns):
        raise ValueError("Training and testing files have different column sets")
    fields = [x for x in train.columns if x not in DROP_COLS]
    if not fields:
        raise ValueError("No non-label features")
    return train, test, fields


def subsample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    sample, _ = train_test_split(df, train_size=limit, random_state=seed,
                                 stratify=df["label"])
    return sample.reset_index(drop=True)


def preprocessing(train_frame: pd.DataFrame, columns: list[str]):
    cat_cols = [x for x in CATEGORICAL if x in columns]
    num_cols = [x for x in columns if x not in cat_cols]
    tx = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ], sparse_threshold=0.0)
    return tx, num_cols, cat_cols


def clean_features(frame: pd.DataFrame, numeric: list[str], categorical: list[str]):
    x = frame[numeric + categorical].copy()
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        x[col] = x[col].astype("string").fillna("unknown").astype(str)
    return x


def fixed_fpr_threshold(scores: np.ndarray, benign: np.ndarray, target_fpr: float) -> float:
    """Select threshold on held-out tuning negatives; never use test labels."""
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")
    negatives = np.asarray(scores, dtype=float)[np.asarray(benign) == 0]
    if negatives.size == 0 or not np.all(np.isfinite(negatives)):
        raise ValueError("Threshold selection requires finite benign scores")
    allowed = int(np.floor(target_fpr * negatives.size))
    cutoff = np.sort(negatives)[::-1][allowed]
    return float(np.nextafter(cutoff, np.inf))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0., 1., bins + 1)
    ids = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    return float(sum(np.mean(ids == k) * abs(y[ids == k].mean() - p[ids == k].mean())
                     for k in range(bins) if np.any(ids == k)))


def metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "auc_roc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "ece_10_bins": ece(y, proba),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "fpr": fp / (fp + tn),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": float(threshold),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    train_full, test_full, columns = load_release(args.train, args.test,
                                                   allow_partial=args.allow_partial)
    train = subsample(train_full, args.max_train, args.seed)
    test = subsample(test_full, args.max_test, args.seed + 1)
    # TRAIN => train (70%), calibration (15%), operating-point tuning (15%).
    dev, threshold_set = train_test_split(train, test_size=0.15, random_state=args.seed,
                                          stratify=train["label"])
    fit_set, calibration_set = train_test_split(dev, test_size=0.15 / 0.85,
                                                random_state=args.seed + 1,
                                                stratify=dev["label"])
    transform, nums, cats = preprocessing(fit_set, columns)
    xfit = transform.fit_transform(clean_features(fit_set, nums, cats))
    xcal = transform.transform(clean_features(calibration_set, nums, cats))
    xtune = transform.transform(clean_features(threshold_set, nums, cats))
    xtest = transform.transform(clean_features(test, nums, cats))
    yfit = fit_set.label.to_numpy(dtype=int)
    ycal = calibration_set.label.to_numpy(dtype=int)
    ytune = threshold_set.label.to_numpy(dtype=int)
    ytest = test.label.to_numpy(dtype=int)

    models = {
        "logistic_baseline": LogisticRegression(max_iter=500, random_state=args.seed),
        "hgb_baseline": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, random_state=args.seed),
    }
    probs = {}
    for name, model in models.items():
        model.fit(xfit, yfit)
        raw_cal = model.predict_proba(xcal)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_cal, ycal)
        probs[name] = (
            np.asarray(calibrator.predict(model.predict_proba(xtune)[:, 1])),
            np.asarray(calibrator.predict(model.predict_proba(xtest)[:, 1])),
        )
    eps = 1e-6
    def fusion(p1, p2):
        p1, p2 = np.clip(p1, eps, 1-eps), np.clip(p2, eps, 1-eps)
        logit = (np.log(p1 / (1-p1)) + np.log(p2 / (1-p2))) / 2
        return 1 / (1 + np.exp(-logit))

    # AMCF-lite: calibrated, equal-weight logit fusion and a bounded risk gate.
    # NO coherence, memory, VoI, drift adaptation, online learning, or actuation.
    (lt, le), (ht, he) = probs["logistic_baseline"], probs["hgb_baseline"]
    probs["amcf_lite_inference_only"] = fusion(lt, ht), fusion(le, he)
    rows = []
    for name, (ptune, ptest) in probs.items():
        threshold = fixed_fpr_threshold(ptune, ytune, args.target_fpr)
        if name == "amcf_lite_inference_only":
            # Ignore risk = p * C_FN; escalation risk = C_TRIAGE.
            # Only virtual ignore/escalate decisions (not real remediation).
            threshold = max(threshold, args.c_triage / args.c_fn)
        rows.append({"model": name, **metrics(ytest, ptest, threshold)})

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "pilot_metrics.csv", index=False)
    provenance = {
        "status": "real_unsw_nb15_feasibility_not_manuscript_reproduction",
        "source": "UNSW-NB15 official designated training/testing CSV release",
        "train_csv": str(args.train.resolve()), "test_csv": str(args.test.resolve()),
        "sha256_train": sha256(args.train), "sha256_test": sha256(args.test),
        "release_train_rows": len(train_full), "release_test_rows": len(test_full),
        "sampled_train_rows": len(train), "sampled_test_rows": len(test),
        "fit_rows": len(fit_set), "calibration_rows": len(calibration_set),
        "threshold_tuning_rows": len(threshold_set), "feature_count_after_encoding": int(xfit.shape[1]),
        "seed": args.seed, "target_fpr": args.target_fpr,
        "labels_test_positive": int(ytest.sum()), "labels_test_negative": int((ytest == 0).sum()),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        "limitations": [
            "No verified event timestamps or incident identifiers in designated training/testing release.",
            "FPR (not FP/hour) and flow-level recall (not long-dwell recall) are reported.",
            "AMCF-lite is a decision-only two-model proxy, NOT the full AMCF architecture.",
            "Release-defined train/test separation, not verified chronological splits.",
            "No containment, safety risk, operator study, ablation, or original R20 metrics reproduced.",
            "A partial/subsampled pilot is not a full-scale benchmark assessment.",
        ],
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).round(5).to_string(index=False))
    print(f"\nProvenance and pilot metrics saved to {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True, help="Actual UNSW_NB15_training-set.csv")
    p.add_argument("--test", type=Path, required=True, help="Actual UNSW_NB15_testing-set.csv")
    p.add_argument("--output", type=Path, default=Path("results/feasibility/unsw_nb15"))
    p.add_argument("--max-train", type=int, default=20_000, help="0 means full train")
    p.add_argument("--max-test", type=int, default=10_000, help="0 means full test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--c-triage", type=float, default=3.0)
    p.add_argument("--c-fn", type=float, default=15.0)
    p.add_argument("--allow-partial", action="store_true", help="Allow noncanonical CSV sizes for smoke testing")
    args = p.parse_args(argv)
    if args.max_train != 0 and args.max_train < 100:
        p.error("--max-train must be 0 or >= 100")
    if args.max_test != 0 and args.max_test < 100:
        p.error("--max-test must be 0 or >= 100")
    if args.c_triage <= 0 or args.c_fn <= 0:
        p.error("Cost parameters must be positive")
    if not 0 < args.target_fpr < 1:
        p.error("--target-fpr must lie in (0,1)")
    return args


if __name__ == "__main__":
    run(parse_args())