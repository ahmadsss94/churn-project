import argparse
import json
import os
import sys
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Optional MLflow import
try:
    import mlflow

    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False


# -------------------------
# IO & utility helpers
# -------------------------
def _read_any(path: str) -> pd.DataFrame:
    """Read CSV/Parquet by extension; default to CSV."""
    ext = os.path.splitext(path)[-1].lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _align_features(df: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    """Align incoming batch to training feature order; add missing cols as NaN; drop extras."""
    X = df.copy()
    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan
    return X[feature_order]


def _ensure_parent(path: str) -> None:
    """Create the parent directory for a path if needed."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# -------------------------
# Metrics
# -------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index using quantile bins taken from the baseline ('expected').
    Falls back to equal-width bins when quantiles collapse.
    """
    expected = np.clip(expected, 0, 1)
    actual = np.clip(actual, 0, 1)

    qs = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, qs)
    cuts = np.unique(cuts)
    if cuts.size < 3:  # baseline scores too concentrated → fallback
        cuts = np.linspace(0, 1, bins + 1)

    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual, bins=cuts)

    e = e_hist.astype(float) / max(e_hist.sum(), 1.0)
    a = a_hist.astype(float) / max(a_hist.sum(), 1.0)

    # avoid log(0)
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)

    return float(np.sum((a - e) * np.log(a / e)))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Simple AP computation without sklearn (monotone PR step area).
    y_true ∈ {0,1}.
    """
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    # cumulative precision & recall
    cum_tp = np.cumsum(y_sorted == 1)
    precision = cum_tp / (np.arange(len(y_sorted)) + 1)
    positives = max(int((y_sorted == 1).sum()), 1)
    recall = cum_tp / positives

    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * float(r - prev_recall)
        prev_recall = float(r)
    return float(ap)


# -------------------------
# Artifacts
# -------------------------
def load_artifact(model_path: str):
    """
    Load a persisted artifact. Supports either:
    - dict with {"model": pipeline, "feature_order": [...], "threshold": float}
    - plain pipeline (fallback)
    """
    art = joblib.load(model_path)
    if isinstance(art, dict) and "model" in art and "feature_order" in art:
        return art
    return {"model": art, "feature_order": None, "threshold": 0.5}


# -------------------------
# Baseline creation
# -------------------------
def create_baseline(
    model_path: str,
    baseline_batch: str,
    out_scores: str,
    label_col: Optional[str] = None,
    out_metrics: Optional[str] = None,
) -> None:
    art = load_artifact(model_path)
    pipe = art["model"]
    feature_order = art.get("feature_order")

    df = _read_any(baseline_batch)
    X = df if feature_order is None else _align_features(df, feature_order)
    scores = pipe.predict_proba(X)[:, 1]

    _ensure_parent(out_scores)
    np.save(out_scores, scores)
    print(f"[baseline] Saved scores -> {out_scores} (n={len(scores)})")

    if out_metrics:
        _ensure_parent(out_metrics)

    if label_col and (label_col in df.columns):
        y = df[label_col].to_numpy().astype(int)
        pr_auc = average_precision(y, scores)
        metrics = {"pr_auc": float(pr_auc), "n": int(len(y))}
        if out_metrics:
            with open(out_metrics, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[baseline] Saved metrics -> {out_metrics} ({metrics})")
        else:
            print(f"[baseline] PR-AUC={pr_auc:.6f} (n={len(y)})")
    elif out_metrics:
        metrics = {"pr_auc": None, "n": int(len(scores))}
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[baseline] Saved metrics -> {out_metrics} (no labels)")


# -------------------------
# Monitoring
# -------------------------
def monitor(
    model_path: str,
    batch_path: str,
    baseline_scores_path: str,
    label_col: Optional[str],
    psi_warn: float,
    psi_action: float,
    drop_threshold: float,
    baseline_metrics_path: Optional[str] = None,
    experiment: str = "churn_monitoring",
    out_report: Optional[str] = None,
    out_scores: Optional[str] = None,
    psi_bins: int = 10,
) -> int:
    """
    Return exit code: 0 OK, 2 WARN, 3 ACTION
    """
    art = load_artifact(model_path)
    pipe = art["model"]
    feature_order = art.get("feature_order")

    df = _read_any(batch_path)
    X = df if feature_order is None else _align_features(df, feature_order)
    scores = pipe.predict_proba(X)[:, 1]

    # save scores if requested
    if out_scores:
        _ensure_parent(out_scores)
        np.save(out_scores, scores)

    baseline_scores = np.load(baseline_scores_path)
    psi_score = psi(baseline_scores, scores, bins=psi_bins)

    pr_auc = None
    pr_drop = None
    baseline_pr = None

    if label_col and (label_col in df.columns):
        y = df[label_col].to_numpy().astype(int)
        pr_auc = average_precision(y, scores)

        if baseline_metrics_path and os.path.exists(baseline_metrics_path):
            try:
                with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                if isinstance(m.get("pr_auc"), (int, float)) and (m.get("pr_auc") is not None):
                    baseline_pr = float(m["pr_auc"])
                    if baseline_pr > 0:
                        pr_drop = (baseline_pr - pr_auc) / baseline_pr
            except Exception as e:
                print(f"[monitor] Could not read baseline metrics: {e}", file=sys.stderr)

    status = "OK"
    exit_code = 0
    if psi_score >= psi_action:
        status = "ACTION"
        exit_code = 3
    elif psi_score >= psi_warn:
        status = "WARN"
        exit_code = 2

    if (pr_drop is not None) and (pr_drop >= drop_threshold):
        status = "ACTION"
        exit_code = max(exit_code, 3)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_batch": int(len(df)),
        "psi_score": float(psi_score),
        "status": status,
        "pr_auc": None if pr_auc is None else float(pr_auc),
        "baseline_pr_auc": baseline_pr,
        "pr_auc_drop_ratio": pr_drop,
        "psi_warn": psi_warn,
        "psi_action": psi_action,
        "drop_threshold": drop_threshold,
        "psi_bins": psi_bins,
    }
    print(json.dumps(summary, indent=2))

    # Save report JSON if requested
    if out_report:
        _ensure_parent(out_report)
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # MLflow logging (optional)
    if _HAS_MLFLOW:
        try:
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=f"monitor_{int(time.time())}"):
                mlflow.log_metric("psi_score", float(psi_score))
                mlflow.log_metric("n_batch", int(len(df)))
                if pr_auc is not None:
                    mlflow.log_metric("pr_auc", float(pr_auc))
                if baseline_pr is not None:
                    mlflow.log_metric("baseline_pr_auc", float(baseline_pr))
                if pr_drop is not None:
                    mlflow.log_metric("pr_auc_drop_ratio", float(pr_drop))
                mlflow.log_param("psi_warn", psi_warn)
                mlflow.log_param("psi_action", psi_action)
                mlflow.log_param("drop_threshold", drop_threshold)
                mlflow.log_param("psi_bins", psi_bins)
        except Exception as e:
            print(f"[monitor] MLflow logging failed: {e}", file=sys.stderr)

    return exit_code


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Churn model monitoring utility (baseline & monitor).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # baseline
    p_base = sub.add_parser("baseline", help="Create baseline score distribution (and optional baseline metrics).")
    p_base.add_argument("--model", required=True, help="Path to churn_model.joblib")
    p_base.add_argument("--batch", required=True, help="Baseline batch file (CSV or Parquet).")
    p_base.add_argument("--out-scores", required=True, help="Output .npy to save baseline scores.")
    p_base.add_argument("--label-col", default=None, help="Optional label column to compute baseline PR-AUC.")
    p_base.add_argument("--out-metrics", default=None, help="Optional JSON path to save baseline metrics.")

    # check
    p_mon = sub.add_parser("check", help="Monitor a new batch against baseline using PSI and optional PR-AUC drop.")
    p_mon.add_argument("--model", required=True, help="Path to churn_model.joblib")
    p_mon.add_argument("--batch", required=True, help="New batch file (CSV or Parquet).")
    p_mon.add_argument("--baseline-scores", required=True, help="Path to .npy baseline scores created by 'baseline'.")
    p_mon.add_argument("--label-col", default=None, help="Optional label column for PR-AUC.")
    p_mon.add_argument("--baseline-metrics", default=None, help="Optional JSON with baseline PR-AUC.")
    p_mon.add_argument("--psi-warn", type=float, default=float(os.getenv("MONITOR_PSI_WARN", 0.10)),
                       help="Warn threshold for PSI (default 0.10).")
    p_mon.add_argument("--psi-action", type=float, default=float(os.getenv("MONITOR_PSI_ACTION", 0.20)),
                       help="Action threshold for PSI (default 0.20).")
    p_mon.add_argument("--drop-threshold", type=float, default=float(os.getenv("MONITOR_PRAUC_DROP", 0.20)),
                       help="Relative PR-AUC drop to trigger action (default 0.20).")
    p_mon.add_argument("--experiment", default="churn_monitoring", help="MLflow experiment name.")
    p_mon.add_argument("--out-report", default=None, help="Optional JSON file to write the summary.")
    p_mon.add_argument("--out-scores", default=None, help="Optional .npy file to write batch scores.")
    p_mon.add_argument("--psi-bins", type=int, default=10, help="Number of PSI bins (default 10).")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "baseline":
        create_baseline(
            model_path=args.model,
            baseline_batch=args.batch,
            out_scores=args.out_scores,
            label_col=args.label_col,
            out_metrics=args.out_metrics,
        )
        return 0
    if args.cmd == "check":
        return monitor(
            model_path=args.model,
            batch_path=args.batch,
            baseline_scores_path=args.baseline_scores,
            label_col=args.label_col,
            psi_warn=args.psi_warn,
            psi_action=args.psi_action,
            drop_threshold=args.drop_threshold,
            baseline_metrics_path=args.baseline_metrics,
            experiment=args.experiment,
            out_report=args.out_report,
            out_scores=args.out_scores,
            psi_bins=args.psi_bins,
        )
    print("Unknown command", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
