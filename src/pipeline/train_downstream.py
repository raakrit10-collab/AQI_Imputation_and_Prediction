import os
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data.dataset import POLLUTANTS, prepare_downstream_dataset
from src.models.ml_models import get_regression_models, get_classification_models
from src.utils.visualization import (
    plot_regression_comparison,
    plot_classification_comparison,
    plot_confusion_matrix
)

# Parse command line arguments for downstream benchmark
def build_arg_parser():
    p = argparse.ArgumentParser(description="Compare regression/classification models on imputed AQI data")
    p.add_argument("--imputed-csv", default="outputs/data_imputed.csv")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p

# Train and evaluate continuous AQI regression models
def run_regression(df, args):
    X = df[POLLUTANTS]
    y = df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    for name, model in get_regression_models(args.seed).items():
        t0 = time.time()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        fit_time = time.time() - t0

        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "Fit time (s)": fit_time})
        print(f"[Regression] {name:22s} MAE={mae:8.3f}  RMSE={rmse:8.3f}  R2={r2:.4f}  ({fit_time:.1f}s)")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    results_df["Best"] = ""
    results_df.loc[0, "Best"] = "★"

    out_path = os.path.join(args.out_dir, "regression_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    fig_path = os.path.join(args.fig_dir, "regression_comparison.png")
    plot_regression_comparison(results_df, fig_path)
    print(f"Saved {fig_path}")

    return results_df

# Train and evaluate AQI bucket classification models
def run_classification(df, args):
    le = LabelEncoder()
    y = le.fit_transform(df["AQI_Bucket"])
    X = df[POLLUTANTS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    preds_by_model = {}
    for name, model in get_classification_models(args.seed).items():
        t0 = time.time()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        fit_time = time.time() - t0
        preds_by_model[name] = pred

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "Fit time (s)": fit_time})
        print(f"[Classification] {name:22s} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ({fit_time:.1f}s)")

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    results_df["Best"] = ""
    results_df.loc[0, "Best"] = "★"

    out_path = os.path.join(args.out_dir, "classification_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    fig_path = os.path.join(args.fig_dir, "classification_comparison.png")
    plot_classification_comparison(results_df, fig_path)
    print(f"Saved {fig_path}")

    best_name = results_df.iloc[0]["Model"]
    best_pred = preds_by_model[best_name]
    cm = confusion_matrix(y_test, best_pred)
    cm_path = os.path.join(args.fig_dir, "best_classifier_confusion_matrix.png")
    plot_confusion_matrix(best_name, cm, le.classes_, cm_path)
    print(f"Saved {cm_path}")

    return results_df

# Entry point function to run complete regression and classification benchmark
def run_benchmark(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    print(f"Loading {args.imputed_csv}...")
    df = prepare_downstream_dataset(args.imputed_csv)

    print("\n--- REGRESSION ---")
    reg_results = run_regression(df, args)

    print("\n--- CLASSIFICATION ---")
    clf_results = run_classification(df, args)

    print("\n--- BENCHMARK SUMMARY ---")
    print(f"Best Regression Model:     {reg_results.iloc[0]['Model']} (R2={reg_results.iloc[0]['R2']:.4f}, MAE={reg_results.iloc[0]['MAE']:.3f})")
    print(f"Best Classification Model: {clf_results.iloc[0]['Model']} (F1={clf_results.iloc[0]['F1']:.4f}, Acc={clf_results.iloc[0]['Accuracy']:.4f})")
