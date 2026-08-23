import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from src.data.dataset import POLLUTANTS

# Identify contiguous True runs in boolean array
def find_contiguous_runs(bool_array):
    runs = []
    in_run = False
    start = None
    for i, v in enumerate(bool_array):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(bool_array) - 1))
    return runs

# Generate multi-page PDF comparing original and imputed time-series for each city
def plot_before_after_pdf(df_orig, df_imp, cities, out_path, highlight="shade"):
    with PdfPages(out_path) as pdf:
        for city in cities:
            o = df_orig[df_orig["City"] == city].sort_values("Date")
            imp = df_imp[df_imp["City"] == city].sort_values("Date")
            dates = pd.to_datetime(imp["Date"]).reset_index(drop=True)

            fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
            fig.suptitle(f"Before / After Imputation — {city}", fontsize=14, y=1.0)

            for ax, pol in zip(axes.flat, POLLUTANTS):
                was_missing = o[pol].isna().values
                ax.plot(dates, imp[pol].values, color="#1f77b4", linewidth=0.9, label="Pollutant value (imputed points blended in)")

                if highlight == "dots" and was_missing.any():
                    ax.scatter(dates[was_missing], imp[pol].values[was_missing], color="#ff7f0e", s=6, zorder=3, label="Imputed (was missing)")
                elif highlight == "shade" and was_missing.any():
                    runs = find_contiguous_runs(was_missing)
                    for i, (start, end) in enumerate(runs):
                        x0 = dates.iloc[start] - pd.Timedelta(hours=12)
                        x1 = dates.iloc[end] + pd.Timedelta(hours=12)
                        ax.axvspan(x0, x1, color="#ff7f0e", alpha=0.15, linewidth=0, label="Imputed stretch" if i == 0 else None)

                ax.set_title(pol, fontsize=10)
                ax.tick_params(labelsize=7)

            handles, labels = axes.flat[0].get_legend_handles_labels()
            seen = {l: h for h, l in zip(handles, labels)}
            fig.legend(seen.values(), seen.keys(), loc="upper right", fontsize=9)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

# Save horizontal bar chart ranking cities by held-out imputation error
def plot_city_ranking(city_summary, out_path):
    ranked = city_summary.sort_values("MAE_normalized")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r((ranked["MAE_normalized"] - ranked["MAE_normalized"].min()) /
                              (ranked["MAE_normalized"].max() - ranked["MAE_normalized"].min() + 1e-9))
    ax.barh(ranked["City"], ranked["MAE_normalized"], color=colors)
    ax.set_xlabel("Normalized MAE (unit-free, lower = better)")
    ax.set_title("Per-City Imputation Quality (held-out evaluation)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

# Plot R2 and RMSE performance comparison for regression models
def plot_regression_comparison(results_df, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = results_df.sort_values("R2")
    axes[0].barh(order["Model"], order["R2"], color="#4a7fb5")
    axes[0].set_xlabel("R² (higher is better)")
    axes[0].set_title("Regression: R² by model")
    order2 = results_df.sort_values("RMSE", ascending=False)
    axes[1].barh(order2["Model"], order2["RMSE"], color="#d67f4a")
    axes[1].set_xlabel("RMSE (lower is better)")
    axes[1].set_title("Regression: RMSE by model")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

# Plot accuracy vs weighted F1 score comparison for classification models
def plot_classification_comparison(results_df, fig_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = results_df.sort_values("F1")
    x = np.arange(len(order))
    width = 0.35
    ax.barh(x - width / 2, order["Accuracy"], height=width, label="Accuracy", color="#4a7fb5")
    ax.barh(x + width / 2, order["F1"], height=width, label="F1 (weighted)", color="#d67f4a")
    ax.set_yticks(x)
    ax.set_yticklabels(order["Model"])
    ax.set_xlabel("Score")
    ax.set_title("Classification: Accuracy vs F1 by model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

# Save confusion matrix plot for best performing classifier
def plot_confusion_matrix(best_name, cm, display_labels, cm_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — Best Model: {best_name}")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
