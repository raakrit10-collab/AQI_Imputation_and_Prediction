import os
import argparse
import numpy as np
import pandas as pd
import torch

from src.data.dataset import POLLUTANTS, load_and_prepare, build_nonoverlapping_windows
from src.models.imputer import BRATISeasonal
from src.utils.visualization import plot_before_after_pdf, plot_city_ranking

# Parse command line arguments for evaluation script
def build_arg_parser():
    p = argparse.ArgumentParser(description="Evaluate BRATI-Seasonal imputation results per city")
    p.add_argument("--data-csv", default="data/data.csv")
    p.add_argument("--imputed-csv", default="outputs/data_imputed.csv")
    p.add_argument("--model-path", default="outputs/best_model.pt")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--highlight", choices=["shade", "dots", "none"], default="shade")
    return p

# Create evaluation batch tensors from sequence windows
def make_eval_batch(values, mask, meta, windows, seq_len, n_pollutants, n_meta):
    B = len(windows)
    v = np.zeros((B, seq_len, n_pollutants), dtype=np.float32)
    m = np.zeros((B, seq_len, n_pollutants), dtype=np.float32)
    me = np.zeros((B, seq_len, n_meta), dtype=np.float32)
    pad = np.zeros((B, seq_len), dtype=np.float32)
    starts, lengths = [], []
    for i, (s, length) in enumerate(windows):
        e = s + length
        v[i, :length] = values[s:e]
        m[i, :length] = mask[s:e]
        me[i, :length] = meta[s:e]
        pad[i, :length] = 1.0
        starts.append(s)
        lengths.append(length)
    return (torch.tensor(v), torch.tensor(m), torch.tensor(me), torch.tensor(pad), np.array(starts), np.array(lengths))

# Perform held-out evaluation on artificially hidden observed entries
def evaluate_per_city(model, cfg, raw, values_filled, mask, meta_vals, scalers, device, holdout_frac, seed):
    n_rows, n_pol = values_filled.shape
    n_meta = meta_vals.shape[1]

    city_ids = raw["city_idx"].values
    windows = build_nonoverlapping_windows(0, n_rows, cfg.seq_len)
    windows_tuple = [(w[0], w[1]) for w in windows]

    v, m, me, pad, starts, lengths = make_eval_batch(
        values_filled, mask, meta_vals, windows_tuple, cfg.seq_len, n_pol, n_meta
    )

    torch.manual_seed(seed)
    rand = torch.rand_like(m)
    holdout = (m == 1) & (rand < holdout_frac)
    m_eval = m.clone()
    m_eval[holdout] = 0.0

    batch_size = 32
    n = v.shape[0]
    all_rows = []

    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, batch_size):
            sl = slice(i0, min(i0 + batch_size, n))
            vb = v[sl].to(device)
            mb = m_eval[sl].to(device)
            meb = me[sl].to(device)
            padb = pad[sl].to(device)
            preds = model(vb * mb, mb, meb, padb).cpu().numpy()

            hb = holdout[sl].numpy()
            true_b = v[sl].numpy()
            for i, (s, length) in enumerate(zip(starts[sl], lengths[sl])):
                hb_i = hb[i, :length]
                if not hb_i.any():
                    continue
                rows, pols = np.where(hb_i)
                all_rows.append(np.column_stack([
                    rows + s,
                    pols,
                    true_b[i, :length][hb_i],
                    preds[i, :length][hb_i],
                ]))

    arr = np.concatenate(all_rows, axis=0)
    row_idx = arr[:, 0].astype(int)
    pol_idx = arr[:, 1].astype(int)
    true_scaled = arr[:, 2]
    pred_scaled = arr[:, 3]

    stds = np.array([scalers[p].scale_[0] for p in POLLUTANTS])
    abs_err_real = np.abs(true_scaled - pred_scaled) * stds[pol_idx]
    sq_err_real = ((true_scaled - pred_scaled) * stds[pol_idx]) ** 2
    abs_err_scaled = np.abs(true_scaled - pred_scaled)
    sq_err_scaled = (true_scaled - pred_scaled) ** 2

    city_of_row = city_ids[row_idx]
    df_err = pd.DataFrame({
        "city_idx": city_of_row,
        "pollutant": [POLLUTANTS[p] for p in pol_idx],
        "abs_err_real": abs_err_real,
        "sq_err_real": sq_err_real,
        "abs_err_scaled": abs_err_scaled,
        "sq_err_scaled": sq_err_scaled,
    })
    return df_err

# Entry point function to evaluate model and generate summary plots
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    raw, values_filled, mask, meta_vals, scalers, num_cities = load_and_prepare(args.data_csv)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = argparse.Namespace(**ckpt["config"])
    model = BRATISeasonal(num_pollutants=len(POLLUTANTS), num_cities=num_cities, cfg=cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model (training val_mse={ckpt.get('val_mse'):.6f}, seq_len={cfg.seq_len})")

    print(f"Running held-out evaluation (holdout_frac={args.holdout_frac})...")
    df_err = evaluate_per_city(model, cfg, raw, values_filled, mask, meta_vals, scalers, device, args.holdout_frac, args.seed)

    idx_to_city = raw.drop_duplicates("city_idx").set_index("city_idx")["City"].to_dict()
    df_err["City"] = df_err["city_idx"].map(idx_to_city)

    by_pol = (df_err.groupby(["City", "pollutant"])
              .agg(n_holdout=("abs_err_real", "size"),
                   MAE_real=("abs_err_real", "mean"),
                   RMSE_real=("sq_err_real", lambda s: np.sqrt(s.mean())))
              .reset_index())
    by_pol_path = os.path.join(args.out_dir, "city_imputation_metrics_by_pollutant.csv")
    by_pol.to_csv(by_pol_path, index=False)
    print(f"Saved {by_pol_path}")

    by_city = (df_err.groupby("City")
               .agg(n_holdout=("abs_err_scaled", "size"),
                    MAE_normalized=("abs_err_scaled", "mean"),
                    RMSE_normalized=("sq_err_scaled", lambda s: np.sqrt(s.mean())))
               .reset_index()
               .sort_values("MAE_normalized"))
    by_city_path = os.path.join(args.out_dir, "city_imputation_metrics.csv")
    by_city.to_csv(by_city_path, index=False)
    print(f"Saved {by_city_path}")

    df_orig = pd.read_csv(args.data_csv)
    df_imp = pd.read_csv(args.imputed_csv)
    cities = sorted(df_orig["City"].unique())

    pdf_path = os.path.join(args.fig_dir, "imputation_before_after.pdf")
    plot_before_after_pdf(df_orig, df_imp, cities, pdf_path, highlight=args.highlight)
    print(f"Saved {pdf_path}")

    ranking_path = os.path.join(args.fig_dir, "city_imputation_ranking.png")
    plot_city_ranking(by_city, ranking_path)
    print(f"Saved {ranking_path}")
