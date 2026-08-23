import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    POLLUTANTS,
    load_and_prepare,
    build_train_val_windows,
    build_full_windows,
    make_train_mask,
    CitySeqDataset
)
from src.models.imputer import BRATISeasonal

DATA_CSV = "data/data.csv"
OUTPUT_CSV = "outputs/data_imputed.csv"
BEST_MODEL_PATH = "outputs/best_model.pt"
SEARCH_LOG_PATH = "outputs/search_log.json"
SEED = 42

# Parse command line arguments for imputer training
def build_arg_parser():
    p = argparse.ArgumentParser(description="BRATI-Seasonal pollutant imputation pipeline")
    p.add_argument("--data-csv", default=DATA_CSV)
    p.add_argument("--output-csv", default=OUTPUT_CSV)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=365)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--embed-city", type=int, default=16)
    p.add_argument("--embed-time", type=int, default=16)
    p.add_argument("--embed-meta-fc", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--masking-prob", type=float, default=0.2)
    p.add_argument("--block-mask-prob", type=float, default=0.5)
    p.add_argument("--min-block", type=int, default=7)
    p.add_argument("--max-block", type=int, default=90)
    p.add_argument("--patience", type=int, default=10, help="Early-stop after N epochs with no val improvement.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-doy-embed", action="store_true", help="Disable day-of-year embedding.")
    p.add_argument("--search", action="store_true", help="Perform hyperparameter search.")
    p.add_argument("--search-epochs", type=int, default=8, help="Epochs per config during search.")
    return p

# Train model for one specific hyperparameter configuration
def train_one_config(cfg, raw, values_filled, mask, meta_vals, num_cities, device, epochs, verbose=True):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_windows, val_windows = build_train_val_windows(raw, cfg.seq_len, val_frac=0.15)
    train_dataset = CitySeqDataset(values_filled, mask, meta_vals, cfg.seq_len, train_windows)
    val_dataset = CitySeqDataset(values_filled, mask, meta_vals, cfg.seq_len, val_windows)
    dataset = CitySeqDataset(values_filled, mask, meta_vals, cfg.seq_len, build_full_windows(raw, cfg.seq_len))

    if verbose:
        print(f"  train windows: {len(train_dataset)}  val windows: {len(val_dataset)}  (seq_len={cfg.seq_len})")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = BRATISeasonal(num_pollutants=len(POLLUTANTS), num_cities=num_cities, cfg=cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss(reduction="sum")

    best_val, best_state, epochs_since_improve = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_points = 0.0, 0
        for batch in train_loader:
            vals = batch["values"].to(device)
            m = batch["mask"].to(device)
            meta = batch["meta"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            train_m = make_train_mask(m, cfg, block_mask=True)
            preds = model(vals * train_m, train_m, meta, pad_mask)

            target_mask = ((train_m == 0.0) & (m == 1.0)).float() * pad_mask.unsqueeze(-1)
            n_pts = target_mask.sum().item()
            if n_pts == 0:
                continue
            loss = criterion(preds * target_mask, vals * target_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_points += n_pts

        model.eval()
        val_loss, val_points = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                vals = batch["values"].to(device)
                m = batch["mask"].to(device)
                meta = batch["meta"].to(device)
                pad_mask = batch["pad_mask"].to(device)
                train_m = make_train_mask(m, cfg, block_mask=True)
                preds = model(vals * train_m, train_m, meta, pad_mask)
                target_mask = ((train_m == 0.0) & (m == 1.0)).float() * pad_mask.unsqueeze(-1)
                n_pts = target_mask.sum().item()
                if n_pts == 0:
                    continue
                loss = criterion(preds * target_mask, vals * target_mask)
                val_loss += loss.item()
                val_points += n_pts

        avg_train = train_loss / train_points if train_points > 0 else 0.0
        avg_val = val_loss / val_points if val_points > 0 else 0.0
        history.append({"epoch": epoch, "train_mse": avg_train, "val_mse": avg_val})
        if verbose:
            print(f"  Epoch {epoch}/{epochs}  train_mse={avg_train:.6f}  val_mse={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if cfg.patience and epochs_since_improve >= cfg.patience:
                if verbose:
                    print(f"  Early stopping (best val_mse={best_val:.6f} at epoch {epoch - epochs_since_improve})")
                break

    return best_val, best_state, history, dataset

# Run inference over full dataset and write non-negative imputed CSV
def impute_full_dataset(model, dataset, values_filled, mask, raw, scalers, cfg, device, output_csv):
    model.eval()
    with torch.no_grad():
        preds_accum = np.zeros_like(values_filled, dtype=float)
        preds_count = np.zeros_like(values_filled, dtype=float)

        for batch in DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False):
            starts = batch["start"].numpy()
            lengths = batch["length"].numpy()
            vals = batch["values"].to(device)
            m = batch["mask"].to(device)
            meta = batch["meta"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            preds = model(vals * m, m, meta, pad_mask).cpu().numpy()
            for i, (s, length) in enumerate(zip(starts, lengths)):
                preds_accum[s:s + length, :] += preds[i, :length]
                preds_count[s:s + length, :] += 1.0

        covered = preds_count > 0
        print(f"Rows covered by predictions: {covered.any(axis=1).sum()} / {len(raw)}")
        averaged = np.zeros_like(preds_accum)
        averaged[covered] = preds_accum[covered] / preds_count[covered]

        imputed_scaled = values_filled.copy()
        missing_positions = (mask == 0)
        imputed_scaled[missing_positions] = averaged[missing_positions]

        imputed_final = imputed_scaled.copy()
        for idx, p in enumerate(POLLUTANTS):
            imputed_final[:, idx] = scalers[p].inverse_transform(imputed_final[:, idx].reshape(-1, 1)).flatten()
        imputed_final[missing_positions] = np.clip(imputed_final[missing_positions], 0.0, None)

    out_df = raw.copy()
    for i, p in enumerate(POLLUTANTS):
        out_df[p] = imputed_final[:, i]
    out_df = (out_df.sort_values("_original_index")
              .drop(columns=["_original_index", "_sorted_index"])
              .reset_index(drop=True))

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved imputed CSV to {output_csv}")

# Default candidate configs for search
def default_search_space():
    return [
        dict(hidden_size=128, lr=1e-3, seq_len=365, batch_size=32, num_layers=2),
        dict(hidden_size=256, lr=1e-3, seq_len=365, batch_size=32, num_layers=2),
        dict(hidden_size=128, lr=5e-4, seq_len=180, batch_size=64, num_layers=2),
        dict(hidden_size=128, lr=1e-3, seq_len=365, batch_size=32, num_layers=3),
    ]

# Execute hyperparameter search across default configuration space
def run_search(base_args, device):
    raw, values_filled, mask, meta_vals, scalers, num_cities = load_and_prepare(base_args.data_csv)
    search_space = default_search_space()

    results = []
    for i, overrides in enumerate(search_space):
        print(f"\n=== Search trial {i+1}/{len(search_space)}: {overrides} ===")
        cfg = argparse.Namespace(**vars(base_args))
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.patience = min(cfg.patience, base_args.search_epochs) if cfg.patience else cfg.patience

        best_val, _, _, _ = train_one_config(cfg, raw, values_filled, mask, meta_vals, num_cities, device, epochs=base_args.search_epochs, verbose=True)
        results.append({"config": overrides, "best_val_mse": best_val})
        print(f"  -> best_val_mse={best_val:.6f}")

    results.sort(key=lambda r: r["best_val_mse"])
    os.makedirs("outputs", exist_ok=True)
    with open(SEARCH_LOG_PATH, "w") as f:
        json.dump(results, f, indent=2)

    winner = results[0]["config"]
    print(f"\nBest config: {winner}\nTraining full run with winner...")

    final_cfg = argparse.Namespace(**vars(base_args))
    for k, v in winner.items():
        setattr(final_cfg, k, v)

    best_val, best_state, _, dataset = train_one_config(final_cfg, raw, values_filled, mask, meta_vals, num_cities, device, epochs=base_args.epochs, verbose=True)
    model = BRATISeasonal(num_pollutants=len(POLLUTANTS), num_cities=num_cities, cfg=final_cfg).to(device)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": vars(final_cfg), "val_mse": best_val}, BEST_MODEL_PATH)
    print(f"Saved model to {BEST_MODEL_PATH}")

    impute_full_dataset(model, dataset, values_filled, mask, raw, scalers, final_cfg, device, base_args.output_csv)

# Entry function to run imputation training or hyperparameter search
def run_imputation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"{args.data_csv} not found")

    if args.search:
        run_search(args, device)
        return

    raw, values_filled, mask, meta_vals, scalers, num_cities = load_and_prepare(args.data_csv)
    print(f"Training imputer: epochs={args.epochs} batch_size={args.batch_size} seq_len={args.seq_len} hidden_size={args.hidden_size} lr={args.lr}")

    best_val, best_state, _, dataset = train_one_config(args, raw, values_filled, mask, meta_vals, num_cities, device, epochs=args.epochs, verbose=True)

    model = BRATISeasonal(num_pollutants=len(POLLUTANTS), num_cities=num_cities, cfg=args).to(device)
    model.load_state_dict(best_state)
    os.makedirs("outputs", exist_ok=True)
    torch.save({"state_dict": best_state, "config": vars(args), "val_mse": best_val}, BEST_MODEL_PATH)
    print(f"Saved model to {BEST_MODEL_PATH}")

    impute_full_dataset(model, dataset, values_filled, mask, raw, scalers, args, device, args.output_csv)
