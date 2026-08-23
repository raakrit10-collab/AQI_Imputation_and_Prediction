import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Target pollutants used for modeling
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

# Official CPCB breakpoint table for sub-index calculations
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    "NH3": [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2000, 401, 500)],
}

# Parse date strings into calendar and cyclic temporal features
def parse_dates(df, date_col="Date"):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df["_date_parsed"] = dates
    df["year"] = dates.dt.year.fillna(0).astype(int)
    df["day_of_year"] = dates.dt.dayofyear.fillna(0).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    df["month"] = dates.dt.month.fillna(0).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df

# Calculate individual pollutant sub-index based on CPCB breakpoints
def calc_sub_index(c, table):
    if pd.isna(c):
        return np.nan
    for bp_lo, bp_hi, i_lo, i_hi in table:
        if bp_lo <= c <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c - bp_lo) + i_lo
    if c > table[-1][1]:
        return table[-1][3]
    return np.nan

# Compute overall AQI value for a row
def calc_aqi_row(row, min_subindices=3):
    vals = [calc_sub_index(row.get(p), t) for p, t in BREAKPOINTS.items()]
    vals = [v for v in vals if not pd.isna(v)]
    if len(vals) < min_subindices:
        return np.nan
    return max(vals)

# Map continuous AQI score to discrete category bucket
def get_aqi_bucket(aqi):
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"

# Generate sliding window indices for sequence model training
def build_sliding_windows(city_ids, region_start, region_end, seq_len):
    windows = []
    if region_end - region_start >= seq_len:
        for s in range(region_start, region_end - seq_len + 1):
            windows.append((s, seq_len, False))
    elif region_end > region_start:
        windows.append((region_start, region_end - region_start, True))
    return windows

# Generate non-overlapping window indices for sequence evaluation
def build_nonoverlapping_windows(region_start, region_end, seq_len):
    windows = []
    pos = region_start
    while pos < region_end:
        length = min(seq_len, region_end - pos)
        windows.append((pos, length, length < seq_len))
        pos += length
    return windows

# Perform per-city chronological split into train and validation windows
def build_train_val_windows(df, seq_len, val_frac=0.15):
    city_ids = df["city_idx"].values
    n = len(df)
    train_windows, val_windows = [], []
    start = 0
    for i in range(1, n + 1):
        if i == n or city_ids[i] != city_ids[start]:
            city_start, city_end = start, i
            val_len = max(1, int(round((city_end - city_start) * val_frac)))
            train_end = city_end - val_len
            train_windows.extend(build_sliding_windows(city_ids, city_start, train_end, seq_len))
            val_windows.extend(build_nonoverlapping_windows(train_end, city_end, seq_len))
            start = i
    return train_windows, val_windows

# Generate non-overlapping windows covering all rows for dataset inference
def build_full_windows(df, seq_len):
    city_ids = df["city_idx"].values
    n = len(df)
    windows = []
    start = 0
    for i in range(1, n + 1):
        if i == n or city_ids[i] != city_ids[start]:
            windows.extend(build_nonoverlapping_windows(start, i, seq_len))
            start = i
    return windows

# Apply contiguous block masking to simulate sensor outage patterns
def apply_block_mask(mask_tensor, min_block, max_block, prob):
    B, T, P = mask_tensor.shape
    out = mask_tensor.clone()
    for b in range(B):
        if random.random() > prob:
            continue
        for _ in range(random.randint(1, 2)):
            length = random.randint(min_block, min(max_block, T))
            start = random.randint(0, max(0, T - length))
            out[b, start:start + length, :] = 0.0
    return out

# Combine block masking and random masking for training data augmentation
def make_train_mask(orig_mask, cfg, block_mask=True):
    if block_mask:
        m = apply_block_mask(orig_mask, cfg.min_block, cfg.max_block, cfg.block_mask_prob)
        rand = (torch.rand_like(orig_mask) < cfg.masking_prob).float()
        m = m * (1.0 - rand) + (orig_mask - orig_mask * (1.0 - rand))
        return m * orig_mask
    rand = (torch.rand_like(orig_mask) < cfg.masking_prob).float()
    return orig_mask * (1.0 - rand)

# PyTorch Dataset returning padded time-series windows
class CitySeqDataset(Dataset):
    def __init__(self, values, mask, meta, seq_len, windows):
        self.seq_len = seq_len
        self.values = values
        self.mask = mask
        self.meta = meta
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s, length, is_padded = self.windows[idx]
        e = s + length
        v = np.zeros((self.seq_len, self.values.shape[1]), dtype=np.float32)
        m = np.zeros((self.seq_len, self.values.shape[1]), dtype=np.float32)
        meta = np.zeros((self.seq_len, self.meta.shape[1]), dtype=np.float32)
        pad_mask = np.zeros((self.seq_len,), dtype=np.float32)

        v[:length] = self.values[s:e]
        m[:length] = self.mask[s:e]
        meta[:length] = self.meta[s:e]
        pad_mask[:length] = 1.0

        return {
            "start": s,
            "length": length,
            "values": torch.tensor(v, dtype=torch.float32),
            "mask": torch.tensor(m, dtype=torch.float32),
            "meta": torch.tensor(meta, dtype=torch.float32),
            "pad_mask": torch.tensor(pad_mask, dtype=torch.float32),
        }

# Load raw dataset, preprocess dates, encode cities, and standardize pollutants
def load_and_prepare(data_csv):
    raw = pd.read_csv(data_csv)
    raw["_original_index"] = np.arange(len(raw))
    raw = parse_dates(raw)

    for p in POLLUTANTS:
        if p not in raw.columns:
            raise ValueError(f"Missing pollutant column: {p}")

    raw = raw.sort_values(["City", "_date_parsed"]).reset_index(drop=True)
    raw["_sorted_index"] = np.arange(len(raw))

    le_city = LabelEncoder()
    raw["city_idx"] = le_city.fit_transform(raw["City"].astype(str))
    num_cities = raw["city_idx"].nunique()

    scalers = {}
    for p in POLLUTANTS:
        scaler = StandardScaler()
        vals = raw[p].values.reshape(-1, 1)
        obs = ~np.isnan(vals[:, 0])
        if obs.sum() > 0:
            scaler.fit(vals[obs])
            raw.loc[obs, p] = scaler.transform(vals[obs]).flatten()
        else:
            scaler.mean_, scaler.scale_ = np.array([0.0]), np.array([1.0])
        scalers[p] = scaler

    values = raw[POLLUTANTS].values.astype(float)
    mask = (~np.isnan(values)).astype(float)
    values_filled = np.where(np.isnan(values), 0.0, values)
    meta_vals = raw[["doy_sin", "doy_cos", "month_sin", "month_cos", "city_idx", "day_of_year"]].values.astype(float)

    return raw, values_filled, mask, meta_vals, scalers, num_cities

# Prepare imputed dataset for downstream regression and classification models
def prepare_downstream_dataset(imputed_csv):
    df = pd.read_csv(imputed_csv)
    drop_cols = [c for c in ["_date_parsed", "year", "day_of_year", "doy_sin", "doy_cos",
                             "month", "month_sin", "month_cos", "city_idx"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    missing_aqi = df["AQI"].isna()
    computed = df.apply(calc_aqi_row, axis=1)
    df.loc[missing_aqi, "AQI"] = computed[missing_aqi]
    df.loc[missing_aqi, "AQI_Bucket"] = computed[missing_aqi].apply(get_aqi_bucket)

    df = df.dropna(subset=POLLUTANTS + ["AQI", "AQI_Bucket"]).reset_index(drop=True)
    return df
