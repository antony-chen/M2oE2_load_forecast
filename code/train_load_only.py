"""
train_load_only.py
------------------
Trains M2oE2 using historical load + the same weather features as the main
script (temp, CDD, HDD, humidity/workday slot, heat-index/season slot, oracle
future-temp path). Encoder looks back 2 weeks; decoder forecasts 1 week ahead.
Peak load accuracy is prioritised via peak fidelity losses.

Usage
-----
    # Minimum — auto-detect all columns, train on all feeders
    python train_load_only.py --csv data.csv

    # Specify feeders and an explicit load column name
    python train_load_only.py --csv data.csv --feeders F1 F2 --load-col KWH

    # All column overrides
    python train_load_only.py --csv data.csv \\
        --load-col   KWH                       \\
        --time-col   DATEHRLWT                 \\
        --feeder-col XFMR                      \\
        --temp-col   SURDPOINTTEMPFAHRENHEIT   \\
        --humidity-col  RELATIVEHUMIDITY       \\
        --heatindex-col HEATINDEXFAHRENHEIT    \\
        --feeders F1 F2

Column auto-detection fallbacks (used when --*-col flags are omitted):
  Timestamp  : first column whose name contains "date", "time", or "datetime"
  Feeder/ID  : first column whose name contains "feeder", "xfmr", "id", or "transformer"
  Load       : first column whose name contains "kwh", "kw", "load", or "power"
  Temp       : first column whose name contains "temp" or "temperature"
  Humidity   : first column whose name contains "humid" or "relativehumid"
  Heat index : first column whose name contains "heatindex" or "heat_index"

All resolved column names are printed at startup for easy debugging.

Debugging / feature tuning
--------------------------
Set DEBUG = True (or omit --no-debug) to print tensor shapes, scaler ranges,
and per-feature statistics after engineering.

To add a custom engineered feature, append to EXTRA_FEATURES near line ~120:
    ("my_feature", lambda load_1d, feat_dict: <computation>, "thermal")
The third element routes the feature to an expert: "thermal", "workday", or "season".
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from model_v2 import VariationalSeq2Seq_meta

# ===========================================================================
# DEBUG FLAG
# ===========================================================================
DEBUG = True

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
ENCODER_WEEKS      = 1
DECODER_WEEKS      = 1
OUTPUT_LEN         = 24
NUM_IN_WEEK        = 168

TOTAL_EPOCHS       = 1000
PEAK_WARMUP_EPOCHS = 150
BATCH_SIZE         = 16
LR                 = 1e-3
KL_WEIGHT          = 0.001

XPRIME_DIM  = 40
HIDDEN_DIM  = 64
LATENT_DIM  = 32
NUM_LAYERS  = 4
TOP_K       = 2
WARMUP_EP   = 10

# Peak loss weights
LAM_THR   = 0.05
LAM_Q     = 0.04
LAM_TIME  = 0.01
LAM_AMP   = 0.03
LAM_TOPK  = 0.01
TOPK_K    = 8
THR_FRAC  = 0.85
TAU       = 0.05
Q_UPPER   = 0.90
SOFTARG_T = 0.12

COOL_BASE_F  = 72.0    # CDD cooling threshold (°F)
HEAT_BASE_F  = 60.0    # HDD heating threshold (°F)

TRAIN_RATIO  = 0.7
GRAD_CLIP    = 1.0
LOGVAR_MIN   = -10.0
LOGVAR_MAX   = 5.0
WEIGHT_DECAY = 1e-4

# ===========================================================================
# EXTRA ENGINEERED FEATURES
# ---------------------------------------------------------------------------
# Add custom features here. Each entry is a 3-tuple:
#   (name, function, expert_group)
#
# function signature:  fn(load_1d, feat_dict) -> np.ndarray [n_hours]
#   load_1d   : 1-D float32 load array (all hours)
#   feat_dict : dict of already-computed raw features (temp, humidity, etc.)
#               Keys present: "load", and any weather columns found in the CSV.
#
# expert_group: "thermal" | "workday" | "season"
#   Routes the feature to the same expert that handles that group.
#
# To disable a feature, comment out its line.
# ===========================================================================
def _wow_ratio(load_1d, _):
    """Week-over-week load ratio."""
    shifted = np.concatenate([np.ones(NUM_IN_WEEK), load_1d[:-NUM_IN_WEEK]])
    return np.clip(load_1d / (shifted + 1e-6), 0.1, 10.0).astype(np.float32)

def _daily_peak_ratio(load_1d, _):
    """Load normalised by that day's peak."""
    out = np.ones_like(load_1d)
    for d in range(len(load_1d) // 24):
        sl = slice(d * 24, (d + 1) * 24)
        peak = load_1d[sl].max()
        if peak > 1e-6:
            out[sl] = load_1d[sl] / peak
    return out.astype(np.float32)

EXTRA_FEATURES = [
    # name                  function            expert_group
    # ("wow_ratio",         _wow_ratio,         "thermal"),
    # ("daily_peak_ratio",  _daily_peak_ratio,  "thermal"),
]


# ===========================================================================
# Column detection helpers
# ===========================================================================
def _find_col(df, candidates, label, override=None):
    """Return override if given, else search df columns by substring."""
    if override:
        if override not in df.columns:
            raise ValueError(
                f"Column '{override}' not found for {label}.\n"
                f"Available: {df.columns.tolist()}"
            )
        print(f"  [cols] {label:15s} -> '{override}'  (explicit)")
        return override
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in col.lower():
                print(f"  [cols] {label:15s} -> '{col}'  (auto-detected from '{cand}')")
                return col
    print(f"  [cols] {label:15s} -> NOT FOUND  (tried: {candidates})")
    return None


# ===========================================================================
# CSV loading
# ===========================================================================
def load_csv(csv_path, feeder_names, col_overrides):
    """
    Returns dict: feeder_name -> pd.DataFrame (hourly, DatetimeIndex)
    with all numeric columns available for feature engineering.
    """
    print(f"\n[CSV] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"  [CSV] Shape: {df.shape}")
    print(f"  [CSV] Columns: {df.columns.tolist()}")

    time_col    = _find_col(df, ["date", "time", "datetime", "timestamp"],
                             "timestamp",   col_overrides.get("time"))
    feeder_col  = _find_col(df, ["feeder", "xfmr", "transformer", "id"],
                             "feeder/ID",   col_overrides.get("feeder"))
    load_col    = _find_col(df, ["kwh", "kw", "load", "power", "energy"],
                             "load",        col_overrides.get("load"))
    temp_col    = _find_col(df, ["temp", "temperature"],
                             "temperature", col_overrides.get("temp"))
    humid_col   = _find_col(df, ["relativehumid", "humid", "humidity"],
                             "humidity",    col_overrides.get("humidity"))
    heatidx_col = _find_col(df, ["heatindex", "heat_index"],
                             "heat index",  col_overrides.get("heatindex"))

    if time_col is None:
        raise ValueError("Cannot find a timestamp column — use --time-col to specify it.")
    if load_col is None:
        raise ValueError("Cannot find a load column — use --load-col to specify it.")
    if feeder_col is None:
        print("  [WARN] No feeder/ID column found — treating entire CSV as one feeder 'ALL'.")
        df["_feeder"] = "ALL"
        feeder_col = "_feeder"

    df[time_col]   = pd.to_datetime(df[time_col], errors="coerce")
    df[feeder_col] = df[feeder_col].astype(str).str.strip()   # always string, no whitespace
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df[load_col] = pd.to_numeric(df[load_col], errors="coerce").fillna(0.0)

    available = sorted(df[feeder_col].unique().tolist())
    print(f"\n  [CSV] Available feeders ({len(available)}): {available}")

    if feeder_names:
        feeder_names = [str(f).strip() for f in feeder_names]
        missing = [f for f in feeder_names if f not in available]
        if missing:
            raise ValueError(f"Feeders not found in CSV: {missing}\nAvailable: {available}")
        df = df[df[feeder_col].isin(feeder_names)]

    col_map = {
        "load":     load_col,
        "temp":     temp_col,
        "humidity": humid_col,
        "heatindex": heatidx_col,
    }

    feeder_data = {}
    for name, grp in df.groupby(feeder_col):
        grp = grp.sort_values(time_col).set_index(time_col)
        grp = grp[~grp.index.duplicated(keep="first")]
        hourly_idx = pd.date_range(grp.index.min().floor("h"),
                                   grp.index.max().ceil("h"), freq="h")
        grp = grp.reindex(hourly_idx)

        # interpolate all numeric columns
        grp = grp.select_dtypes(include=[np.number]).interpolate("linear").fillna(0.0)

        feeder_data[str(name)] = (grp, col_map)
        n_hours = len(grp)
        print(f"  [CSV] Feeder '{name}': {n_hours} hours  "
              f"({hourly_idx.min().date()} -> {hourly_idx.max().date()})")

    return feeder_data


# ===========================================================================
# Feature engineering — mirrors main_M2oE2_Final feature_dict_all
# ===========================================================================
def build_features(grp_df, col_map):
    """
    Build the same feature set as main_M2oE2_Final:
      Thermal expert : temp, cdd, hdd, temp_fc_tplus00 … temp_fc_tplus{OUTPUT_LEN-1}
      Workday slot   : humidity  (RELATIVEHUMIDITY in ONCOR)
      Season slot    : heat index (HEATINDEXFAHRENHEIT in ONCOR)
      + any EXTRA_FEATURES

    Returns:
      feat_1d  : dict  name -> np.ndarray [n_hours]
      expert_map : dict  name -> "thermal" | "workday" | "season"
    """
    load_col     = col_map["load"]
    temp_col     = col_map["temp"]
    humid_col    = col_map["humidity"]
    heatidx_col  = col_map["heatindex"]

    load_1d = grp_df[load_col].values.astype(np.float32)
    n       = len(load_1d)

    feat_1d    = {"load": load_1d}
    expert_map = {}                  # feature_name -> expert group

    # --- temperature ---
    if temp_col and temp_col in grp_df.columns:
        temp_1d = grp_df[temp_col].values.astype(np.float32)
        cdd     = np.maximum(temp_1d - COOL_BASE_F, 0.0).astype(np.float32)
        hdd     = np.maximum(HEAT_BASE_F - temp_1d, 0.0).astype(np.float32)

        feat_1d["temp"] = temp_1d
        feat_1d["cdd"]  = cdd
        feat_1d["hdd"]  = hdd
        expert_map["temp"] = "thermal"
        expert_map["cdd"]  = "thermal"
        expert_map["hdd"]  = "thermal"

        # oracle future-temp path: temp_fc_tplus00 … temp_fc_tplus{OUTPUT_LEN-1}
        for h in range(OUTPUT_LEN):
            shifted = np.concatenate([temp_1d[h:], np.full(h, temp_1d[-1])])
            key = f"temp_fc_tplus{h:02d}"
            feat_1d[key]    = shifted.astype(np.float32)
            expert_map[key] = "thermal"

        if DEBUG:
            print(f"    [feat] temp    : min={temp_1d.min():.1f}  max={temp_1d.max():.1f}")
            print(f"    [feat] cdd     : min={cdd.min():.2f}  max={cdd.max():.2f}")
            print(f"    [feat] hdd     : min={hdd.min():.2f}  max={hdd.max():.2f}")
            print(f"    [feat] oracle temp path: {OUTPUT_LEN} channels")
    else:
        print("  [WARN] No temperature column — thermal expert will receive no external features.")

    # --- humidity (workday slot) ---
    if humid_col and humid_col in grp_df.columns:
        humid_1d = grp_df[humid_col].values.astype(np.float32)
        feat_1d["humidity"]    = humid_1d
        expert_map["humidity"] = "workday"
        if DEBUG:
            print(f"    [feat] humidity: min={humid_1d.min():.2f}  max={humid_1d.max():.2f}")
    else:
        print("  [WARN] No humidity column — workday expert slot will be empty.")

    # --- heat index (season slot) ---
    if heatidx_col and heatidx_col in grp_df.columns:
        hi_1d = grp_df[heatidx_col].values.astype(np.float32)
        feat_1d["heatindex"]    = hi_1d
        expert_map["heatindex"] = "season"
        if DEBUG:
            print(f"    [feat] heatindex: min={hi_1d.min():.2f}  max={hi_1d.max():.2f}")
    else:
        print("  [WARN] No heat-index column — season expert slot will be empty.")

    # --- extra engineered features ---
    for name, fn, group in EXTRA_FEATURES:
        try:
            arr = fn(load_1d, feat_1d).astype(np.float32)
            assert len(arr) == n, f"Length mismatch for '{name}'"
            feat_1d[name]    = arr
            expert_map[name] = group
            if DEBUG:
                print(f"    [feat] {name:20s}: min={arr.min():.3f}  max={arr.max():.3f}  "
                      f"expert='{group}'")
        except Exception as e:
            print(f"  [WARN] Extra feature '{name}' failed: {e} — skipping.")

    return feat_1d, expert_map


def reshape_weekly(arr_1d):
    n = len(arr_1d) // NUM_IN_WEEK * NUM_IN_WEEK
    return arr_1d[:n].reshape(-1, NUM_IN_WEEK)


# ===========================================================================
# Seq2seq tensor builder
# ===========================================================================
def build_seq2seq_tensors(feat_1d, expert_map, encoder_weeks, decoder_weeks,
                          output_len, train_ratio, device):
    """
    Scales all features, builds sliding-window encoder/decoder tensors,
    and returns train_dict, test_dict, scalers, and expert index mappings.
    """
    # reshape to weekly
    feat_weekly = {k: reshape_weekly(v) for k, v in feat_1d.items()}
    n_weeks     = feat_weekly["load"].shape[0]
    need        = encoder_weeks + decoder_weeks
    if n_weeks < need:
        raise ValueError(f"Need >= {need} weeks of data, got {n_weeks}.")

    print(f"  [data] {len(feat_1d['load'])} hours  ->  {n_weeks} complete weeks")

    # scale
    scalers, processed = {}, {}
    for k, arr in feat_weekly.items():
        flat = arr.reshape(-1, 1).astype(np.float32)
        sc   = MinMaxScaler()
        sc.fit(flat)
        processed[k] = sc.transform(flat).reshape(arr.shape)
        scalers[k]   = sc
        if DEBUG:
            print(f"  [scaler] {k:25s}: [{sc.data_min_[0]:.4f}, {sc.data_max_[0]:.4f}]")

    ext_keys = [k for k in feat_weekly if k != "load"]
    K_ext    = len(ext_keys)
    ext_idx  = {k: i for i, k in enumerate(ext_keys)}
    L        = decoder_weeks * NUM_IN_WEEK - output_len

    # build expert index lists for the model constructor
    thermal_indices = [ext_idx[k] for k in ext_keys if expert_map.get(k) == "thermal"]
    workday_keys    = [k for k in ext_keys if expert_map.get(k) == "workday"]
    season_keys     = [k for k in ext_keys if expert_map.get(k) == "season"]
    workday_index   = ext_idx[workday_keys[0]] if workday_keys else None
    season_index    = ext_idx[season_keys[0]]  if season_keys  else None

    if DEBUG:
        print(f"\n  [expert routing]")
        print(f"    thermal_indices ({len(thermal_indices)}): {thermal_indices}")
        print(f"    workday_index : {workday_index}  ({workday_keys})")
        print(f"    season_index  : {season_index}   ({season_keys})")

    # sliding window
    X_enc_l, X_enc_ext = [], []
    X_dec_l, X_dec_ext = [], []
    Y_target            = []

    for w in range(n_weeks - encoder_weeks - decoder_weeks + 1):
        enc_l    = processed["load"][w : w + encoder_weeks].reshape(-1)
        dec_full = processed["load"][w + encoder_weeks :
                                     w + encoder_weeks + decoder_weeks].reshape(-1)

        enc_ext = (np.stack([processed[k][w : w + encoder_weeks].reshape(-1)
                              for k in ext_keys], axis=-1)
                   if K_ext > 0 else np.empty((encoder_weeks * NUM_IN_WEEK, 0), np.float32))
        dec_ext = (np.stack([processed[k][w + encoder_weeks :
                                          w + encoder_weeks + decoder_weeks].reshape(-1)[:L]
                              for k in ext_keys], axis=-1)
                   if K_ext > 0 else np.empty((L, 0), np.float32))

        targets = np.stack([dec_full[i:i+output_len] for i in range(L+1)], axis=0)

        X_enc_l.append(enc_l);   X_enc_ext.append(enc_ext)
        X_dec_l.append(dec_full[:L]); X_dec_ext.append(dec_ext)
        Y_target.append(targets)

    def to_t(a):
        return torch.tensor(np.array(a), dtype=torch.float32).to(device)

    tensors = {
        "X_enc_l":   to_t(X_enc_l).unsqueeze(-1),
        "X_enc_ext": to_t(X_enc_ext),
        "X_dec_l":   to_t(X_dec_l).unsqueeze(-1),
        "X_dec_ext": to_t(X_dec_ext),
        "Y_target":  to_t(Y_target).unsqueeze(-1),
    }

    if DEBUG:
        print("\n  [tensors]")
        for k, v in tensors.items():
            print(f"    {k:15s}: {tuple(v.shape)}")

    B     = tensors["X_enc_l"].shape[0]
    split = int(train_ratio * B)
    train = {k: v[:split] for k, v in tensors.items()}
    test  = {k: v[split:]  for k, v in tensors.items()}
    print(f"\n  [split] train={split}  test={B - split}  (ratio={train_ratio})")

    return train, test, scalers, K_ext, thermal_indices, workday_index, season_index


def make_loader(d, batch_size, shuffle):
    ds = TensorDataset(d["X_enc_l"], d["X_enc_ext"],
                       d["X_dec_l"], d["X_dec_ext"], d["Y_target"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ===========================================================================
# Loss helpers — match v7_peak_fidelity_loss in main_M2oE2_Final exactly
# ===========================================================================
def gaussian_nll(mu, logvar, y):
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    return 0.5 * (logvar + math.log(2 * math.pi) + (y - mu) ** 2 / (logvar.exp() + 1e-12))


def gaussian_icdf(p, device):
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        2 * torch.as_tensor(p, device=device) - 1
    )


def pinball_loss(y, yq, q):
    e = y - yq
    return torch.where(e >= 0, q * e, (q - 1) * e)


def soft_threshold_mask(y, thr_frac, tau):
    """Peak-region weight: sigmoid ramp above thr_frac * max(y)."""
    B    = y.size(0)
    ymax = y.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1)
    return torch.sigmoid((y - thr_frac * ymax) / (tau + 1e-12))


def softargmax_time(y, temp):
    B, T = y.shape
    idx  = torch.arange(T, device=y.device, dtype=y.dtype).view(1, T)
    return (torch.softmax(y / (temp + 1e-12), dim=1) * idx).sum(dim=1)


def peak_fidelity_loss(mu_preds, logvar_preds, tgt, w_peak):
    """
    Mirrors v7_peak_fidelity_loss from main_M2oE2_Final:
      - L_thr : weighted MSE, normalised by sum of peak weights (not plain mean)
      - L_q   : Gaussian quantile pinball, peak-weighted
      - L_time: absolute peak-timing error, normalised by sequence length
      - L_amp : peak amplitude MSE across full decoder
      - L_topk: top-k hours MSE across full decoder
    NLL is computed and added separately in the training loop.
    """
    mu   = mu_preds.squeeze(-1)     # [B, L+1, output_len]
    y    = tgt.squeeze(-1)
    logv = logvar_preds.squeeze(-1).clamp(LOGVAR_MIN, LOGVAR_MAX)
    sigma = (0.5 * logv).exp()

    w = soft_threshold_mask(y, THR_FRAC, TAU)          # [B, L+1, output_len]

    # 1) Weighted threshold MSE (normalised by weight sum, not count)
    err2  = (mu - y).pow(2)
    L_thr = (w * err2).sum() / (w.sum() + 1e-12)

    # 2) Gaussian quantile pinball, peak-weighted
    zq    = gaussian_icdf(Q_UPPER, device=mu.device)
    yq    = mu + zq * sigma
    pl    = pinball_loss(y, yq, Q_UPPER)
    L_q   = (w * pl).sum() / (w.sum() + 1e-12)

    # 3) Peak timing: absolute error normalised by sequence length
    B, L1, out = y.shape
    y_flat  = y.reshape(B, -1)
    mu_flat = mu.reshape(B, -1)
    T       = y_flat.size(1)
    L_time  = (softargmax_time(mu_flat, SOFTARG_T) -
               softargmax_time(y_flat,  SOFTARG_T)).abs().mean() / (T + 1e-12)

    # 4) Peak amplitude MSE (full decoder)
    L_amp  = ((mu_flat.max(dim=1).values - y_flat.max(dim=1).values) ** 2).mean()

    # 5) Top-k hours MSE (full decoder)
    k      = min(TOPK_K, mu_flat.size(1))
    L_topk = ((torch.topk(mu_flat, k, dim=1).values -
               torch.topk(y_flat,  k, dim=1).values) ** 2).mean()

    return w_peak * (LAM_THR * L_thr + LAM_Q * L_q +
                     LAM_TIME * L_time + LAM_AMP * L_amp + LAM_TOPK * L_topk), {
        "thr":  float(L_thr.detach().cpu()),
        "q":    float(L_q.detach().cpu()),
        "time": float(L_time.detach().cpu()),
        "amp":  float(L_amp.detach().cpu()),
        "topk": float(L_topk.detach().cpu()),
    }


# ===========================================================================
# Training loop
# ===========================================================================
def train_model(model, loader, total_epochs, device, save_path):
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss  = float("inf")
    best_epoch = -1

    for ep in range(1, total_epochs + 1):
        model.train()
        running  = 0.0
        skipped  = 0
        sum_parts = {"nll": 0.0, "thr": 0.0, "q": 0.0, "time": 0.0,
                     "amp": 0.0, "topk": 0.0, "kl": 0.0}
        cnt_parts = 0
        w_peak = min(1.0, ep / max(1, PEAK_WARMUP_EPOCHS))
        lam_thr_ep  = LAM_THR  * w_peak
        lam_q_ep    = LAM_Q    * w_peak
        lam_time_ep = LAM_TIME * w_peak
        lam_amp_ep  = LAM_AMP  * w_peak
        lam_topk_ep = LAM_TOPK * w_peak

        for enc_l, enc_ext, dec_l, dec_ext, tgt in loader:
            optimizer.zero_grad()
            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=TOP_K, warmup_epochs=WARMUP_EP,
            )
            nll        = gaussian_nll(mu_preds, logvar_preds, tgt).mean()
            kl         = -0.5 * (1 + logvar_z - mu_z ** 2 - logvar_z.exp()).mean()
            peak, parts = peak_fidelity_loss(mu_preds, logvar_preds, tgt, w_peak)
            loss       = nll + KL_WEIGHT * kl + peak

            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item() * enc_l.size(0)

            sum_parts["nll"]  += float(nll.detach().cpu())
            sum_parts["kl"]   += float(kl.detach().cpu())
            for k, v in parts.items():
                sum_parts[k] += v
            cnt_parts += 1

        denom      = max(1, len(loader.dataset) - skipped * loader.batch_size)
        epoch_loss = running / denom

        can_save = (ep >= PEAK_WARMUP_EPOCHS)
        if can_save and epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, ep
            torch.save(model.state_dict(), save_path)

        if ep == 1 or ep % 5 == 0 or ep == total_epochs:
            best_str = f"{best_loss:.6f} (ep {best_epoch})" if best_epoch >= 0 \
                       else "N/A (warmup phase)"
            print(
                f"Epoch {ep:4d}/{total_epochs} | loss={epoch_loss:.6f} | best={best_str} | "
                f"lam_thr={lam_thr_ep:.3f} lam_q={lam_q_ep:.3f} "
                f"lam_time={lam_time_ep:.3f} lam_amp={lam_amp_ep:.3f} "
                f"lam_topk={lam_topk_ep:.3f} | skipped={skipped}"
            )
            if cnt_parts > 0:
                p = {k: v / cnt_parts for k, v in sum_parts.items()}
                print(
                    f"  [parts] nll={p['nll']:.4f} thr={p['thr']:.4f} q={p['q']:.4f} "
                    f"time={p['time']:.4f} amp={p['amp']:.4f} topk={p['topk']:.4f} "
                    f"kl={p['kl']:.4f} | weighted: "
                    f"+{lam_thr_ep*p['thr']:.4f} +{lam_q_ep*p['q']:.4f} "
                    f"+{lam_time_ep*p['time']:.4f} +{lam_amp_ep*p['amp']:.4f} "
                    f"+{lam_topk_ep*p['topk']:.4f} +{KL_WEIGHT*p['kl']:.4f}"
                )

    print(f"\n[✓] Best model saved: '{save_path}'  (epoch {best_epoch}, loss {best_loss:.6f})")
    return best_loss


# ===========================================================================
# Per-feeder pipeline
# ===========================================================================
def run_feeder(feeder_name, grp_df, col_map, device):
    print(f"\n{'='*62}")
    print(f"  FEEDER: {feeder_name}")
    print(f"{'='*62}")

    print("\n[FEATURES]")
    feat_1d, expert_map = build_features(grp_df, col_map)

    print("\n[TENSORS]")
    train_dict, test_dict, scalers, K_ext, thermal_indices, workday_index, season_index = \
        build_seq2seq_tensors(feat_1d, expert_map,
                              ENCODER_WEEKS, DECODER_WEEKS, OUTPUT_LEN,
                              TRAIN_RATIO, device)

    train_loader = make_loader(train_dict, BATCH_SIZE, shuffle=True)

    model = VariationalSeq2Seq_meta(
        xprime_dim      = XPRIME_DIM,
        input_dim       = 1,
        hidden_size     = HIDDEN_DIM,
        latent_size     = LATENT_DIM,
        output_len      = OUTPUT_LEN,
        n_externals     = K_ext,
        output_dim      = 1,
        num_layers      = NUM_LAYERS,
        dropout         = 0.1,
        thermal_indices = thermal_indices,
        workday_index   = workday_index,
        season_index    = season_index,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] {n_params:,} params  |  K_ext={K_ext}  "
          f"encoder={ENCODER_WEEKS}w  decoder={DECODER_WEEKS}w  output_len={OUTPUT_LEN}h")

    safe_name = feeder_name.replace("/", "_").replace(" ", "_")
    save_path = f"load_only_{safe_name}_best.pt"

    print(f"\n[TRAIN] {TOTAL_EPOCHS} epochs  batch={BATCH_SIZE}  lr={LR}  "
          f"kl={KL_WEIGHT}  peak_warmup={PEAK_WARMUP_EPOCHS}\n")
    train_model(model, train_loader, TOTAL_EPOCHS, device, save_path)

    # --- evaluation ---
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    sc = scalers["load"]
    mse_sum = peak_err_sum = 0.0
    n_total = 0

    with torch.no_grad():
        for enc_l, enc_ext, dec_l, dec_ext, tgt in make_loader(test_dict, BATCH_SIZE, False):
            mu_preds, _, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
            mu_fh  = mu_preds[:, :, 0].cpu().numpy()
            tgt_fh = tgt[:, :, 0].cpu().numpy()
            B = mu_fh.shape[0]
            mu_dn  = sc.inverse_transform(mu_fh.reshape(-1, 1)).reshape(B, -1)
            tgt_dn = sc.inverse_transform(tgt_fh.reshape(-1, 1)).reshape(B, -1)
            mse_sum      += ((mu_dn - tgt_dn) ** 2).mean(axis=1).sum()
            peak_err_sum += np.abs(mu_dn.max(axis=1) - tgt_dn.max(axis=1)).sum()
            n_total      += B

    rmse      = math.sqrt(mse_sum / max(1, n_total))
    peak_err  = peak_err_sum / max(1, n_total)
    print(f"\n[EVAL] '{feeder_name}'  RMSE={rmse:.2f}  Mean peak error={peak_err:.2f}")

    return {"feeder": feeder_name, "rmse": rmse, "peak_error": peak_err, "checkpoint": save_path}


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Train load-only M2oE2 with original weather features per feeder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",          required=True,  help="Path to input CSV")
    p.add_argument("--feeders",      nargs="*",       help="Feeder names (default: all)")
    p.add_argument("--epochs",       type=int, default=TOTAL_EPOCHS)

    # explicit column overrides
    p.add_argument("--load-col",     default=None, help="Load column name  (e.g. KWH)")
    p.add_argument("--time-col",     default=None, help="Timestamp column  (e.g. DATEHRLWT)")
    p.add_argument("--feeder-col",   default=None, help="Feeder/ID column  (e.g. XFMR)")
    p.add_argument("--temp-col",     default=None, help="Temperature column")
    p.add_argument("--humidity-col", default=None, help="Humidity column    (workday expert slot)")
    p.add_argument("--heatindex-col",default=None, help="Heat-index column  (season expert slot)")

    p.add_argument("--no-debug",     action="store_true", help="Suppress debug output")
    return p.parse_args()


def main():
    global DEBUG, TOTAL_EPOCHS
    args = parse_args()
    if args.no_debug:
        DEBUG = False
    TOTAL_EPOCHS = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")
    print(f"Encoder lookback : {ENCODER_WEEKS} weeks")
    print(f"Decoder horizon  : {DECODER_WEEKS} week")
    print(f"Output len       : {OUTPUT_LEN} hours/step")

    col_overrides = {
        "load":      args.load_col,
        "time":      args.time_col,
        "feeder":    args.feeder_col,
        "temp":      args.temp_col,
        "humidity":  args.humidity_col,
        "heatindex": args.heatindex_col,
    }
    # remove None entries so _find_col falls through to auto-detection
    col_overrides = {k: v for k, v in col_overrides.items() if v is not None}

    feeder_data = load_csv(args.csv, args.feeders, col_overrides)

    results = []
    for feeder_name, (grp_df, col_map) in feeder_data.items():
        result = run_feeder(feeder_name, grp_df, col_map, device)
        results.append(result)

    print(f"\n{'='*62}")
    print("SUMMARY")
    print(f"{'='*62}")
    for r in results:
        print(f"  {r['feeder']:30s}  RMSE={r['rmse']:.2f}  "
              f"PeakErr={r['peak_error']:.2f}  -> {r['checkpoint']}")


if __name__ == "__main__":
    main()
