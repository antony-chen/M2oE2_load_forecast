"""
train_load_only.py
------------------
Trains M2oE2 using only historical load data (no external weather features).
Encoder looks back 2 weeks; decoder forecasts 1 week ahead.
Peak load accuracy is prioritized via peak fidelity losses.

Usage
-----
    python train_load_only.py --csv path/to/data.csv --feeders F1 F2 F3

CSV format (flexible column detection):
  - Timestamp column  : any name starting with "date", "time", or "datetime"
  - Feeder/ID column  : any name containing "feeder", "xfmr", "id", or "transformer"
  - Load column       : any name containing "kwh", "kw", "load", or "power"

All column detection is printed at startup for easy debugging.

Debugging / feature tuning
--------------------------
Edit the FEATURES section (~line 80) to add or remove engineered features.
Each feature is a named function: feature_name -> np.ndarray [n_hours].
Set DEBUG = True to print tensor shapes, sample values, and scaler ranges.
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

# ---- import model from same directory ----
sys.path.insert(0, os.path.dirname(__file__))
from model_v2 import VariationalSeq2Seq_meta

# ===========================================================================
# DEBUG FLAG — set True to print shapes, sample values, and scaler ranges
# ===========================================================================
DEBUG = True

# ===========================================================================
# HYPERPARAMETERS — edit freely
# ===========================================================================
ENCODER_WEEKS    = 2        # how many weeks the encoder looks back
DECODER_WEEKS    = 1        # forecast horizon in weeks
OUTPUT_LEN       = 24       # hours predicted per decoder step
NUM_IN_WEEK      = 168      # hours per week (do not change)

TOTAL_EPOCHS     = 1000
PEAK_WARMUP_EPOCHS = 150    # ramp peak losses in over this many epochs
BATCH_SIZE       = 16
LR               = 1e-3
KL_WEIGHT        = 0.001

# Model dimensions
XPRIME_DIM  = 40
HIDDEN_DIM  = 64
LATENT_DIM  = 32
NUM_LAYERS  = 4
TOP_K       = 2             # expert top-k gating sparsity
WARMUP_EP   = 10            # expert gating warmup epochs

# Peak loss weights (active after PEAK_WARMUP_EPOCHS)
LAM_THR     = 0.05          # soft-threshold region MSE
LAM_Q       = 0.04          # pinball loss on upper quantile
LAM_TIME    = 0.01          # peak timing loss
LAM_AMP     = 0.03          # peak amplitude loss
LAM_TOPK    = 0.01          # top-k hours MSE
TOPK_K      = 8             # number of top hours in topk loss
THR_FRAC    = 0.85          # fraction of max defining "peak region"
TAU         = 0.05          # softness of peak threshold
Q_UPPER     = 0.90          # quantile for pinball peak loss
SOFTARG_T   = 0.12          # temperature for softargmax peak timing

TRAIN_RATIO  = 0.7
GRAD_CLIP    = 1.0
LOGVAR_MIN   = -10.0
LOGVAR_MAX   = 5.0
WEIGHT_DECAY = 1e-4

# ===========================================================================
# FEATURES — add or remove engineered features here
#
# Each entry is (name, function).
# The function receives the full 1-D load array (all hours, already
# interpolated) and returns a 1-D array of the same length.
#
# To add a new feature, append:
#   ("my_feature", lambda load: <your computation>)
#
# To disable a feature, comment out its line.
# ===========================================================================
def _wow_ratio(load):
    """Week-over-week load ratio: load[t] / load[t-168], clipped to [0.1, 10]."""
    shifted = np.concatenate([np.ones(NUM_IN_WEEK), load[:-NUM_IN_WEEK]])
    return np.clip(load / (shifted + 1e-6), 0.1, 10.0)

def _daily_peak_ratio(load):
    """Load normalized by that day's peak: separates curve shape from magnitude."""
    out = np.ones_like(load)
    for d in range(len(load) // 24):
        sl = slice(d * 24, (d + 1) * 24)
        peak = load[sl].max()
        if peak > 1e-6:
            out[sl] = load[sl] / peak
    return out

def _hour_of_week(load):
    """Cyclic hour-of-week encoding (sin component), range [-1, 1]."""
    t = np.arange(len(load))
    return np.sin(2 * math.pi * (t % NUM_IN_WEEK) / NUM_IN_WEEK)

ENGINEERED_FEATURES = [
    # name               function
    ("wow_ratio",        _wow_ratio),        # week-over-week ratio
    ("daily_peak_ratio", _daily_peak_ratio), # within-day shape
    ("hour_of_week_sin", _hour_of_week),     # cyclic time encoding
]
# To go fully load-only with zero external features, set this to []:
# ENGINEERED_FEATURES = []


# ===========================================================================
# Column detection
# ===========================================================================
def _find_col(df, candidates, label):
    """Case-insensitive substring search across column names."""
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in col.lower():
                print(f"  [cols] {label} -> '{col}'")
                return col
    raise ValueError(
        f"Cannot find {label} column. Tried substrings: {candidates}.\n"
        f"Available columns: {df.columns.tolist()}"
    )


def load_csv(csv_path, feeder_names):
    """
    Load CSV, filter to requested feeders, return dict:
        feeder_name -> pd.Series (hourly load, DatetimeIndex)
    """
    print(f"\n[CSV] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"  [CSV] Shape: {df.shape}  |  Columns: {df.columns.tolist()}")

    time_col   = _find_col(df, ["date", "time", "datetime", "timestamp"], "timestamp")
    feeder_col = _find_col(df, ["feeder", "xfmr", "transformer", "id"], "feeder/ID")
    load_col   = _find_col(df, ["kwh", "kw", "load", "power", "energy"], "load")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    available = df[feeder_col].unique().tolist()
    print(f"  [CSV] Available feeders ({len(available)}): {available}")

    if feeder_names:
        missing = [f for f in feeder_names if f not in available]
        if missing:
            raise ValueError(f"Feeders not found in CSV: {missing}\nAvailable: {available}")
        df = df[df[feeder_col].isin(feeder_names)]
    else:
        print("  [CSV] No feeders specified — using all.")

    feeder_series = {}
    for name, grp in df.groupby(feeder_col):
        s = grp.set_index(time_col)[load_col].copy()
        s = s[~s.index.duplicated(keep="first")]
        hourly_idx = pd.date_range(s.index.min().floor("h"), s.index.max().ceil("h"), freq="h")
        s = s.reindex(hourly_idx).interpolate("linear").fillna(0.0)
        feeder_series[str(name)] = s
        print(f"  [CSV] Feeder '{name}': {len(s)} hours  "
              f"({s.index.min().date()} -> {s.index.max().date()})")

    return feeder_series


# ===========================================================================
# Feature engineering
# ===========================================================================
def build_features(load_1d):
    """
    Given a 1-D hourly load array, return a dict:
        feature_name -> np.ndarray [n_hours]
    The 'load' key is always present.
    Add / remove features by editing ENGINEERED_FEATURES above.
    """
    load_1d = np.asarray(load_1d, dtype=np.float32)
    features = {"load": load_1d}

    for name, fn in ENGINEERED_FEATURES:
        try:
            arr = fn(load_1d).astype(np.float32)
            assert len(arr) == len(load_1d), f"Feature '{name}' length mismatch"
            features[name] = arr
            if DEBUG:
                print(f"    [feat] '{name}': min={arr.min():.3f}  max={arr.max():.3f}  "
                      f"mean={arr.mean():.3f}")
        except Exception as e:
            print(f"  [WARN] Feature '{name}' failed: {e} — skipping.")

    return features


def reshape_weekly(feat_1d, num_in_week=168):
    """Reshape flat 1-D array to [n_complete_weeks, num_in_week]."""
    n = len(feat_1d) // num_in_week * num_in_week
    return feat_1d[:n].reshape(-1, num_in_week)


# ===========================================================================
# Sliding-window seq2seq data builder
# ===========================================================================
def build_seq2seq_tensors(feature_dict_weekly, encoder_weeks, decoder_weeks,
                          output_len, train_ratio, device):
    """
    Returns train_dict, test_dict, scalers.
    feature_dict_weekly: {name: ndarray [n_weeks, 168]}
    """
    n_weeks = feature_dict_weekly["load"].shape[0]
    need    = encoder_weeks + decoder_weeks
    if n_weeks < need:
        raise ValueError(f"Need at least {need} weeks of data, got {n_weeks}.")

    # Scale all features globally (fit on full dataset — no look-ahead in scaling)
    scalers   = {}
    processed = {}
    for k, arr in feature_dict_weekly.items():
        flat = arr.reshape(-1, 1).astype(np.float32)
        sc   = MinMaxScaler()
        sc.fit(flat)
        processed[k] = sc.transform(flat).reshape(arr.shape)
        scalers[k]   = sc
        if DEBUG:
            print(f"  [scaler] '{k}': data_min={sc.data_min_[0]:.4f}  "
                  f"data_max={sc.data_max_[0]:.4f}")

    ext_keys = [k for k in feature_dict_weekly if k != "load"]
    K_ext    = len(ext_keys)
    L        = decoder_weeks * NUM_IN_WEEK - output_len

    enc_len = encoder_weeks * NUM_IN_WEEK
    dec_len = decoder_weeks * NUM_IN_WEEK

    X_enc_l, X_enc_ext = [], []
    X_dec_l, X_dec_ext = [], []
    Y_target            = []

    last_start = n_weeks - encoder_weeks - decoder_weeks
    for w in range(last_start + 1):
        # slice in week units, then flatten to hours
        enc_load_w = processed["load"][w : w + encoder_weeks]           # [enc_weeks, 168]
        dec_load_w = processed["load"][w + encoder_weeks :
                                       w + encoder_weeks + decoder_weeks] # [dec_weeks, 168]

        enc_l    = enc_load_w.reshape(-1)        # [enc_len]
        dec_full = dec_load_w.reshape(-1)        # [dec_len]

        if K_ext > 0:
            enc_ext = np.stack(
                [processed[k][w : w + encoder_weeks].reshape(-1) for k in ext_keys],
                axis=-1)                         # [enc_len, K]
            dec_ext_flat = np.stack(
                [processed[k][w + encoder_weeks :
                               w + encoder_weeks + decoder_weeks].reshape(-1)[:L]
                 for k in ext_keys], axis=-1)    # [L, K]
        else:
            enc_ext      = np.empty((enc_len, 0), dtype=np.float32)
            dec_ext_flat = np.empty((L, 0),       dtype=np.float32)

        targets = np.stack([dec_full[i:i+output_len] for i in range(L+1)], axis=0)  # [L+1, output_len]

        X_enc_l.append(enc_l)
        X_enc_ext.append(enc_ext)
        X_dec_l.append(dec_full[:L])
        X_dec_ext.append(dec_ext_flat)
        Y_target.append(targets)

    def to_t(a):
        return torch.tensor(np.array(a), dtype=torch.float32).to(device)

    tensors = {
        "X_enc_l":   to_t(X_enc_l).unsqueeze(-1),   # [B, enc_len, 1]
        "X_enc_ext": to_t(X_enc_ext),                # [B, enc_len, K]
        "X_dec_l":   to_t(X_dec_l).unsqueeze(-1),    # [B, L, 1]
        "X_dec_ext": to_t(X_dec_ext),                # [B, L, K]
        "Y_target":  to_t(Y_target).unsqueeze(-1),   # [B, L+1, output_len, 1]
    }

    if DEBUG:
        print("\n  [tensors]")
        for k, v in tensors.items():
            print(f"    {k:15s}: {tuple(v.shape)}")

    B     = tensors["X_enc_l"].shape[0]
    split = int(train_ratio * B)
    train = {k: v[:split] for k, v in tensors.items()}
    test  = {k: v[split:]  for k, v in tensors.items()}
    print(f"\n  [split] train={split} samples  test={B - split} samples")
    return train, test, scalers


def make_loader(data_dict, batch_size, shuffle):
    ds = TensorDataset(
        data_dict["X_enc_l"], data_dict["X_enc_ext"],
        data_dict["X_dec_l"], data_dict["X_dec_ext"],
        data_dict["Y_target"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ===========================================================================
# Loss helpers (copied/adapted from main_M2oE2_Final)
# ===========================================================================
def gaussian_nll(mu, logvar, y, logvar_min=-10.0, logvar_max=5.0):
    logvar = logvar.clamp(logvar_min, logvar_max)
    return 0.5 * (logvar + math.log(2 * math.pi) + (y - mu) ** 2 / (logvar.exp() + 1e-12))


def soft_threshold_mask(y, thr_frac, tau):
    B    = y.size(0)
    ymax = y.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1)
    thr  = thr_frac * ymax
    return torch.sigmoid((y - thr) / (tau + 1e-12))


def softargmax_time(y, temp):
    B, T = y.shape
    idx  = torch.arange(T, device=y.device, dtype=y.dtype).view(1, T)
    p    = torch.softmax(y / (temp + 1e-12), dim=1)
    return (p * idx).sum(dim=1)


def peak_fidelity_loss(mu_preds, logvar_preds, tgt, *,
                       thr_frac, tau, q_upper, softarg_temp,
                       lam_thr, lam_q, lam_time, lam_amp, lam_topk, topk_k):
    """Combined peak-focused loss."""
    B = tgt.size(0)

    # Use first-horizon slice for peak losses [B, L+1]
    mu_fh  = mu_preds[:, :, 0]
    tgt_fh = tgt[:, :, 0]

    # 1) Soft-threshold MSE — penalize errors in the peak region
    w      = soft_threshold_mask(tgt_fh.unsqueeze(-1), thr_frac, tau).squeeze(-1)
    L_thr  = (w * (mu_fh - tgt_fh) ** 2).mean()

    # 2) Pinball loss on upper quantile
    err    = tgt_fh - mu_fh
    L_q    = torch.max(q_upper * err, (q_upper - 1) * err).mean()

    # 3) Peak timing loss (softargmax)
    t_true = softargmax_time(tgt_fh, softarg_temp)
    t_pred = softargmax_time(mu_fh,  softarg_temp)
    L_time = ((t_pred - t_true) ** 2).mean()

    # 4) Peak amplitude loss
    L_amp  = ((mu_fh.reshape(B, -1).max(dim=1).values -
               tgt_fh.reshape(B, -1).max(dim=1).values) ** 2).mean()

    # 5) Top-k hours MSE
    k      = min(topk_k, mu_fh.size(1))
    L_topk = ((torch.topk(mu_fh,  k, dim=1).values -
               torch.topk(tgt_fh, k, dim=1).values) ** 2).mean()

    return (lam_thr  * L_thr  +
            lam_q    * L_q    +
            lam_time * L_time +
            lam_amp  * L_amp  +
            lam_topk * L_topk)


# ===========================================================================
# Training loop
# ===========================================================================
def train(model, loader, total_epochs, lr, device, save_path,
          kl_weight, peak_warmup_epochs, grad_clip):

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best_loss = float("inf")
    best_epoch = -1

    for ep in range(1, total_epochs + 1):
        model.train()
        running = 0.0
        w_peak  = min(1.0, ep / max(1, peak_warmup_epochs))

        for enc_l, enc_ext, dec_l, dec_ext, tgt in loader:
            optimizer.zero_grad()

            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=TOP_K, warmup_epochs=WARMUP_EP,
            )

            # NLL loss
            nll  = gaussian_nll(mu_preds, logvar_preds, tgt,
                                 LOGVAR_MIN, LOGVAR_MAX).mean()

            # KL regularisation on latent space
            kl   = -0.5 * (1 + logvar_z - mu_z ** 2 - logvar_z.exp()).mean()

            # Peak fidelity losses (ramped in over peak_warmup_epochs)
            peak = peak_fidelity_loss(
                mu_preds, logvar_preds, tgt,
                thr_frac=THR_FRAC, tau=TAU, q_upper=Q_UPPER,
                softarg_temp=SOFTARG_T,
                lam_thr=LAM_THR   * w_peak,
                lam_q=LAM_Q       * w_peak,
                lam_time=LAM_TIME * w_peak,
                lam_amp=LAM_AMP   * w_peak,
                lam_topk=LAM_TOPK * w_peak,
                topk_k=TOPK_K,
            )

            loss = nll + kl_weight * kl + peak
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * enc_l.size(0)

        epoch_loss = running / len(loader.dataset)

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_epoch = ep
            torch.save(model.state_dict(), save_path)

        if ep % 50 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{total_epochs}  loss={epoch_loss:.5f}  "
                  f"best={best_loss:.5f} (ep {best_epoch})  "
                  f"peak_w={w_peak:.2f}")

    print(f"\n[✓] Best model saved to '{save_path}'  (epoch {best_epoch}, loss {best_loss:.5f})")
    return best_loss


# ===========================================================================
# Per-feeder pipeline
# ===========================================================================
def run_feeder(feeder_name, load_series, device):
    print(f"\n{'='*60}")
    print(f"  FEEDER: {feeder_name}")
    print(f"{'='*60}")

    load_1d = load_series.values.astype(np.float32)

    # --- build features ---
    print("\n[FEATURES]")
    feat_flat = build_features(load_1d)

    # --- reshape to weekly ---
    feat_weekly = {k: reshape_weekly(v) for k, v in feat_flat.items()}
    n_weeks = feat_weekly["load"].shape[0]
    print(f"\n  [data] {len(load_1d)} hours  ->  {n_weeks} complete weeks")

    # --- build tensors ---
    print("\n[TENSORS]")
    train_dict, test_dict, scalers = build_seq2seq_tensors(
        feat_weekly,
        encoder_weeks=ENCODER_WEEKS,
        decoder_weeks=DECODER_WEEKS,
        output_len=OUTPUT_LEN,
        train_ratio=TRAIN_RATIO,
        device=device,
    )

    train_loader = make_loader(train_dict, BATCH_SIZE, shuffle=True)
    n_ext        = train_dict["X_enc_ext"].shape[-1]

    # --- build model ---
    # With load-only, no experts are needed for external features —
    # the model falls back to a single expert path when n_externals=0.
    # If engineered features are present they get their own expert block.
    if n_ext == 0:
        thermal_indices = []
        workday_index   = None
        season_index    = None
    else:
        # All engineered features go to the thermal expert block
        thermal_indices = list(range(n_ext))
        workday_index   = None
        season_index    = None

    model = VariationalSeq2Seq_meta(
        xprime_dim      = XPRIME_DIM,
        input_dim       = 1,
        hidden_size     = HIDDEN_DIM,
        latent_size     = LATENT_DIM,
        output_len      = OUTPUT_LEN,
        n_externals     = n_ext,
        output_dim      = 1,
        num_layers      = NUM_LAYERS,
        dropout         = 0.1,
        thermal_indices = thermal_indices,
        workday_index   = workday_index,
        season_index    = season_index,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] Parameters: {n_params:,}  |  n_ext={n_ext}  "
          f"|  encoder={ENCODER_WEEKS}w  decoder={DECODER_WEEKS}w  output_len={OUTPUT_LEN}h")

    # --- train ---
    safe_name = feeder_name.replace("/", "_").replace(" ", "_")
    save_path = f"load_only_{safe_name}_best.pt"

    print(f"\n[TRAIN] {TOTAL_EPOCHS} epochs  batch={BATCH_SIZE}  lr={LR}  "
          f"kl={KL_WEIGHT}  peak_warmup={PEAK_WARMUP_EPOCHS}\n")
    train(model, train_loader, TOTAL_EPOCHS, LR, device,
          save_path, KL_WEIGHT, PEAK_WARMUP_EPOCHS, GRAD_CLIP)

    # --- quick evaluation on test set ---
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    sc_load = scalers["load"]

    mse_sum, n_samples = 0.0, 0
    peak_err_sum       = 0.0

    with torch.no_grad():
        test_loader = make_loader(test_dict, BATCH_SIZE, shuffle=False)
        for enc_l, enc_ext, dec_l, dec_ext, tgt in test_loader:
            mu_preds, _, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
            mu_fh  = mu_preds[:, :, 0].cpu().numpy()   # [B, L+1]
            tgt_fh = tgt[:, :, 0].cpu().numpy()

            # denormalize
            B = mu_fh.shape[0]
            mu_dn  = sc_load.inverse_transform(mu_fh.reshape(-1, 1)).reshape(B, -1)
            tgt_dn = sc_load.inverse_transform(tgt_fh.reshape(-1, 1)).reshape(B, -1)

            mse_sum      += ((mu_dn - tgt_dn) ** 2).mean(axis=1).sum()
            peak_err_sum += np.abs(mu_dn.max(axis=1) - tgt_dn.max(axis=1)).sum()
            n_samples    += B

    rmse      = math.sqrt(mse_sum / n_samples)
    mean_peak_err = peak_err_sum / n_samples
    print(f"\n[EVAL] Feeder '{feeder_name}'")
    print(f"  Test RMSE       : {rmse:.2f} kW")
    print(f"  Mean peak error : {mean_peak_err:.2f} kW")

    return {"feeder": feeder_name, "rmse": rmse, "mean_peak_error": mean_peak_err,
            "checkpoint": save_path}


# ===========================================================================
# Entry point
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train load-only M2oE2 model per feeder.")
    p.add_argument("--csv",      required=True,  help="Path to input CSV file")
    p.add_argument("--feeders",  nargs="*",       help="Feeder names to train on (default: all)")
    p.add_argument("--epochs",   type=int,        default=TOTAL_EPOCHS, help="Training epochs")
    p.add_argument("--no-debug", action="store_true", help="Suppress debug output")
    return p.parse_args()


def main():
    global DEBUG, TOTAL_EPOCHS

    args = parse_args()
    if args.no_debug:
        DEBUG = False
    TOTAL_EPOCHS = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder lookback : {ENCODER_WEEKS} weeks ({ENCODER_WEEKS * NUM_IN_WEEK} hours)")
    print(f"Decoder horizon  : {DECODER_WEEKS} week  ({DECODER_WEEKS * NUM_IN_WEEK} hours)")
    print(f"Output len       : {OUTPUT_LEN} hours per decoder step")
    print(f"Engineered features: {[n for n, _ in ENGINEERED_FEATURES]}")

    feeder_series = load_csv(args.csv, args.feeders)

    results = []
    for feeder_name, series in feeder_series.items():
        result = run_feeder(feeder_name, series, device)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['feeder']:30s}  RMSE={r['rmse']:.2f} kW  "
              f"PeakErr={r['mean_peak_error']:.2f} kW  -> {r['checkpoint']}")


if __name__ == "__main__":
    main()
