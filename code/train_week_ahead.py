"""
M2OE2 week-ahead load forecasting — training.

Designed to run from a Jupyter notebook.  Set the CONFIG variables in the
section near the bottom of this file, then call run() in a notebook cell.

KEY DIFFERENCE from original training
    The decoder is trained with PRIOR-WEEK LOAD as its input instead of
    teacher-forced future load.  Training and inference use exactly the
    same decoder input, so predict_week_ahead.py (ALPHA=0) runs in the
    conditions the model was actually trained under.

HOW TO USE
    1. Set paths and options in the CONFIG section near the bottom.
    2. Call run() from a notebook cell (or execute the whole file as a cell).
    3. Copy the three output paths printed at the end into predict_week_ahead.py:
           CHECKPOINT_PATH  = "checkpoint_best_<tag>.pt"
           SCALER_META_PATH = "vae_base_scaler_meta_<tag>.json"
           TRAIN_CFG_PATH   = "train_config_<tag>.json"
           ALPHA            = 0.0   # prior-week mode matches training

INPUT CSV FORMAT
    A single CSV file covering all feeders (or one feeder), with columns:
        TIME                     hourly timestamp (any format pd.to_datetime accepts)
        KWH                      load (kWh)
        SURDPOINTTEMPFAHRENHEIT  dew-point temperature (°F)
        RELATIVEHUMIDITY         relative humidity (%)
        HEATINDEXFAHRENHEIT      heat index (°F)
        FEEDER  (optional)       feeder / transformer ID;
                                 if absent, the whole file is one feeder

RECOMMENDATIONS
───────────────
Epochs (TOTAL_EPOCHS)
    600 is a safe default — the model converges well and the peak-loss
    warm-up (150 epochs) finishes with plenty of room to keep improving.
    Use 1000 for best results; the script saves the best checkpoint
    automatically so you can stop early without losing progress.

Number of feeders
    The model learns the general load/weather relationship across all
    feeders, so more is better up to a point of diminishing returns.

    Minimum viable   :  5 feeders, ≥ 1 year of data each
    Recommended      : 15–30 feeders, 1–2 years each
    Overkill         : 50+ feeders (still fine, just slower to process)

    Rule of thumb — aim for at least 500 training samples total.
    Each feeder with W weeks of data contributes (W − 1) training pairs.
        20 feeders × 52 weeks =  1 020 pairs  →  strong
        10 feeders × 52 weeks =    510 pairs  →  solid
         5 feeders × 52 weeks =    255 pairs  →  borderline

Weeks per feeder
    Minimum :  ~10 weeks (9 samples per feeder)
    Good    :  52 weeks (1 full year — captures all seasonal variation)
    Best    : 104 weeks (2 years — model sees repeated seasons)
"""

import os
import sys
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from model_v2 import VariationalSeq2Seq_meta


NUM_IN_WEEK = 168

COL_TIME      = "TIME"
COL_LOAD      = "KWH"
COL_TEMP      = "SURDPOINTTEMPFAHRENHEIT"
COL_HUMIDITY  = "RELATIVEHUMIDITY"
COL_HEATINDEX = "HEATINDEXFAHRENHEIT"


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Loss functions ────────────────────────────────────────────────────────────

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def gaussian_icdf(p, device):
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        2 * torch.as_tensor(p, device=device) - 1
    )


def pinball_loss(y, yq, q):
    e = y - yq
    return torch.where(e >= 0, q * e, (q - 1) * e)


def gaussian_nll_pointwise(mu, logvar, y, logvar_min=-10.0, logvar_max=5.0):
    logvar = logvar.clamp(min=logvar_min, max=logvar_max)
    nll = 0.5 * (logvar + math.log(2.0 * math.pi) + (y - mu) ** 2 / (logvar.exp() + 1e-12))
    return nll, logvar


def softargmax_time(y, temp: float):
    idx = torch.arange(y.size(1), device=y.device, dtype=y.dtype).view(1, -1)
    p = torch.softmax(y / (temp + 1e-12), dim=1)
    return (p * idx).sum(dim=1)


def peak_fidelity_loss(
    mu_preds, logvar_preds, tgt, *,
    thr_frac=0.85, tau=0.05, q_upper=0.90, softarg_temp=0.12,
    lam_thr=0.05, lam_q=0.04, lam_time=0.01, lam_amp=0.03, lam_topk=0.01,
    load_scale=1.0, logvar_min=-10.0, logvar_max=5.0, topk_k=8,
    return_parts=False,
):
    mu   = mu_preds.squeeze(-1)
    y    = tgt.squeeze(-1)
    logv = logvar_preds.squeeze(-1)

    nll, logv_c = gaussian_nll_pointwise(mu, logv, y, logvar_min, logvar_max)
    sigma = (0.5 * logv_c).exp()

    # Soft mask that up-weights near-peak hours
    B = y.size(0)
    y_max  = y.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1)
    w_peak = torch.sigmoid((y - thr_frac * y_max) / (tau + 1e-12))

    err2  = (mu - y).pow(2)
    L_thr = (w_peak * err2).sum() / (w_peak.sum() + 1e-12) * (load_scale ** 2)

    zq  = gaussian_icdf(q_upper, device=mu.device)
    L_q = (w_peak * pinball_loss(y, mu + zq * sigma, q_upper)).sum() / (w_peak.sum() + 1e-12) * load_scale

    y_f  = y.reshape(B, -1)
    mu_f = mu.reshape(B, -1)
    L_time = (softargmax_time(mu_f, softarg_temp) - softargmax_time(y_f, softarg_temp)).abs().mean() / (y_f.size(1) + 1e-12)

    mu_peak  = mu_f.max(dim=1).values
    y_peak   = y_f.max(dim=1).values
    L_amp    = ((mu_peak - y_peak) ** 2).mean() * (load_scale ** 2)

    k = min(topk_k, mu_f.size(1))
    L_topk = ((torch.topk(mu_f, k, dim=1).values - torch.topk(y_f, k, dim=1).values) ** 2).mean() * (load_scale ** 2)

    nll_mean = nll.mean()
    loss = nll_mean + lam_thr * L_thr + lam_q * L_q + lam_time * L_time + lam_amp * L_amp + lam_topk * L_topk

    if return_parts:
        return loss, dict(
            nll=float(nll_mean.detach().cpu()), thr=float(L_thr.detach().cpu()),
            q=float(L_q.detach().cpu()),        time=float(L_time.detach().cpu()),
            amp=float(L_amp.detach().cpu()),    topk=float(L_topk.detach().cpu()),
        )
    return loss


# ── Data loading ──────────────────────────────────────────────────────────────

def fill_missing_timestamps(df, time_col="TIME", device_col="FEEDER", freq="h"):
    """
    Ensure every 168-hour week in the data has exactly (168h / freq) rows.

    For each device, weeks that have at least one existing row are expanded
    to a full set of timestamps.  Missing values are filled by linear
    interpolation across the device's entire time series (so interpolation
    can span week boundaries).  Leading and trailing gaps that have no
    bounding value on one side are forward- or back-filled.

    Weeks with no rows at all are left out — only weeks that already exist
    in the data are filled.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `time_col`, `device_col`, and one or more numeric columns.
    time_col : str
        Name of the timestamp column (default "TIME").
    device_col : str
        Name of the feeder/device column (default "FEEDER").
    freq : str
        Expected time step (default "h").  Any fixed pandas offset alias works:
        "15min", "30min", etc.

    Returns
    -------
    pd.DataFrame
        Same columns as `df`, sorted by device then timestamp.  Every week
        that had data now has exactly (168h / freq) rows.
    """
    step_td        = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    steps_per_week = int(pd.Timedelta(hours=168) / step_td)
    step_s         = int(step_td.total_seconds())

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Numeric columns to interpolate (exclude device and time identifiers)
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in (time_col, device_col)
    ]

    pieces:   list = []
    gap_map:  dict = {}   # {timestamp -> [feeder_ids that were missing it]}
    warn_feeders: list = []

    for feeder, grp in df.groupby(device_col, sort=False):
        grp = grp.drop(columns=device_col).set_index(time_col).sort_index()

        # Collapse DST duplicate timestamps (fall-back repeated hour)
        if grp.index.duplicated().any():
            grp = grp.groupby(level=0).mean(numeric_only=True)

        # ── Build full expected index for all weeks that have data ────────────
        t0        = grp.index.min()
        elapsed_s = (grp.index - t0).total_seconds().values
        unique_weeks   = np.unique((elapsed_s / (168 * 3600)).astype(int))
        week_offsets_s = unique_weeks.astype(np.int64) * 168 * 3600
        step_offsets_s = np.arange(steps_per_week, dtype=np.int64) * step_s
        all_offsets_s  = (week_offsets_s[:, None] + step_offsets_s[None, :]).ravel()
        full_idx       = t0 + pd.to_timedelta(all_offsets_s, unit="s")

        # Record which timestamps are new
        inserted = full_idx.difference(grp.index).tolist()
        for ts in inserted:
            gap_map.setdefault(ts, []).append(str(feeder))

        # ── Reindex then interpolate across the whole time series ─────────────
        grp = grp.reindex(full_idx)
        grp[numeric_cols] = (
            grp[numeric_cols]
            .interpolate(method="time")   # linear between known timestamps
            .ffill()                       # fill leading edge
            .bfill()                       # fill trailing edge
        )

        if grp[numeric_cols].isna().any().any():
            warn_feeders.append(str(feeder))

        grp[device_col] = feeder
        grp.index.name  = time_col
        pieces.append(grp.reset_index())

    out = pd.concat(pieces, ignore_index=True)

    # ── Summary output ────────────────────────────────────────────────────────
    n_devices = df[device_col].nunique()
    if gap_map:
        total_ins = sum(len(v) for v in gap_map.values())
        affected  = len({d for v in gap_map.values() for d in v})
        print(f"Filled {total_ins} missing row(s) across {affected}/{n_devices} feeder(s):")

        pattern_map: dict = {}
        for ts, feeders in sorted(gap_map.items()):
            pattern_map.setdefault(tuple(sorted(feeders)), []).append(ts)

        for feeders, timestamps in sorted(pattern_map.items(), key=lambda x: -len(x[0])):
            n      = len(feeders)
            ts_str = ", ".join(str(t) for t in sorted(timestamps))
            if n == n_devices:
                print(f"  {len(timestamps)} timestamp(s) missing from all {n} feeder(s):  {ts_str}")
            elif n <= 5:
                print(f"  {len(timestamps)} timestamp(s) missing from "
                      f"{n} feeder(s) ({', '.join(feeders)}):  {ts_str}")
            else:
                print(f"  {len(timestamps)} timestamp(s) missing from "
                      f"{n}/{n_devices} feeder(s):  {ts_str}")

    if warn_feeders:
        print(f"  [WARN] Could not fill all gaps (feeder has only 1 row in a week): "
              f"{', '.join(warn_feeders)}")

    return out[df.columns]


def load_training_data(csv_path: str, feeder_col: str = "FEEDER", feeder_ids: list = None):
    """
    Load CSV data and segment each feeder into complete 168-hour weeks.

    Parameters
    ----------
    feeder_ids : list of str, optional
        Specific feeder IDs to include. If None or empty, all feeders are used.

    Returns
    -------
    List of dicts, one per feeder, each containing:
        load, temp, workday, season : np.ndarray [n_weeks, 168]
    Only fully-observed weeks (no NaN in any column) are kept.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for col in [COL_LOAD, COL_TEMP, COL_HUMIDITY, COL_HEATINDEX]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)

    if feeder_col not in df.columns:
        df[feeder_col] = "ALL"

    if feeder_ids:
        feeder_ids_str = [str(f) for f in feeder_ids]
        df = df[df[feeder_col].astype(str).isin(feeder_ids_str)]
        if df.empty:
            raise ValueError(f"None of the specified FEEDER_IDS were found in column '{feeder_col}'.")
        print(f"  Filtering to {len(feeder_ids_str)} specified feeders: {feeder_ids_str}")

    feeders = []
    for fid, gdf in df.groupby(feeder_col):
        gdf = gdf.set_index(COL_TIME).sort_index()
        # Collapse duplicate timestamps (e.g. DST fall-back creates a repeated hour)
        gdf = gdf.groupby(level=0).mean(numeric_only=True)
        # Fill gaps with a continuous hourly index so week boundaries are aligned
        full_idx = pd.date_range(start=gdf.index.min(), end=gdf.index.max(), freq="h")
        gdf = gdf.reindex(full_idx)

        n = (len(gdf) // NUM_IN_WEEK) * NUM_IN_WEEK
        if n == 0:
            continue

        def col_weeks(c):
            return gdf[c].to_numpy(dtype=float)[:n].reshape(-1, NUM_IN_WEEK)

        load_w    = col_weeks(COL_LOAD)
        temp_w    = col_weeks(COL_TEMP)
        workday_w = col_weeks(COL_HUMIDITY)
        season_w  = col_weeks(COL_HEATINDEX)

        valid = ~(
            np.isnan(load_w).any(axis=1)    |
            np.isnan(temp_w).any(axis=1)    |
            np.isnan(workday_w).any(axis=1) |
            np.isnan(season_w).any(axis=1)
        )

        if valid.sum() < 2:
            print(f"  [SKIP] Feeder {fid}: only {valid.sum()} complete weeks (need ≥ 2)")
            continue

        feeders.append(dict(
            id=fid,
            load=load_w[valid],
            temp=temp_w[valid],
            workday=workday_w[valid],
            season=season_w[valid],
        ))
        print(f"  [DATA] Feeder {fid}: {valid.sum()} complete weeks")

    if not feeders:
        raise ValueError("No usable feeder data. Check CSV format and column names.")

    return feeders


def build_features(feeders: list, cool_base: float, heat_base: float, fc_horizon: int) -> list:
    """
    Add CDD, HDD, and per-hour future temperature features to each feeder dict.
    Returns the same list with added keys in each dict.
    """
    for fd in feeders:
        temp = fd["temp"]                                    # [n_weeks, 168]
        fd["cdd"] = np.maximum(temp - cool_base, 0.0)
        fd["hdd"] = np.maximum(heat_base - temp, 0.0)

        # temp_fc_tplus{h}: at each hour t, the temperature h hours ahead (edge-padded)
        flat = temp.reshape(-1)
        T    = len(flat)
        gather = np.clip(np.arange(T)[:, None] + np.arange(fc_horizon)[None, :], 0, T - 1)
        future = flat[gather].reshape(temp.shape[0], NUM_IN_WEEK, fc_horizon)

        for h in range(fc_horizon):
            fd[f"temp_fc_tplus{h:02d}"] = future[:, :, h]

    return feeders


# ── Dataset construction ──────────────────────────────────────────────────────

def fit_scalers(feeders: list) -> dict:
    """
    Fit one global MinMaxScaler per feature across all feeders combined.
    Returns dict of {feature_name: fitted MinMaxScaler}.
    """
    feature_keys = [k for k in feeders[0] if k not in ("id",)]
    all_data = {k: [] for k in feature_keys}
    for fd in feeders:
        for k in feature_keys:
            all_data[k].append(fd[k].flatten())

    scalers = {}
    for k in feature_keys:
        vec = np.concatenate(all_data[k]).reshape(-1, 1)
        sc = MinMaxScaler()
        sc.fit(vec)
        scalers[k] = sc
    return scalers


def build_seq2seq_dataset(
    feeders: list,
    scalers: dict,
    *,
    train_ratio: float = 0.8,
    output_len: int = 24,
    device=None,
):
    """
    Build training samples using PRIOR-WEEK load as the decoder input.

    For each consecutive pair of weeks within the SAME feeder:
        enc_l   = normalized encoder load (week t)           [168]
        dec_l   = enc_l[:L]                                  [L=144]
        dec_ext = normalized decoder weather features        [L, K]
        target  = L+1 overlapping 24h windows of week t+1   [L+1, 24]

    Pairs are never created across feeder boundaries.
    """
    L = NUM_IN_WEEK - output_len  # 144
    feature_keys = [k for k in feeders[0] if k not in ("id",)]
    ext_keys = [k for k in feature_keys if k != "load"]

    def norm(arr, key):
        sc = scalers[key]
        return sc.transform(arr.reshape(-1, 1)).flatten().reshape(arr.shape)

    X_enc_l, X_dec_l, X_enc_ext, X_dec_ext, Y_target = [], [], [], [], []

    for fd in feeders:
        n_weeks = fd["load"].shape[0]
        load_n = norm(fd["load"], "load")                          # [n_weeks, 168]
        ext_n  = {k: norm(fd[k], k) for k in ext_keys}            # [n_weeks, 168] each

        for w in range(n_weeks - 1):
            enc_l    = load_n[w]                                   # [168]
            dec_full = load_n[w + 1]                               # [168]

            enc_ext = np.stack([ext_n[k][w]      for k in ext_keys], axis=-1)   # [168, K]
            dec_ext = np.stack([ext_n[k][w+1,:L] for k in ext_keys], axis=-1)   # [L,   K]

            targets = np.stack([dec_full[i:i+output_len] for i in range(L+1)], axis=0)  # [L+1, 24]

            X_enc_l.append(enc_l)
            X_dec_l.append(enc_l[:L])    # ← prior-week load, not teacher-forced future
            X_enc_ext.append(enc_ext)
            X_dec_ext.append(dec_ext)
            Y_target.append(targets)

    to_t = lambda a: torch.tensor(np.array(a, dtype=np.float32)).to(device)

    data = {
        "X_enc_l":   to_t(X_enc_l).unsqueeze(-1),    # [N, 168, 1]
        "X_enc_ext": to_t(X_enc_ext),                 # [N, 168, K]
        "X_dec_l":   to_t(X_dec_l).unsqueeze(-1),     # [N, L,   1]
        "X_dec_ext": to_t(X_dec_ext),                 # [N, L,   K]
        "Y_target":  to_t(Y_target).unsqueeze(-1),    # [N, L+1, 24, 1]
    }

    N     = data["X_enc_l"].shape[0]
    split = int(train_ratio * N)
    train = {k: v[:split] for k, v in data.items()}
    val   = {k: v[split:] for k, v in data.items()}

    print(f"\n  Total samples : {N}  →  {split} train / {N - split} val")
    for k, v in data.items():
        print(f"    {k:12s}: {tuple(v.shape)}")

    return train, val


def make_loader(split_dict: dict, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        split_dict["X_enc_l"],
        split_dict["X_enc_ext"],
        split_dict["X_dec_l"],
        split_dict["X_dec_ext"],
        split_dict["Y_target"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, epoch, best_loss, best_epoch):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_train":           best_loss,
        "best_epoch":           best_epoch,
    }, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"], strict=True)
        if optimizer is not None and "optimizer_state_dict" in obj:
            optimizer.load_state_dict(obj["optimizer_state_dict"])
        return int(obj.get("epoch", 0)), float(obj.get("best_train", float("inf"))), int(obj.get("best_epoch", -1))
    model.load_state_dict(obj, strict=True)
    return 0, float("inf"), -1


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model, train_loader, *,
    total_epochs, lr, device,
    top_k=2, kl_weight=0.001, warmup_epochs=10, peak_warmup_epochs=150,
    grad_clip=1.0, load_scale=1.0, topk_k=8,
    thr_frac=0.85, tau=0.05, q_upper=0.90, softarg_temp=0.12,
    lam_thr=0.05, lam_q=0.04, lam_time=0.01, lam_amp=0.03, lam_topk=0.01,
    checkpoint_best, checkpoint_latest,
    start_epoch=1, best_loss_init=float("inf"), best_epoch_init=-1,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if start_epoch > 1 and os.path.exists(checkpoint_latest):
        _, _, _ = load_checkpoint(checkpoint_latest, model, optimizer, device)

    best_loss  = best_loss_init
    best_epoch = best_epoch_init

    for ep in range(start_epoch, total_epochs + 1):
        model.train()
        running  = 0.0
        skipped  = 0
        ps       = dict(nll=0.0, thr=0.0, q=0.0, time=0.0, amp=0.0, topk=0.0, kl=0.0)
        n_batches = 0

        # Peak-loss terms warm up gradually over the first peak_warmup_epochs
        w_peak = 1.0 if peak_warmup_epochs <= 0 else min(1.0, ep / float(peak_warmup_epochs))

        for enc_l, enc_ext, dec_l, dec_ext, tgt in train_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = (
                enc_l.to(device), enc_ext.to(device),
                dec_l.to(device), dec_ext.to(device), tgt.to(device),
            )
            optimizer.zero_grad()

            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=top_k, warmup_epochs=warmup_epochs,
            )

            loss_main, parts = peak_fidelity_loss(
                mu_preds, logvar_preds, tgt,
                thr_frac=thr_frac, tau=tau, q_upper=q_upper,
                softarg_temp=softarg_temp,
                lam_thr=lam_thr * w_peak, lam_q=lam_q * w_peak,
                lam_time=lam_time * w_peak, lam_amp=lam_amp * w_peak,
                lam_topk=lam_topk * w_peak,
                load_scale=load_scale, topk_k=topk_k,
                return_parts=True,
            )

            if not torch.isfinite(loss_main):
                skipped += 1
                continue

            kl   = kl_loss(mu_z, logvar_z)
            loss = loss_main + kl_weight * kl

            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running += loss.item() * enc_l.size(0)
            for k in parts:
                ps[k] += parts[k]
            ps["kl"] += float(kl.detach().cpu())
            n_batches += 1

        avg = running / max(1, len(train_loader.dataset) - skipped * train_loader.batch_size)

        save_checkpoint(checkpoint_latest, model, optimizer, ep, best_loss, best_epoch)

        # Don't save best until the peak-loss warmup is complete
        if ep >= peak_warmup_epochs and avg < best_loss:
            best_loss  = avg
            best_epoch = ep
            save_checkpoint(checkpoint_best, model, optimizer, ep, best_loss, best_epoch)
            print(f"  ✓ New best at epoch {ep}: loss={best_loss:.6f}")

        if ep == start_epoch or ep % 10 == 0 or ep == total_epochs:
            best_str = f"{best_loss:.6f} (ep {best_epoch})" if best_epoch >= 0 else f"N/A (warmup until ep {peak_warmup_epochs})"
            print(f"Epoch {ep:4d}/{total_epochs}  loss={avg:.6f}  best={best_str}  skipped={skipped}")
            if n_batches > 0:
                p = {k: v / n_batches for k, v in ps.items()}
                print(f"  nll={p['nll']:.4f}  thr={p['thr']:.4f}  q={p['q']:.4f}  "
                      f"time={p['time']:.4f}  amp={p['amp']:.4f}  topk={p['topk']:.4f}  kl={p['kl']:.4f}")

    print(f"\nTraining complete.  Best epoch: {best_epoch}  Best loss: {best_loss:.6f}")
    return model


# ── CONFIG — set these before running ────────────────────────────────────────

# Paths
CSV_PATH   = "TransformerLoadData.csv"   # input data (all feeders in one file)
FEEDER_COL = "FEEDER"                    # column with feeder IDs (set None if absent)
OUTPUT_DIR = "."                         # directory for checkpoints and JSON files
MODEL_TAG  = "prior_week_v1"             # tag appended to all output filenames

# Feeders to include in training. Leave empty to train on all feeders in the CSV.
# Example: FEEDER_IDS = ["377136683", "377136684", "377136685"]
FEEDER_IDS = []

# Training  (see recommendations at the top of this file)
TOTAL_EPOCHS = 600     # recommended range: 600–1000
BATCH_SIZE   = 16
LR           = 1e-3
TRAIN_RATIO  = 0.8     # fraction of samples used for training
SEED         = 42
RESUME       = False   # set True to continue from the latest checkpoint

# Model architecture (keep consistent across training and inference)
XPRIME_DIM = 40
HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_LAYERS = 4
OUTPUT_LEN = 24
TOP_K      = 2
WARMUP_EP  = 10

# Feature engineering
COOL_BASE_F = 72.0
HEAT_BASE_F = 60.0

# Loss weights
KL_WEIGHT       = 0.001
PEAK_WARMUP_EP  = 150   # epochs before best-checkpoint saving is enabled
LAM_THR         = 0.05
LAM_Q           = 0.04
LAM_TIME        = 0.01
LAM_AMP         = 0.03
LAM_TOPK        = 0.01
THR_FRAC        = 0.85
TAU             = 0.05
Q_UPPER         = 0.90
SOFTARG_TEMP    = 0.12
TOPK_K          = 8

# ─────────────────────────────────────────────────────────────────────────────

def run():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ckpt_best   = os.path.join(OUTPUT_DIR, f"checkpoint_best_{MODEL_TAG}.pt")
    ckpt_latest = os.path.join(OUTPUT_DIR, f"checkpoint_latest_{MODEL_TAG}.pt")
    scaler_json = os.path.join(OUTPUT_DIR, f"vae_base_scaler_meta_{MODEL_TAG}.json")
    cfg_json    = os.path.join(OUTPUT_DIR, f"train_config_{MODEL_TAG}.json")

    # ── 1. Load and segment data ──────────────────────────────────────────────
    print(f"Loading data from {CSV_PATH} ...")
    feeders = load_training_data(CSV_PATH, feeder_col=FEEDER_COL or "FEEDER", feeder_ids=FEEDER_IDS or None)
    total_weeks = sum(fd["load"].shape[0] for fd in feeders)
    print(f"Total: {total_weeks} complete weeks across {len(feeders)} feeders")

    # ── 2. Build derived features ─────────────────────────────────────────────
    feeders = build_features(feeders,
                             cool_base=COOL_BASE_F,
                             heat_base=HEAT_BASE_F,
                             fc_horizon=OUTPUT_LEN)

    feature_keys = [k for k in feeders[0] if k != "id"]
    ext_keys     = [k for k in feature_keys if k != "load"]
    ext_idx_map  = {k: i for i, k in enumerate(ext_keys)}

    thermal_feature_names = ["temp", "cdd", "hdd"] + [f"temp_fc_tplus{h:02d}" for h in range(OUTPUT_LEN)]
    workday_feature_name  = "workday"
    season_feature_name   = "season"
    thermal_indices = [ext_idx_map[k] for k in thermal_feature_names]
    workday_index   = ext_idx_map[workday_feature_name]
    season_index    = ext_idx_map[season_feature_name]

    print(f"External features: {len(ext_keys)}  ({ext_keys[:4]} ...)")

    # ── 3. Fit global scalers ─────────────────────────────────────────────────
    print("\nFitting global scalers ...")
    scalers = fit_scalers(feeders)

    scaler_meta = {}
    for k, sc in scalers.items():
        scaler_meta[f"{k}_min"] = float(sc.data_min_[0])
        scaler_meta[f"{k}_max"] = float(sc.data_max_[0])
    with open(scaler_json, "w") as f:
        json.dump(scaler_meta, f, indent=2)
    print(f"Scaler meta → {scaler_json}")

    load_scale  = float(scalers["load"].data_max_[0] - scalers["load"].data_min_[0])
    n_externals = len(ext_keys)

    # ── 4. Build dataset ──────────────────────────────────────────────────────
    print("\nBuilding seq2seq dataset (prior-week decoder input) ...")
    train_data, val_data = build_seq2seq_dataset(
        feeders, scalers,
        train_ratio=TRAIN_RATIO,
        output_len=OUTPUT_LEN,
        device=device,
    )
    train_loader = make_loader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # ── 5. Save train config ──────────────────────────────────────────────────
    train_cfg = dict(
        MODEL_TAG=MODEL_TAG,
        seed=SEED,
        decoder_input_mode="prior_week",
        temp_bases_F=dict(cool_base=COOL_BASE_F, heat_base=HEAT_BASE_F),
        model_dims=dict(
            xprime_dim=XPRIME_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
            num_layers=NUM_LAYERS, output_len=OUTPUT_LEN,
            encoder_len_weeks=1, decoder_len_weeks=1,
            input_dim=1, output_dim=1,
            n_externals=n_externals, warmup_ep=WARMUP_EP,
        ),
        ext_keys=ext_keys,
        ext_idx_map=ext_idx_map,
        experts=dict(
            thermal_feature_names=thermal_feature_names,
            workday_feature_name=workday_feature_name,
            season_feature_name=season_feature_name,
            top_k=TOP_K,
        ),
    )
    with open(cfg_json, "w") as f:
        json.dump(train_cfg, f, indent=2)
    print(f"Train config  → {cfg_json}")

    # ── 6. Build model ────────────────────────────────────────────────────────
    model = VariationalSeq2Seq_meta(
        xprime_dim=XPRIME_DIM, input_dim=1, hidden_size=HIDDEN_DIM,
        latent_size=LATENT_DIM, output_len=OUTPUT_LEN, n_externals=n_externals,
        output_dim=1, num_layers=NUM_LAYERS, dropout=0.1,
        thermal_indices=thermal_indices,
        workday_index=workday_index,
        season_index=season_index,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    start_epoch     = 1
    best_loss_init  = float("inf")
    best_epoch_init = -1

    if RESUME and os.path.exists(ckpt_latest):
        start_epoch, best_loss_init, best_epoch_init = load_checkpoint(
            ckpt_latest, model, device=device,
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, best loss so far: {best_loss_init:.6f}")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {TOTAL_EPOCHS} epochs"
          f"  (best checkpoint saved after warmup at epoch {PEAK_WARMUP_EP})\n")

    train(
        model, train_loader,
        total_epochs=TOTAL_EPOCHS, lr=LR, device=device,
        top_k=TOP_K, kl_weight=KL_WEIGHT, warmup_epochs=WARMUP_EP,
        peak_warmup_epochs=PEAK_WARMUP_EP, grad_clip=1.0, load_scale=load_scale,
        thr_frac=THR_FRAC, tau=TAU, q_upper=Q_UPPER, softarg_temp=SOFTARG_TEMP,
        lam_thr=LAM_THR, lam_q=LAM_Q, lam_time=LAM_TIME,
        lam_amp=LAM_AMP, lam_topk=LAM_TOPK, topk_k=TOPK_K,
        checkpoint_best=ckpt_best, checkpoint_latest=ckpt_latest,
        start_epoch=start_epoch, best_loss_init=best_loss_init,
        best_epoch_init=best_epoch_init,
    )

    # ── 8. Print inference instructions ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  Checkpoint (best)   : {ckpt_best}")
    print(f"  Checkpoint (latest) : {ckpt_latest}")
    print(f"  Scaler meta         : {scaler_json}")
    print(f"  Train config        : {cfg_json}")
    print()
    print("To use for inference, set in predict_week_ahead.py:")
    print(f"  CHECKPOINT_PATH  = '{ckpt_best}'")
    print(f"  SCALER_META_PATH = '{scaler_json}'")
    print(f"  TRAIN_CFG_PATH   = '{cfg_json}'")
    print(f"  ALPHA            = 0.0   # prior-week mode matches training")
    print("=" * 60)


run()
