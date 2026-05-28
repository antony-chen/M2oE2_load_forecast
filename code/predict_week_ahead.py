"""
Standalone inference script for M2OE2 week-ahead load forecasting.

Set the file paths and CSV path in the CONFIG section at the bottom of this
file, then run it directly: python predict_week_ahead.py

The CSV must have columns:
    TIME                  - hourly timestamp
    KWH                        - load (only first 168 rows are used)
    SURDPOINTTEMPFAHRENHEIT    - dew point temperature
    RELATIVEHUMIDITY           - relative humidity
    HEATINDEXFAHRENHEIT        - heat index

Rows 0-167   -> encoder (past week actuals, KWH is read)
Rows 168-335 -> decoder (forecast week weather only, KWH is ignored)

Output CSV columns:
    TIME       - timestamps from the forecast week
    predicted_kwh   - mean forecast
    predicted_std   - uncertainty (std dev)
    lower_90        - lower bound of 90% prediction interval
    upper_90        - upper bound of 90% prediction interval
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from model_v2 import VariationalSeq2Seq_meta
from data_utils_Final import reconstruct_sequence


# ── Column names in the ONCOR CSV ───────────────────────────────────────────
COL_TIME      = "TIME"
COL_LOAD      = "KWH"
COL_TEMP      = "SURDPOINTTEMPFAHRENHEIT"   # dew point — maps to model key "temp"
COL_HUMIDITY  = "RELATIVEHUMIDITY"           # maps to model key "workday"
COL_HEATINDEX = "HEATINDEXFAHRENHEIT"        # maps to model key "season"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_feature_array(key: str, temp, humidity, heatindex, cool_base: float, heat_base: float) -> np.ndarray:
    """
    Build a single [T] feature array from raw inputs, matching the feature
    construction logic in main_M2oE2_Final exactly.
    """
    temp      = np.asarray(temp,      dtype=float)
    humidity  = np.asarray(humidity,  dtype=float)
    heatindex = np.asarray(heatindex, dtype=float)
    T = len(temp)

    if key == "temp":
        return temp
    elif key == "workday":
        return humidity
    elif key == "season":
        return heatindex
    elif key == "cdd":
        return np.maximum(temp - cool_base, 0.0)
    elif key == "hdd":
        return np.maximum(heat_base - temp, 0.0)
    elif key.startswith("temp_fc_tplus"):
        h = int(key.split("tplus")[1])
        return temp[np.clip(np.arange(T) + h, 0, T - 1)]
    else:
        raise KeyError(f"Unknown feature key: '{key}'")


def _normalize(arr: np.ndarray, key: str, meta: dict) -> np.ndarray:
    lo = meta[f"{key}_min"]
    hi = meta[f"{key}_max"]
    return (np.asarray(arr, dtype=float) - lo) / (hi - lo + 1e-12)


def _load_model(checkpoint_path: str, train_cfg_path: str, device: torch.device):
    with open(train_cfg_path) as f:
        cfg = json.load(f)
    dims = cfg["model_dims"]

    # Reconstruct block-expert indices so the model is built in the same
    # mode it was trained in (3-expert block mode vs n_externals fallback).
    thermal_indices = None
    workday_index   = None
    season_index    = None

    ext_idx_map = cfg.get("ext_idx_map", {})
    experts     = cfg.get("experts", {})
    if ext_idx_map and experts:
        thermal_names   = experts.get("thermal_feature_names", [])
        workday_name    = experts.get("workday_feature_name")
        season_name     = experts.get("season_feature_name")
        thermal_indices = [ext_idx_map[k] for k in thermal_names if k in ext_idx_map] or None
        workday_index   = ext_idx_map.get(workday_name)
        season_index    = ext_idx_map.get(season_name)

    model = VariationalSeq2Seq_meta(
        xprime_dim      = dims["xprime_dim"],
        input_dim       = dims["input_dim"],
        hidden_size     = dims["hidden_dim"],
        latent_size     = dims["latent_dim"],
        output_len      = dims["output_len"],
        n_externals     = dims["n_externals"],
        output_dim      = dims["output_dim"],
        num_layers      = dims["num_layers"],
        dropout         = 0.0,
        thermal_indices = thermal_indices,
        workday_index   = workday_index,
        season_index    = season_index,
    ).to(device)

    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"], strict=True)
    else:
        model.load_state_dict(obj, strict=True)
    model.eval()
    return model, cfg


# ── Public API ───────────────────────────────────────────────────────────────

def predict_week_ahead(
    checkpoint_path: str,
    scaler_meta_path: str,
    train_cfg_path: str,
    # Encoder inputs — last 168h of metered actuals
    past_168h_load: np.ndarray,       # KWH
    past_168h_temp: np.ndarray,       # SURDPOINTTEMPFAHRENHEIT (dew point °F)
    past_168h_humidity: np.ndarray,   # RELATIVEHUMIDITY (%)
    past_168h_heatindex: np.ndarray,  # HEATINDEXFAHRENHEIT (°F)
    # Decoder inputs — next 168h weather forecast (KWH is NOT an input here)
    forecast_168h_temp: np.ndarray,       # SURDPOINTTEMPFAHRENHEIT forecast
    forecast_168h_humidity: np.ndarray,   # RELATIVEHUMIDITY forecast
    forecast_168h_heatindex: np.ndarray,  # HEATINDEXFAHRENHEIT forecast
    device: torch.device = None,
):
    """
    Predict the next 168 hours of KWH load given last week's actuals and next
    week's weather forecast.

    Parameters
    ----------
    checkpoint_path   : path to the trained model .pt file
    scaler_meta_path  : path to vae_base_scaler_meta_*.json (from training)
    train_cfg_path    : path to train_config_*.json (from training)
    past_168h_*       : 168-element arrays of last week's observed values
    forecast_168h_*   : 168-element arrays of next week's weather forecast
    device            : torch device; auto-detected if None

    Returns
    -------
    mu_kwh  : np.ndarray shape [168] — predicted mean KWH for each hour
    std_kwh : np.ndarray shape [168] — predicted std KWH for each hour
                                       use ± 1.645 * std for 90% interval
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(scaler_meta_path) as f:
        meta = json.load(f)

    model, cfg = _load_model(checkpoint_path, train_cfg_path, device)
    output_len = cfg["model_dims"]["output_len"]
    L = 168 - output_len  # decoder input length (144 when output_len=24)

    ext_keys  = cfg["ext_keys"]
    cool_base = cfg.get("temp_bases_F", {}).get("cool_base", 72.0)
    heat_base = cfg.get("temp_bases_F", {}).get("heat_base", 60.0)

    # ── Build encoder tensors from last week's actuals ───────────────────────
    enc_l_np = _normalize(past_168h_load, "load", meta)  # [168]

    enc_ext_np = np.stack([
        _normalize(
            _build_feature_array(k, past_168h_temp, past_168h_humidity, past_168h_heatindex, cool_base, heat_base),
            k, meta,
        )
        for k in ext_keys
    ], axis=-1)  # [168, K_ext]

    # ── Build decoder tensors from next week's weather forecast ─────────────
    # Only the first L hours are needed for the decoder input.
    # The future load (dec_l) is unknown so we feed zeros.
    dec_ext_np = np.stack([
        _normalize(
            _build_feature_array(k, forecast_168h_temp, forecast_168h_humidity, forecast_168h_heatindex, cool_base, heat_base)[:L],
            k, meta,
        )
        for k in ext_keys
    ], axis=-1)  # [L, K_ext]

    # Use last week's actual load as decoder input rather than zeros.
    # The model was trained with teacher forcing (real future load fed at each
    # decoder step), so feeding zeros causes the GRU hidden state to drift after
    # ~100 steps, making predictions collapse toward zero. Week-over-week load
    # is highly correlated, so the prior week is a far better proxy than zeros.
    dec_l_np = enc_l_np[:L].reshape(L, 1).astype(np.float32)

    # ── Convert to batched tensors (batch size = 1) ──────────────────────────
    enc_l   = torch.tensor(enc_l_np,   dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1,168,1]
    enc_ext = torch.tensor(enc_ext_np, dtype=torch.float32).unsqueeze(0).to(device)                # [1,168,7]
    dec_l   = torch.tensor(dec_l_np,   dtype=torch.float32).unsqueeze(0).to(device)                # [1,144,1]
    dec_ext = torch.tensor(dec_ext_np, dtype=torch.float32).unsqueeze(0).to(device)                # [1,144,7]

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        mu_preds, logvar_preds, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
        # mu_preds:     [1, L+1, output_len, 1]  =  [1, 145, 24, 1]
        # logvar_preds: [1, L+1, output_len, 1]

    # ── Reconstruct full 168h from overlapping 24h windows ───────────────────
    mu_norm  = reconstruct_sequence(mu_preds[0, :, :, 0].cpu())            # [168]
    std_norm = reconstruct_sequence((0.5 * logvar_preds[0, :, :, 0].cpu()).exp())  # [168]

    # ── Denormalize to original KWH units ────────────────────────────────────
    lo, hi  = meta["load_min"], meta["load_max"]
    mu_kwh  = mu_norm.numpy()  * (hi - lo) + lo
    std_kwh = std_norm.numpy() * (hi - lo)

    return mu_kwh, std_kwh


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_forecast(
    past_load: np.ndarray,       # [168] historical KWH
    mu_kwh: np.ndarray,          # [168] predicted mean KWH
    std_kwh: np.ndarray,         # [168] predicted std KWH
    past_temp: np.ndarray,       # [168] historical temperature
    forecast_temp: np.ndarray,   # [168] forecast temperature
    past_timestamps,             # [168] datetime-like values for past week
    forecast_timestamps,         # [168] datetime-like values for forecast week
    out_png: str,
    feeder: str = "",
    actual_load: np.ndarray = None,  # [168] actual KWH for forecast week, if known
):
    past_dt    = pd.to_datetime(past_timestamps)
    forecast_dt = pd.to_datetime(forecast_timestamps)

    fig, ax = plt.subplots(figsize=(12, 3.6))

    # Load: history and forecast
    ax.plot(past_dt,    past_load, color="black", linewidth=1.5, label="History")
    ax.plot(forecast_dt, mu_kwh,  color="blue",  linewidth=1.5, label="Forecast (mean)")
    ax.fill_between(
        forecast_dt,
        mu_kwh - std_kwh,
        mu_kwh + std_kwh,
        color="blue", alpha=0.15, label="Forecast (±1σ)"
    )

    if actual_load is not None:
        ax.plot(forecast_dt, actual_load, color="black", linewidth=1.5,
                linestyle="--", label="Actual")

    # Vertical line at history/forecast boundary
    ax.axvline(forecast_dt[0], color="grey", linestyle="--", alpha=0.5)

    ax.set_ylabel("Load (KWH)")
    ax.set_xlabel("Date")
    title = f"Week-ahead forecast"
    if feeder:
        title += f"  |  FEEDER {feeder}"
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Temperature on secondary axis
    ax2 = ax.twinx()
    ax2.plot(past_dt,    past_temp,    linestyle=":", linewidth=1.0, color="orange", alpha=0.6, label="Temp (hist)")
    ax2.plot(forecast_dt, forecast_temp, linestyle=":", linewidth=1.0, color="darkorange", alpha=0.6, label="Temp (fore)")
    ax2.set_ylabel("Temperature (°F)")

    # Keep load on top of temperature
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out_png}")


# ── CONFIG — set these paths before running ──────────────────────────────────

CSV_PATH         = "two_week_data.csv"
CHECKPOINT_PATH  = "Oncor_load_M2OE2_v1_temp_24h_Enc1w_v5_v1temp_oracle_base_best.pt"
SCALER_META_PATH = "vae_base_scaler_meta_v5_v1temp_oracle.json"
TRAIN_CFG_PATH   = "train_config_v5_v1temp_oracle.json"
OUTPUT_CSV_PATH  = "forecast_output.csv"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    if len(df) < 336:
        raise ValueError(
            f"CSV must have at least 336 rows (168 past + 168 forecast). Found {len(df)}."
        )

    for col in [COL_LOAD, COL_TEMP, COL_HUMIDITY, COL_HEATINDEX]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    past  = df.iloc[:168]
    fcast = df.iloc[168:336]

    mu_kwh, std_kwh = predict_week_ahead(
        checkpoint_path         = CHECKPOINT_PATH,
        scaler_meta_path        = SCALER_META_PATH,
        train_cfg_path          = TRAIN_CFG_PATH,
        past_168h_load          = past[COL_LOAD].to_numpy(dtype=float),
        past_168h_temp          = past[COL_TEMP].to_numpy(dtype=float),
        past_168h_humidity      = past[COL_HUMIDITY].to_numpy(dtype=float),
        past_168h_heatindex     = past[COL_HEATINDEX].to_numpy(dtype=float),
        forecast_168h_temp      = fcast[COL_TEMP].to_numpy(dtype=float),
        forecast_168h_humidity  = fcast[COL_HUMIDITY].to_numpy(dtype=float),
        forecast_168h_heatindex = fcast[COL_HEATINDEX].to_numpy(dtype=float),
    )

    timestamps = fcast[COL_TIME].values if COL_TIME in fcast.columns else np.arange(168)

    # Detect whether actual load values are present for the forecast week.
    # Treat all-zero or all-NaN as "not available".
    fcast_load_raw = fcast[COL_LOAD].to_numpy(dtype=float)
    actual_kwh = None
    if not (np.all(np.isnan(fcast_load_raw)) or np.all(fcast_load_raw == 0)):
        actual_kwh = fcast_load_raw

    out_dict = {
        COL_TIME:        timestamps,
        "predicted_kwh": mu_kwh,
        "predicted_std": std_kwh,
        "lower_90":      mu_kwh - 1.645 * std_kwh,
        "upper_90":      mu_kwh + 1.645 * std_kwh,
    }
    if actual_kwh is not None:
        out_dict["actual_kwh"] = actual_kwh

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Forecast saved to {OUTPUT_CSV_PATH}  ({len(out_df)} hourly rows)")
    print(f"  Mean KWH range : {mu_kwh.min():.3f} – {mu_kwh.max():.3f}")
    print(f"  Mean std range : {std_kwh.min():.3f} – {std_kwh.max():.3f}")
    if actual_kwh is not None:
        print(f"  Actual KWH range: {actual_kwh.min():.3f} – {actual_kwh.max():.3f}")

    out_png = OUTPUT_CSV_PATH.replace(".csv", ".png")
    feeder_id = df["FEEDER"].iloc[0] if "FEEDER" in df.columns else ""
    plot_forecast(
        past_load           = past[COL_LOAD].to_numpy(dtype=float),
        mu_kwh              = mu_kwh,
        std_kwh             = std_kwh,
        past_temp           = past[COL_TEMP].to_numpy(dtype=float),
        forecast_temp       = fcast[COL_TEMP].to_numpy(dtype=float),
        past_timestamps     = past[COL_TIME].values,
        forecast_timestamps = fcast[COL_TIME].values,
        out_png             = out_png,
        feeder              = str(feeder_id),
        actual_load         = actual_kwh,
    )
