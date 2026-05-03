# =========================================================
# Wavelet + AR Extension for ALL runs
# Target: 20 wave cycles
# Saves:
#   1. extended CSV inside each run folder
#   2. PNG plot inside each run folder
#   3. summary CSV
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler

try:
    import pywt
except ImportError:
    !pip install PyWavelets
    import pywt

# =========================================================
# SETTINGS
# =========================================================

base_dir = "/content/drive/MyDrive/Final_Manual_Dataset_Augmented"

required_wave_cycles = 20

best_lag = 2
wavelet_name = "db4"
wavelet_level = 2

imu_columns = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]

wave_periods = {
    "SouthWest_2_56.3": 0.600,
    "SouthWest_1_56.3": 0.600,

    "South_2_56.3": 0.600,
    "South_1_56.3": 0.600,
    "South_1_39.1": 0.500,

    "NorthWest_2_56.3": 0.600,
    "NorthWest_1_100": 0.800,
    "NorthWest_1_56.3": 0.600,
    "NorthWest_1_25": 0.400,
    "NorthWest_0.7_56.3": 0.600,
    "NorthWest_0.4_100": 0.800,

    "North_2_25": 0.400,
    "North_0.7_39.1": 0.500
}

# =========================================================
# HELPERS
# =========================================================

def estimate_imu_frequency(df):
    df["timestamp"] = pd.to_datetime(df["timestamp_iso_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    time_diffs = df["timestamp"].diff().dropna()
    time_diffs_sec = time_diffs.dt.total_seconds()
    time_diffs_sec = time_diffs_sec[time_diffs_sec > 0]

    if len(time_diffs_sec) == 0:
        raise ValueError("No valid timestamp differences found.")

    avg_dt = time_diffs_sec.mean()
    imu_freq = 1 / avg_dt

    return df, avg_dt, imu_freq


def ar_forecast(series, n_steps, lag):
    if n_steps == 0:
        return np.array([])

    if len(series) <= lag + 2:
        return np.ones(n_steps) * series[-1]

    lag = min(lag, len(series) - 2)

    try:
        model = AutoReg(series, lags=lag, old_names=False).fit()
        pred = model.predict(
            start=len(series),
            end=len(series) + n_steps - 1,
            dynamic=False
        )
        return np.array(pred)

    except Exception:
        return np.ones(n_steps) * series[-1]


def wavelet_ar_forecast(signal, n_future, wavelet_name="db4", level=2, lag=2):
    if n_future == 0:
        return np.array([])

    scaler = MinMaxScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    max_level = pywt.dwt_max_level(
        data_len=len(signal_scaled),
        filter_len=pywt.Wavelet(wavelet_name).dec_len
    )

    level = min(level, max_level)

    if level < 1:
        forecast_scaled = ar_forecast(signal_scaled, n_future, lag)
        return scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    coeffs = pywt.wavedec(
        signal_scaled,
        wavelet=wavelet_name,
        level=level,
        mode="symmetric"
    )

    components = []

    for i in range(len(coeffs)):
        coeffs_component = []

        for j, c in enumerate(coeffs):
            if i == j:
                coeffs_component.append(c)
            else:
                coeffs_component.append(np.zeros_like(c))

        component = pywt.waverec(
            coeffs_component,
            wavelet=wavelet_name,
            mode="symmetric"
        )

        component = component[:len(signal_scaled)]
        components.append(component)

    future_components = []

    for comp in components:
        comp_future = ar_forecast(comp, n_future, lag)
        future_components.append(comp_future)

    future_components = np.array(future_components)

    forecast_scaled = future_components.sum(axis=0)

    forecast = scaler.inverse_transform(
        forecast_scaled.reshape(-1, 1)
    ).flatten()

    return forecast


# =========================================================
# MAIN LOOP
# =========================================================

summary_rows = []

for run_name, wave_period in wave_periods.items():

    print("\n======================================")
    print("Processing:", run_name)
    print("======================================")

    run_dir = os.path.join(base_dir, run_name)
    csv_path = os.path.join(run_dir, "data.csv")

    if not os.path.exists(csv_path):
        print("Missing:", csv_path)
        continue

    df = pd.read_csv(csv_path)

    if "timestamp_iso_ms" not in df.columns:
        print("Skipping because timestamp_iso_ms is missing:", run_name)
        continue

    try:
        df, avg_dt, imu_frequency = estimate_imu_frequency(df)
    except Exception as e:
        print("Skipping due to timestamp problem:", e)
        continue

    current_samples = len(df)

    required_samples = int(
        np.ceil(required_wave_cycles * wave_period * imu_frequency)
    )

    n_future = max(0, required_samples - current_samples)

    print("Wave period:", wave_period)
    print("IMU frequency:", imu_frequency)
    print("Current samples:", current_samples)
    print("Required samples:", required_samples)
    print("Future samples needed:", n_future)

    # =====================================================
    # Create future timestamps
    # =====================================================

    last_timestamp = df["timestamp"].iloc[-1]

    future_timestamps = [
        last_timestamp + pd.to_timedelta((i + 1) * avg_dt, unit="s")
        for i in range(n_future)
    ]

    # =====================================================
    # Create extended dataframe
    # =====================================================

    extended_df = df.copy()

    future_df = pd.DataFrame(index=range(n_future))

    if n_future > 0:
        future_df["timestamp"] = future_timestamps
        future_df["timestamp_iso_ms"] = [
            ts.isoformat() for ts in future_timestamps
        ]

        if "timestamp_ns" in df.columns:
            future_df["timestamp_ns"] = [
                int(ts.value) for ts in future_timestamps
            ]

    # =====================================================
    # Forecast IMU columns
    # =====================================================

    available_imu_columns = [
        col for col in imu_columns if col in df.columns
    ]

    for col in available_imu_columns:

        signal = df[col].values.astype(float)

        forecast = wavelet_ar_forecast(
            signal=signal,
            n_future=n_future,
            wavelet_name=wavelet_name,
            level=wavelet_level,
            lag=best_lag
        )

        if n_future > 0:
            future_df[col] = forecast

    # =====================================================
    # Fill non-IMU columns for forecast rows
    # =====================================================

    for col in df.columns:

        if col in future_df.columns:
            continue

        if col in available_imu_columns:
            continue

        if col == "timestamp":
            continue

        if col == "timestamp_iso_ms":
            continue

        if col == "timestamp_ns":
            continue

        # labels/metadata are copied from last real row
        if col in ["wave_height", "wave_direction", "direction", "label"]:
            future_df[col] = df[col].iloc[-1]

        # image files should be empty for generated samples
        elif "image" in col.lower():
            future_df[col] = None

        else:
            future_df[col] = df[col].iloc[-1]

    # Add source column
    extended_df["source"] = "real"

    if n_future > 0:
        future_df["source"] = "forecast"
        extended_df = pd.concat(
            [extended_df, future_df],
            ignore_index=True
        )

    # =====================================================
    # Save extended CSV in same run folder
    # =====================================================

    extended_csv_path = os.path.join(
        run_dir,
        f"data_wavelet_AR_extended_{required_wave_cycles}cycles.csv"
    )

    extended_df.to_csv(extended_csv_path, index=False)

    print("Saved extended CSV:", extended_csv_path)

    # =====================================================
    # Plot main IMU signals
    # =====================================================

    plot_cols = [c for c in ["gx", "gy", "gz"] if c in extended_df.columns]

    if len(plot_cols) == 0:
        plot_cols = available_imu_columns[:3]

    plt.figure(figsize=(14, 6))

    for col in plot_cols:
        plt.plot(
            range(current_samples),
            extended_df[col].iloc[:current_samples],
            marker="o",
            label=f"{col} real"
        )

        if n_future > 0:
            plt.plot(
                range(current_samples, len(extended_df)),
                extended_df[col].iloc[current_samples:],
                marker="x",
                linewidth=2,
                label=f"{col} forecast"
            )

    plt.axvline(
        current_samples - 1,
        linestyle="--",
        color="black",
        label="Start of extension"
    )

    plt.title(
        f"Wavelet+AR IMU Extension - {run_name}\n"
        f"{required_wave_cycles} cycles, T={wave_period}s, fs={imu_frequency:.2f}Hz"
    )

    plt.xlabel("Sample Index")
    plt.ylabel("IMU Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plot_path = os.path.join(
        run_dir,
        f"wavelet_AR_extension_{required_wave_cycles}cycles.png"
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved plot:", plot_path)

    # =====================================================
    # Add summary
    # =====================================================

    summary_rows.append({
        "run_name": run_name,
        "wave_period_sec": wave_period,
        "required_wave_cycles": required_wave_cycles,
        "imu_frequency_Hz": imu_frequency,
        "avg_dt_sec": avg_dt,
        "current_samples": current_samples,
        "required_samples": required_samples,
        "forecast_samples_added": n_future,
        "extended_samples": len(extended_df),
        "extended_csv_path": extended_csv_path,
        "plot_path": plot_path
    })

# =========================================================
# Save summary CSV
# =========================================================

summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    base_dir,
    f"wavelet_AR_extension_summary_{required_wave_cycles}cycles.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n======================================")
print("DONE")
print("Summary saved to:")
print(summary_path)
print("======================================")

print(summary_df)
