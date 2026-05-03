# =========================================================
# Batch Validation: Pure STL vs Pure Wavelet
# Across all runs, using middle + last test segments
# Saves RMSE / MAE / R2 bar graphs
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import STL

try:
    import pywt
except ImportError:
    !pip install PyWavelets
    import pywt

# =========================================================
# SETTINGS
# =========================================================

base_dir = "/content/drive/MyDrive/Final_Manual_Dataset_Augmented"

run_names = [
    "SouthWest_2_56.3",
    "SouthWest_1_56.3",
    "South_2_56.3",
    "South_1_56.3",
    "South_1_39.1",
    "NorthWest_2_56.3",
    "NorthWest_1_100",
    "NorthWest_1_56.3",
    "NorthWest_1_25",
    "NorthWest_0.7_56.3",
    "NorthWest_0.4_100",
    "North_2_25",
    "North_0.7_39.1"
]

signal_name = "gx"

save_dir = "/content/drive/MyDrive/validation_pure_stl_wavelet"
os.makedirs(save_dir, exist_ok=True)

test_ratio = 0.20

stl_period = 6
wavelet_name = "db4"
wavelet_level = 2

# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

# =========================================================
# PURE STL FORECAST
# =========================================================

def pure_stl_forecast(train_signal, n_steps, stl_period=6):

    if n_steps == 0:
        return np.array([])

    if stl_period >= len(train_signal) // 2:
        return np.ones(n_steps) * train_signal[-1]

    try:
        stl = STL(train_signal, period=stl_period, robust=True)
        result = stl.fit()

        trend = result.trend
        seasonal = result.seasonal

        # Trend extension using last slope
        trend_slope = trend[-1] - trend[-2]

        trend_future = np.array([
            trend[-1] + (i + 1) * trend_slope
            for i in range(n_steps)
        ])

        # Seasonal extension by repeating last cycle
        last_cycle = seasonal[-stl_period:]

        seasonal_future = np.tile(
            last_cycle,
            int(np.ceil(n_steps / stl_period))
        )[:n_steps]

        return trend_future + seasonal_future

    except Exception:
        return np.ones(n_steps) * train_signal[-1]

# =========================================================
# PURE WAVELET FORECAST
# =========================================================

def pure_wavelet_forecast(train_signal, n_steps, wavelet_name="db4", level=2):

    if n_steps == 0:
        return np.array([])

    try:
        max_level = pywt.dwt_max_level(
            data_len=len(train_signal),
            filter_len=pywt.Wavelet(wavelet_name).dec_len
        )

        level = min(level, max_level)

        if level < 1:
            return np.ones(n_steps) * train_signal[-1]

        coeffs = pywt.wavedec(
            train_signal,
            wavelet=wavelet_name,
            level=level,
            mode="symmetric"
        )

        components = []

        for i in range(len(coeffs)):

            coeffs_i = []

            for j, c in enumerate(coeffs):
                if i == j:
                    coeffs_i.append(c)
                else:
                    coeffs_i.append(np.zeros_like(c))

            component = pywt.waverec(
                coeffs_i,
                wavelet=wavelet_name,
                mode="symmetric"
            )

            component = component[:len(train_signal)]
            components.append(component)

        future_components = []

        for idx, comp in enumerate(components):

            if idx == 0:
                # approximation / trend component
                slope = comp[-1] - comp[-2]

                comp_future = np.array([
                    comp[-1] + (i + 1) * slope
                    for i in range(n_steps)
                ])

            else:
                # detail components: repeat recent oscillatory tail
                tail_length = min(8, len(comp))
                tail = comp[-tail_length:]

                comp_future = np.tile(
                    tail,
                    int(np.ceil(n_steps / tail_length))
                )[:n_steps]

            future_components.append(comp_future)

        future_components = np.array(future_components)

        forecast = future_components.sum(axis=0)

        return forecast

    except Exception:
        return np.ones(n_steps) * train_signal[-1]

# =========================================================
# VALIDATION FUNCTION
# =========================================================

def validate_method(signal_scaled, method_name):

    n = len(signal_scaled)
    test_len = max(3, int(n * test_ratio))

    # Middle test
    middle_start = int(n * 0.45)
    middle_end = middle_start + test_len

    if middle_end >= n:
        middle_end = n - test_len
        middle_start = middle_end - test_len

    middle_train = signal_scaled[:middle_start]
    middle_test = signal_scaled[middle_start:middle_end]

    # Last test
    last_start = n - test_len
    last_end = n

    last_train = signal_scaled[:last_start]
    last_test = signal_scaled[last_start:last_end]

    if method_name == "Pure_STL":

        middle_pred = pure_stl_forecast(
            middle_train,
            len(middle_test),
            stl_period=stl_period
        )

        last_pred = pure_stl_forecast(
            last_train,
            len(last_test),
            stl_period=stl_period
        )

    elif method_name == "Pure_Wavelet":

        middle_pred = pure_wavelet_forecast(
            middle_train,
            len(middle_test),
            wavelet_name=wavelet_name,
            level=wavelet_level
        )

        last_pred = pure_wavelet_forecast(
            last_train,
            len(last_test),
            wavelet_name=wavelet_name,
            level=wavelet_level
        )

    else:
        raise ValueError("Unknown method")

    middle_metrics = compute_metrics(middle_test, middle_pred)
    last_metrics = compute_metrics(last_test, last_pred)

    return {
        "middle_RMSE": middle_metrics["RMSE"],
        "middle_MAE": middle_metrics["MAE"],
        "middle_R2": middle_metrics["R2"],

        "last_RMSE": last_metrics["RMSE"],
        "last_MAE": last_metrics["MAE"],
        "last_R2": last_metrics["R2"],

        "avg_RMSE": (middle_metrics["RMSE"] + last_metrics["RMSE"]) / 2,
        "avg_MAE": (middle_metrics["MAE"] + last_metrics["MAE"]) / 2,
        "avg_R2": (middle_metrics["R2"] + last_metrics["R2"]) / 2,

        "middle_start": middle_start,
        "middle_end": middle_end,
        "last_start": last_start,
        "last_end": last_end
    }

# =========================================================
# MAIN LOOP
# =========================================================

methods = ["Pure_STL", "Pure_Wavelet"]

all_results = []

for run_name in run_names:

    csv_path = os.path.join(base_dir, run_name, "data.csv")

    if not os.path.exists(csv_path):
        print("Missing:", csv_path)
        continue

    df = pd.read_csv(csv_path)

    if signal_name not in df.columns:
        print(f"{signal_name} not found in {run_name}")
        continue

    signal = df[signal_name].values.astype(float)

    if len(signal) < 15:
        print("Skipping too-short run:", run_name)
        continue

    scaler = MinMaxScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    print("Processing:", run_name, "| samples:", len(signal_scaled))

    for method in methods:

        try:
            result = validate_method(signal_scaled, method)

            all_results.append({
                "run_name": run_name,
                "signal": signal_name,
                "method": method,
                "samples": len(signal_scaled),
                **result
            })

        except Exception as e:
            print(f"Failed {run_name} - {method}: {e}")

# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(all_results)

results_path = f"{save_dir}/pure_stl_wavelet_validation_{signal_name}.csv"
results_df.to_csv(results_path, index=False)

print("\n===== FULL RESULTS =====")
print(results_df)

print("\nSaved full results to:")
print(results_path)

# =========================================================
# SUMMARY
# =========================================================

summary = results_df.groupby("method")[[
    "avg_RMSE",
    "avg_MAE",
    "avg_R2"
]].agg(["mean", "std"])

summary_path = f"{save_dir}/summary_pure_stl_wavelet_{signal_name}.csv"
summary.to_csv(summary_path)

print("\n===== SUMMARY =====")
print(summary)

print("\nSaved summary to:")
print(summary_path)

# =========================================================
# BAR GRAPHS: SAME GRAPH STYLE
# =========================================================

mean_results = results_df.groupby("method")[[
    "avg_RMSE",
    "avg_MAE",
    "avg_R2"
]].mean()

# RMSE and MAE together
plt.figure(figsize=(9, 5))

mean_results[["avg_RMSE", "avg_MAE"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title(f"Pure STL vs Pure Wavelet: RMSE and MAE ({signal_name})")
plt.ylabel("Error")
plt.xlabel("Method")
plt.grid(axis="y")
plt.tight_layout()

plot_path = f"{save_dir}/bar_RMSE_MAE_pure_stl_wavelet_{signal_name}.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved RMSE/MAE plot to:", plot_path)

# R2 graph
plt.figure(figsize=(8, 5))

mean_results["avg_R2"].plot(kind="bar")

plt.title(f"Pure STL vs Pure Wavelet: Average R² ({signal_name})")
plt.ylabel("Average R²")
plt.xlabel("Method")
plt.grid(axis="y")
plt.tight_layout()

plot_path = f"{save_dir}/bar_R2_pure_stl_wavelet_{signal_name}.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved R2 plot to:", plot_path)
