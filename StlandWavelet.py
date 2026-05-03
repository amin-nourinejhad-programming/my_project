# =========================================================
# Compute required IMU samples for ITTC wave-cycle criterion
# For all selected tank-test runs
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================

base_dir = "/content/drive/MyDrive/Final_Manual_Dataset_Augmented"

save_dir = "/content/drive/MyDrive/ittc_sample_requirement"
os.makedirs(save_dir, exist_ok=True)

required_wave_cycles = 10

# Manually define wave periods from the tank-test matrix
# Format:
# RunName : Wave period [seconds]

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
# MAIN LOOP
# =========================================================

results = []

for run_name, wave_period in wave_periods.items():

    csv_path = os.path.join(base_dir, run_name, "data.csv")

    if not os.path.exists(csv_path):
        print("Missing:", csv_path)
        continue

    df = pd.read_csv(csv_path)

    if "timestamp_iso_ms" not in df.columns:
        print(f"timestamp_iso_ms missing in {run_name}")
        continue

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp_iso_ms"])

    # Estimate IMU frequency
    time_diffs = df["timestamp"].diff().dropna()
    time_diffs_sec = time_diffs.dt.total_seconds()

    avg_dt = time_diffs_sec.mean()
    median_dt = time_diffs_sec.median()

    imu_freq_mean = 1 / avg_dt
    imu_freq_median = 1 / median_dt

    # Current samples
    current_samples = len(df)

    # Required samples according to:
    # N_required = required_wave_cycles * wave_period * imu_frequency
    required_samples_mean = int(np.ceil(
        required_wave_cycles * wave_period * imu_freq_mean
    ))

    required_samples_median = int(np.ceil(
        required_wave_cycles * wave_period * imu_freq_median
    ))

    # Missing samples
    missing_samples_mean = max(0, required_samples_mean - current_samples)
    missing_samples_median = max(0, required_samples_median - current_samples)

    results.append({
        "run_name": run_name,
        "wave_period_sec": wave_period,
        "required_wave_cycles": required_wave_cycles,

        "current_samples": current_samples,

        "avg_dt_sec": avg_dt,
        "median_dt_sec": median_dt,

        "imu_freq_mean_Hz": imu_freq_mean,
        "imu_freq_median_Hz": imu_freq_median,

        "required_samples_mean_freq": required_samples_mean,
        "required_samples_median_freq": required_samples_median,

        "missing_samples_mean_freq": missing_samples_mean,
        "missing_samples_median_freq": missing_samples_median
    })

# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n===== ITTC SAMPLE REQUIREMENT RESULTS =====")
print(results_df)

save_path = os.path.join(save_dir, "ittc_required_samples_all_runs.csv")
results_df.to_csv(save_path, index=False)

print("\nSaved results to:")
print(save_path)

# =========================================================
# SIMPLE BAR PLOT: CURRENT VS REQUIRED SAMPLES
# =========================================================

plt.figure(figsize=(14, 6))

x = np.arange(len(results_df))

plt.bar(
    x - 0.2,
    results_df["current_samples"],
    width=0.4,
    label="Current samples"
)

plt.bar(
    x + 0.2,
    results_df["required_samples_mean_freq"],
    width=0.4,
    label="Required samples"
)

plt.xticks(
    x,
    results_df["run_name"],
    rotation=90
)

plt.ylabel("Number of samples")
plt.title("Current vs ITTC-required IMU samples")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

plot_path = os.path.join(save_dir, "current_vs_required_samples.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved plot to:")
print(plot_path)

# =========================================================
# BAR PLOT: MISSING SAMPLES
# =========================================================

plt.figure(figsize=(14, 6))

plt.bar(
    results_df["run_name"],
    results_df["missing_samples_mean_freq"]
)

plt.xticks(rotation=90)

plt.ylabel("Missing samples")
plt.title("Additional IMU samples required to satisfy ITTC criterion")
plt.grid(axis="y")
plt.tight_layout()

plot_path = os.path.join(save_dir, "missing_samples_per_run.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved plot to:")
print(plot_path)
