# =========================================================
# IMU Window Detection:
# middle line -> crest -> trough -> middle line
# For gx signal across all runs
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

extended_csv_name = "data_wavelet_AR_extended_20cycles.csv"

save_summary_path = os.path.join(
    base_dir,
    "gx_middle_crest_trough_middle_window_summary.csv"
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_previous_middle_crossing(signal, mid_value, peak_idx):
    """
    Find the last index before crest where signal crosses the middle line.
    """
    for i in range(peak_idx - 1, 0, -1):
        y1 = signal[i - 1] - mid_value
        y2 = signal[i] - mid_value

        if y1 == 0:
            return i - 1

        if y1 * y2 <= 0:
            return i

    return 0


def find_next_middle_crossing(signal, mid_value, trough_idx):
    """
    Find the first index after trough where signal crosses the middle line.
    """
    for i in range(trough_idx + 1, len(signal)):
        y1 = signal[i - 1] - mid_value
        y2 = signal[i] - mid_value

        if y2 == 0:
            return i

        if y1 * y2 <= 0:
            return i

    return len(signal) - 1


def detect_middle_crest_trough_middle_window(signal):
    """
    Detect dominant pattern:
    middle -> crest -> trough -> middle
    """

    # Middle line can be mean or median
    mid_value = np.mean(signal)

    # Detect crests and troughs
    crests, _ = find_peaks(signal, distance=3)
    troughs, _ = find_peaks(-signal, distance=3)

    if len(crests) == 0 or len(troughs) == 0:
        return None

    best_pair = None
    best_amplitude = -np.inf

    # Find crest followed by trough
    for crest_idx in crests:
        troughs_after = troughs[troughs > crest_idx]

        if len(troughs_after) == 0:
            continue

        trough_idx = troughs_after[0]

        amplitude = signal[crest_idx] - signal[trough_idx]

        if amplitude > best_amplitude:
            best_amplitude = amplitude
            best_pair = (crest_idx, trough_idx)

    if best_pair is None:
        return None

    crest_idx, trough_idx = best_pair

    # Find middle-line crossings
    start_idx = find_previous_middle_crossing(
        signal,
        mid_value,
        crest_idx
    )

    end_idx = find_next_middle_crossing(
        signal,
        mid_value,
        trough_idx
    )

    window_size = end_idx - start_idx + 1

    return {
        "mid_value": mid_value,
        "crest_idx": crest_idx,
        "trough_idx": trough_idx,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "window_size_samples": window_size,
        "crest_value": signal[crest_idx],
        "trough_value": signal[trough_idx],
        "amplitude": best_amplitude
    }

# =========================================================
# MAIN LOOP
# =========================================================

summary_rows = []

for run_name in run_names:

    run_dir = os.path.join(base_dir, run_name)

    csv_path = os.path.join(run_dir, extended_csv_name)

    if not os.path.exists(csv_path):
        print("Missing:", csv_path)
        continue

    df = pd.read_csv(csv_path)

    if signal_name not in df.columns:
        print("gx missing in:", run_name)
        continue

    signal = df[signal_name].values.astype(float)

    result = detect_middle_crest_trough_middle_window(signal)

    if result is None:
        print("Could not detect window for:", run_name)
        continue

    start_idx = result["start_idx"]
    crest_idx = result["crest_idx"]
    trough_idx = result["trough_idx"]
    end_idx = result["end_idx"]
    mid_value = result["mid_value"]

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(14, 6))

    # Plot real and forecast separately if source exists
    if "source" in df.columns:

        real_df = df[df["source"] == "real"]
        forecast_df = df[df["source"] == "forecast"]

        plt.plot(
            real_df.index,
            real_df[signal_name],
            marker="o",
            color="blue",
            label="Real gx"
        )

        if len(forecast_df) > 0:
            plt.plot(
                forecast_df.index,
                forecast_df[signal_name],
                marker="x",
                color="red",
                linewidth=2,
                label="Forecast gx"
            )

            plt.axvline(
                real_df.index[-1],
                linestyle="--",
                color="black",
                label="Start of extension"
            )

    else:
        plt.plot(
            signal,
            marker="o",
            color="blue",
            label="gx"
        )

    # Middle line
    plt.axhline(
        mid_value,
        linestyle="--",
        color="purple",
        label=f"Middle line = {mid_value:.2f}"
    )

    # Window boundaries
    plt.axvline(
        start_idx,
        linestyle="--",
        color="green",
        label=f"Window start = {start_idx}"
    )

    plt.axvline(
        end_idx,
        linestyle="--",
        color="orange",
        label=f"Window end = {end_idx}"
    )

    # Highlight window
    plt.axvspan(
        start_idx,
        end_idx,
        color="yellow",
        alpha=0.2,
        label=f"Detected window = {result['window_size_samples']} samples"
    )

    # Mark crest and trough
    plt.scatter(
        crest_idx,
        signal[crest_idx],
        s=140,
        color="red",
        label=f"Crest = {crest_idx}"
    )

    plt.scatter(
        trough_idx,
        signal[trough_idx],
        s=140,
        color="black",
        label=f"Trough = {trough_idx}"
    )

    # Draw path: middle -> crest -> trough -> middle
    plt.plot(
        [start_idx, crest_idx, trough_idx, end_idx],
        [signal[start_idx], signal[crest_idx], signal[trough_idx], signal[end_idx]],
        color="magenta",
        linewidth=3,
        label="middle → crest → trough → middle"
    )

    plt.title(
        f"IMU Window Detection: middle → crest → trough → middle\n"
        f"{run_name}, gx, window = {result['window_size_samples']} samples"
    )

    plt.xlabel("Sample Index")
    plt.ylabel("gx")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    save_plot_path = os.path.join(
        run_dir,
        f"gx_window_middle_crest_trough_middle_{run_name}.png"
    )

    plt.savefig(
        save_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_plot_path)

    # =====================================================
    # SAVE SUMMARY ROW
    # =====================================================

    summary_rows.append({
        "run_name": run_name,
        "signal": signal_name,
        "middle_value": mid_value,
        "window_start_middle_idx": start_idx,
        "crest_idx": crest_idx,
        "trough_idx": trough_idx,
        "window_end_middle_idx": end_idx,
        "window_size_samples": result["window_size_samples"],
        "crest_value": result["crest_value"],
        "trough_value": result["trough_value"],
        "amplitude": result["amplitude"],
        "plot_path": save_plot_path
    })

# =========================================================
# SAVE SUMMARY CSV
# =========================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    save_summary_path,
    index=False
)

print("\n===== SUMMARY =====")
print(summary_df)

print("\nSaved summary CSV to:")
print(save_summary_path)
