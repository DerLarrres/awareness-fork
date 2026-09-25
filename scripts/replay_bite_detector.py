from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_DIR = Path("models/bite_detector/v1_validated")
MODEL_PATH = MODEL_DIR / "bite_lstm_best.keras"
CONFIG_PATH = MODEL_DIR / "deployment_config.json"
NORMALIZATION_PATH = MODEL_DIR / "bite_normalization_stats.csv"
DEFAULT_OUTPUT_DIR = Path("outputs/live_replay")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sampleweises CSV-Replay mit derselben Eventlogik wie in test_bite_lstm.py."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {source_name}: {', '.join(missing)}")


def main() -> None:
    args = parse_arguments()
    for path in [MODEL_PATH, CONFIG_PATH, NORMALIZATION_PATH, args.input_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    feature_order = [str(x) for x in config["feature_order"]]
    window_size = int(config["window_size_samples"])
    step_size = int(config["step_size_samples"])
    threshold = float(config["bite_probability_threshold"])
    min_consecutive = int(config["min_consecutive_positive_windows"])
    refractory_s = float(config["refractory_period_s"])

    sensor = pd.read_csv(args.input_csv)
    require_columns(sensor, ["timestamp_ms", "sample"] + feature_order, args.input_csv.name)
    for column in ["timestamp_ms", "sample"] + feature_order:
        sensor[column] = pd.to_numeric(sensor[column], errors="coerce")
    if sensor[["timestamp_ms", "sample"] + feature_order].isna().any().any():
        raise ValueError("Die Sensor-CSV enthält ungültige oder fehlende Werte.")
    if not sensor["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("timestamp_ms muss aufsteigend sein.")

    stats = pd.read_csv(NORMALIZATION_PATH).set_index("feature").reindex(feature_order)
    require_columns(stats.reset_index(), ["feature", "train_mean", "train_std"], NORMALIZATION_PATH.name)
    mean = stats["train_mean"].to_numpy(dtype=np.float32)
    std = stats["train_std"].to_numpy(dtype=np.float32)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Ungültige Normalisierungswerte.")

    model = tf.keras.models.load_model(MODEL_PATH)
    values = sensor[feature_order].to_numpy(dtype=np.float32)
    timestamps_ms = sensor["timestamp_ms"].to_numpy(dtype=np.float64)
    samples = sensor["sample"].to_numpy(dtype=np.int64)
    start_timestamp_ms = timestamps_ms[0]

    probability_rows = []
    for end_index in range(window_size - 1, len(sensor), step_size):
        start_index = end_index - window_size + 1
        raw_window = values[start_index : end_index + 1]
        normalized_window = (raw_window - mean.reshape(1, -1)) / std.reshape(1, -1)
        probability = float(model.predict(normalized_window[None, ...], verbose=0)[0, 0])
        probability_rows.append(
            {
                "window_index": len(probability_rows),
                "window_start_time_s": (timestamps_ms[start_index] - start_timestamp_ms) / 1000.0,
                "window_end_time_s": (timestamps_ms[end_index] - start_timestamp_ms) / 1000.0,
                "window_start_sample": int(samples[start_index]),
                "window_end_sample": int(samples[end_index]),
                "bite_probability": probability,
            }
        )

    probabilities = pd.DataFrame(probability_rows)
    is_positive = probabilities["bite_probability"].to_numpy() >= threshold

    candidates = []
    index = 0
    while index < len(probabilities):
        if not is_positive[index]:
            index += 1
            continue
        run_start = index
        while index < len(probabilities) and is_positive[index]:
            index += 1
        run_end = index
        if run_end - run_start < min_consecutive:
            continue
        run = probabilities.iloc[run_start:run_end]
        peak_row = run.loc[run["bite_probability"].idxmax()]
        candidates.append(
            {
                "run_start_time_s": float(run.iloc[0]["window_start_time_s"]),
                "run_end_time_s": float(run.iloc[-1]["window_end_time_s"]),
                "run_windows": int(len(run)),
                "peak_time_s": float((peak_row["window_start_time_s"] + peak_row["window_end_time_s"]) / 2.0),
                "peak_probability": float(peak_row["bite_probability"]),
                "peak_window_start_time_s": float(peak_row["window_start_time_s"]),
                "peak_window_end_time_s": float(peak_row["window_end_time_s"]),
            }
        )

    events = []
    for candidate in candidates:
        if not events:
            events.append(candidate)
            continue
        previous = events[-1]
        if candidate["peak_time_s"] - previous["peak_time_s"] < refractory_s:
            if candidate["peak_probability"] > previous["peak_probability"]:
                events[-1] = candidate
        else:
            events.append(candidate)

    for event_id, event in enumerate(events, start=1):
        event["event_id"] = event_id
        print(
            f"BITE #{event_id} | t={event['peak_time_s']:.2f} s | "
            f"p={event['peak_probability']:.4f} | "
            f"Run={event['run_start_time_s']:.2f}–{event['run_end_time_s']:.2f} s "
            f"({event['run_windows']} Fenster)"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_csv.stem
    probabilities["threshold"] = threshold
    probabilities["is_positive_window"] = is_positive
    probabilities.to_csv(output_dir / f"{stem}_bite_probabilities.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(events).to_csv(output_dir / f"{stem}_bite_events.csv", index=False, encoding="utf-8-sig")

    summary = "\n".join(
        [
            "=== BITE-DETEKTOR V1: CSV-REPLAY ===",
            f"Input: {args.input_csv}",
            f"Fenster-Inferenzen: {len(probabilities)}",
            f"Threshold: {threshold:.2f}",
            f"Min. positive Folgefenster: {min_consecutive}",
            f"Sperrzeit: {refractory_s:.2f} s",
            f"Ausgelöste Bite-Ereignisse: {len(events)}",
        ]
    )
    (output_dir / f"{stem}_bite_replay_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
