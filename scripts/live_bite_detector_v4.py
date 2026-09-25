from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import serial
import tensorflow as tf


BAUDRATE = 115200
EXPECTED_SENSOR_COLUMNS = 8
SERIAL_TIMEOUT_S = 0.20
CLOSE_BITE_INTERVAL_S = 10.0

MODEL_DIR = Path("models/bite_detector/v4_validated")
MODEL_PATH = MODEL_DIR / "bite_lstm_best.keras"
CONFIG_PATH = MODEL_DIR / "deployment_config.json"
NORMALIZATION_PATH = MODEL_DIR / "bite_normalization_stats.csv"
OUTPUT_ROOT = Path("outputs/live/v4")

RAW_HEADER = ["timestamp_ms", "sample", "ax", "ay", "az", "gx", "gy", "gz"]
PROBABILITY_HEADER = [
    "inference_id",
    "window_start_time_s",
    "window_end_time_s",
    "window_start_timestamp_ms",
    "window_end_timestamp_ms",
    "window_start_sample",
    "window_end_sample",
    "bite_probability",
    "threshold",
    "is_positive_window",
    "positive_run_inferences",
    "new_samples_since_previous_inference",
    "inference_duration_ms",
]
EVENT_HEADER = [
    "event_id",
    "event_time_s",
    "event_timestamp_ms",
    "peak_probability",
    "run_start_time_s",
    "run_end_time_s",
    "run_start_timestamp_ms",
    "run_end_timestamp_ms",
    "run_inferences",
    "peak_window_start_time_s",
    "peak_window_end_time_s",
    "peak_window_start_sample",
    "peak_window_end_sample",
    "event_count",
    "interval_since_previous_bite_s",
    "too_short_bite_interval",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Echte V4-Live-Bite-Erkennung mit Serial-Reader-Thread und "
            "Latest-Window-Scheduling."
        )
    )
    parser.add_argument("--port", required=True, help="COM-Port, z. B. COM3")
    parser.add_argument("--name", required=True, help="Sessionname, z. B. v4_live_01")
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="Dauer in Sekunden; 0 = bis Strg+C. Standard: 60.",
    )
    return parser.parse_args()


def parse_sensor_line(
    line: str,
) -> tuple[float, int, float, float, float, float, float, float] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != EXPECTED_SENSOR_COLUMNS:
        return None

    try:
        timestamp_ms = float(parts[0])
        sample = int(float(parts[1]))
        values = [float(value) for value in parts[2:]]
    except ValueError:
        return None

    return (timestamp_ms, sample, *values)


def load_runtime_artifacts() -> tuple[dict, np.ndarray, np.ndarray, tf.keras.Model]:
    for path in [MODEL_PATH, CONFIG_PATH, NORMALIZATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Benötigte Datei nicht gefunden: {path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    required_keys = [
        "window_size_samples",
        "step_size_samples",
        "feature_order",
        "bite_probability_threshold",
        "min_consecutive_positive_windows",
        "refractory_period_s",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError("Fehlende Config-Werte: " + ", ".join(missing))

    feature_order = [str(feature) for feature in config["feature_order"]]
    expected_feature_order = ["ax", "ay", "az", "gx", "gy", "gz"]
    if feature_order != expected_feature_order:
        raise ValueError(f"Unerwartete Feature-Reihenfolge: {feature_order}")

    stats = pd.read_csv(NORMALIZATION_PATH)
    needed_stats = {"feature", "train_mean", "train_std"}
    if not needed_stats.issubset(stats.columns):
        raise ValueError("Normalisierungsdatei enthält nicht die erwarteten Spalten.")

    stats = stats.set_index("feature").reindex(feature_order)
    if stats[["train_mean", "train_std"]].isna().any().any():
        raise ValueError("Normalisierungswerte fehlen für mindestens ein Feature.")

    mean = stats["train_mean"].to_numpy(dtype=np.float32)
    std = stats["train_std"].to_numpy(dtype=np.float32)
    if np.any(std <= 0):
        raise ValueError("Normalisierungsstandardabweichungen müssen positiv sein.")

    model = tf.keras.models.load_model(MODEL_PATH)

    warmup_input = np.zeros(
        (1, int(config["window_size_samples"]), len(feature_order)),
        dtype=np.float32,
    )
    for _ in range(5):
        _ = model(warmup_input, training=False).numpy()

    return config, mean, std, model


def build_output_paths(session_name: str) -> tuple[Path, Path, Path, Path]:
    safe_name = session_name.strip().replace(" ", "_")
    if not safe_name:
        raise ValueError("--name darf nicht leer sein.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"live_{safe_name}_{timestamp}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    return (
        OUTPUT_ROOT / f"{stem}_raw.csv",
        OUTPUT_ROOT / f"{stem}_probabilities.csv",
        OUTPUT_ROOT / f"{stem}_bite_events.csv",
        OUTPUT_ROOT / f"{stem}_summary.txt",
    )


def finalize_positive_run(
    run_rows: list[dict],
    last_event_peak_time_s: float,
    refractory_period_s: float,
) -> dict | None:
    if not run_rows:
        return None

    peak_row = max(run_rows, key=lambda row: row["bite_probability"])
    peak_time_s = (
        peak_row["window_start_time_s"] + peak_row["window_end_time_s"]
    ) / 2.0

    if peak_time_s - last_event_peak_time_s < refractory_period_s:
        return None

    return {
        "event_time_s": peak_time_s,
        "event_timestamp_ms": (
            peak_row["window_start_timestamp_ms"]
            + peak_row["window_end_timestamp_ms"]
        )
        / 2.0,
        "peak_probability": peak_row["bite_probability"],
        "run_start_time_s": run_rows[0]["window_start_time_s"],
        "run_end_time_s": run_rows[-1]["window_end_time_s"],
        "run_start_timestamp_ms": run_rows[0]["window_start_timestamp_ms"],
        "run_end_timestamp_ms": run_rows[-1]["window_end_timestamp_ms"],
        "run_inferences": len(run_rows),
        "peak_window_start_time_s": peak_row["window_start_time_s"],
        "peak_window_end_time_s": peak_row["window_end_time_s"],
        "peak_window_start_sample": peak_row["window_start_sample"],
        "peak_window_end_sample": peak_row["window_end_sample"],
    }


def main() -> None:
    args = parse_arguments()
    if args.seconds < 0:
        raise ValueError("--seconds darf nicht negativ sein.")

    config, mean, std, model = load_runtime_artifacts()
    window_size = int(config["window_size_samples"])
    step_size = int(config["step_size_samples"])
    threshold = float(config["bite_probability_threshold"])
    min_consecutive = int(config["min_consecutive_positive_windows"])
    refractory_period_s = float(config["refractory_period_s"])

    raw_path, probabilities_path, events_path, summary_path = build_output_paths(args.name)

    buffer_lock = threading.Lock()
    buffer_updated = threading.Event()
    stop_event = threading.Event()
    reader_ready = threading.Event()
    serial_error: list[str] = []

    features: deque[np.ndarray] = deque(maxlen=window_size)
    timestamps_ms: deque[float] = deque(maxlen=window_size)
    sample_ids: deque[int] = deque(maxlen=window_size)
    latest_sequence = 0

    counters = {
        "valid_rows": 0,
        "invalid_rows": 0,
    }
    counter_lock = threading.Lock()

    raw_file = raw_path.open("w", newline="", encoding="utf-8")
    raw_writer = csv.writer(raw_file)
    raw_writer.writerow(RAW_HEADER)

    def serial_reader() -> None:
        nonlocal latest_sequence

        try:
            with serial.Serial(
                args.port,
                BAUDRATE,
                timeout=SERIAL_TIMEOUT_S,
            ) as serial_port:
                time.sleep(2)
                serial_port.reset_input_buffer()
                reader_ready.set()

                while not stop_event.is_set():
                    raw_line = serial_port.readline().decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                    if not raw_line:
                        continue

                    record = parse_sensor_line(raw_line)
                    if record is None:
                        with counter_lock:
                            counters["invalid_rows"] += 1
                        continue

                    timestamp_ms, sample_id, ax, ay, az, gx, gy, gz = record
                    raw_writer.writerow(record)

                    with buffer_lock:
                        features.append(
                            np.asarray([ax, ay, az, gx, gy, gz], dtype=np.float32)
                        )
                        timestamps_ms.append(timestamp_ms)
                        sample_ids.append(sample_id)
                        latest_sequence += 1

                    with counter_lock:
                        counters["valid_rows"] += 1
                    buffer_updated.set()

        except serial.SerialException as error:
            serial_error.append(str(error))
            stop_event.set()
            reader_ready.set()
        finally:
            reader_ready.set()

    reader_thread = threading.Thread(
        target=serial_reader,
        name="serial-reader",
        daemon=True,
    )
    reader_thread.start()

    print(f"Öffne {args.port} bei {BAUDRATE} Baud im Reader-Thread ...")
    print(f"Modell: {MODEL_PATH}")
    print(f"Normalisierung: {NORMALIZATION_PATH}")
    print(f"Rohdaten: {raw_path}")
    print(f"Wahrscheinlichkeiten: {probabilities_path}")
    print(f"Bite-Events: {events_path}")
    print(
        "Zusatzregel: 'BITE INTERVAL TOO SHORT' bei weniger als "
        f"{CLOSE_BITE_INTERVAL_S:.1f} s zwischen zwei Bite-Events."
    )
    print("\nEchte V4-Live-Erkennung gestartet. Beenden mit Strg+C.")

    if not reader_ready.wait(timeout=5):
        raw_file.close()
        raise RuntimeError("Serial-Reader wurde nicht rechtzeitig gestartet.")
    if serial_error:
        raw_file.close()
        raise RuntimeError(f"Serial-Verbindung fehlgeschlagen: {serial_error[0]}")

    start_monotonic = time.monotonic()
    ended_by_keyboard = False
    session_start_timestamp_ms: float | None = None
    last_inferred_sequence = 0
    inference_id = 0
    skipped_inference_opportunities = 0
    event_count = 0
    positive_run: list[dict] = []
    last_event_peak_time_s = -np.inf
    previous_bite_event_time_s: float | None = None
    inference_durations_ms: list[float] = []
    short_bite_intervals = 0

    try:
        with probabilities_path.open("w", newline="", encoding="utf-8") as probability_file, events_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as event_file:
            probability_writer = csv.DictWriter(
                probability_file,
                fieldnames=PROBABILITY_HEADER,
            )
            event_writer = csv.DictWriter(event_file, fieldnames=EVENT_HEADER)
            probability_writer.writeheader()
            event_writer.writeheader()

            while not stop_event.is_set():
                if args.seconds > 0 and time.monotonic() - start_monotonic >= args.seconds:
                    stop_event.set()
                    break

                buffer_updated.wait(timeout=0.10)
                buffer_updated.clear()

                with buffer_lock:
                    current_sequence = latest_sequence
                    if len(features) < window_size:
                        continue
                    if current_sequence - last_inferred_sequence < step_size:
                        continue

                    window_features = np.asarray(features, dtype=np.float32).copy()
                    window_timestamps = np.asarray(timestamps_ms, dtype=np.float64).copy()
                    window_samples = np.asarray(sample_ids, dtype=np.int64).copy()

                new_samples_since_previous_inference = (
                    current_sequence - last_inferred_sequence
                )
                if last_inferred_sequence:
                    skipped_inference_opportunities += max(
                        0,
                        new_samples_since_previous_inference // step_size - 1,
                    )
                last_inferred_sequence = current_sequence

                if session_start_timestamp_ms is None:
                    session_start_timestamp_ms = float(window_timestamps[0])

                normalized_window = (
                    window_features - mean.reshape(1, -1)
                ) / std.reshape(1, -1)
                model_input = normalized_window.reshape(1, window_size, 6).astype(
                    np.float32,
                    copy=False,
                )

                inference_started = time.perf_counter()
                probability = float(model(model_input, training=False).numpy()[0, 0])
                inference_duration_ms = (time.perf_counter() - inference_started) * 1000.0
                inference_durations_ms.append(inference_duration_ms)
                inference_id += 1

                window_start_timestamp_ms = float(window_timestamps[0])
                window_end_timestamp_ms = float(window_timestamps[-1])
                window_start_time_s = (
                    window_start_timestamp_ms - session_start_timestamp_ms
                ) / 1000.0
                window_end_time_s = (
                    window_end_timestamp_ms - session_start_timestamp_ms
                ) / 1000.0
                is_positive = probability >= threshold

                probability_row = {
                    "inference_id": inference_id,
                    "window_start_time_s": round(window_start_time_s, 4),
                    "window_end_time_s": round(window_end_time_s, 4),
                    "window_start_timestamp_ms": round(window_start_timestamp_ms, 3),
                    "window_end_timestamp_ms": round(window_end_timestamp_ms, 3),
                    "window_start_sample": int(window_samples[0]),
                    "window_end_sample": int(window_samples[-1]),
                    "bite_probability": probability,
                    "threshold": threshold,
                    "is_positive_window": is_positive,
                    "positive_run_inferences": (
                        len(positive_run) + 1 if is_positive else len(positive_run)
                    ),
                    "new_samples_since_previous_inference": (
                        new_samples_since_previous_inference
                    ),
                    "inference_duration_ms": round(inference_duration_ms, 2),
                }
                probability_writer.writerow(probability_row)

                if is_positive:
                    positive_run.append(probability_row)
                    continue

                if len(positive_run) >= min_consecutive:
                    event = finalize_positive_run(
                        positive_run,
                        last_event_peak_time_s,
                        refractory_period_s,
                    )
                    if event is not None:
                        event_count += 1
                        event["event_id"] = event_count
                        event["event_count"] = event_count

                        interval_since_previous_bite_s: float | None = None
                        if previous_bite_event_time_s is not None:
                            interval_since_previous_bite_s = (
                                event["event_time_s"] - previous_bite_event_time_s
                            )

                        too_short_bite_interval = (
                            interval_since_previous_bite_s is not None
                            and interval_since_previous_bite_s < CLOSE_BITE_INTERVAL_S
                        )
                        if too_short_bite_interval:
                            short_bite_intervals += 1

                        event["interval_since_previous_bite_s"] = (
                            round(interval_since_previous_bite_s, 3)
                            if interval_since_previous_bite_s is not None
                            else ""
                        )
                        event["too_short_bite_interval"] = too_short_bite_interval

                        event_writer.writerow(event)
                        event_file.flush()

                        last_event_peak_time_s = event["event_time_s"]
                        previous_bite_event_time_s = event["event_time_s"]

                        print(
                            f"BITE #{event_count} | t={event['event_time_s']:.2f} s | "
                            f"p={event['peak_probability']:.4f} | "
                            f"Run={event['run_start_time_s']:.2f}–"
                            f"{event['run_end_time_s']:.2f} s "
                            f"({event['run_inferences']} Inferenzen)"
                        )

                        if too_short_bite_interval:
                            print(
                                "  >>> BITE INTERVAL TOO SHORT: "
                                f"{interval_since_previous_bite_s:.2f} s "
                                f"(< {CLOSE_BITE_INTERVAL_S:.2f} s)"
                            )

                positive_run = []

    except KeyboardInterrupt:
        ended_by_keyboard = True
        stop_event.set()
        print("\nLive-Erkennung durch Benutzer beendet.")
    finally:
        stop_event.set()
        reader_thread.join(timeout=2)
        raw_file.flush()
        raw_file.close()

    elapsed_s = time.monotonic() - start_monotonic
    with counter_lock:
        valid_rows = counters["valid_rows"]
        invalid_rows = counters["invalid_rows"]

    input_rate_hz = valid_rows / elapsed_s if elapsed_s > 0 else 0.0
    inference_rate_hz = inference_id / elapsed_s if elapsed_s > 0 else 0.0
    pending_run_note = "ja" if positive_run else "nein"

    if inference_durations_ms:
        inference_mean_ms = float(np.mean(inference_durations_ms))
        inference_median_ms = float(np.median(inference_durations_ms))
        inference_p95_ms = float(np.percentile(inference_durations_ms, 95))
        inference_max_ms = float(np.max(inference_durations_ms))
    else:
        inference_mean_ms = np.nan
        inference_median_ms = np.nan
        inference_p95_ms = np.nan
        inference_max_ms = np.nan

    summary_lines = [
        "=== LIVE-BITE-DETEKTOR V4 (LATEST-WINDOW) ===",
        f"Session: {args.name}",
        f"Port: {args.port}",
        f"Beendet durch Strg+C: {ended_by_keyboard}",
        f"Laufzeit: {elapsed_s:.2f} s",
        f"Gültige Sensordatenzeilen: {valid_rows}",
        f"Ungültige/übersprungene Serial-Zeilen: {invalid_rows}",
        f"Gemessene Eingangsrate: {input_rate_hz:.2f} Hz",
        f"Modell-Inferenzen: {inference_id}",
        f"Gemessene Inferenzrate: {inference_rate_hz:.2f} Hz",
        f"Übersprungene veraltete Inferenzgelegenheiten: {skipped_inference_opportunities}",
        f"Mittlere Modell-Inferenzzeit: {inference_mean_ms:.2f} ms",
        f"Mediane Modell-Inferenzzeit: {inference_median_ms:.2f} ms",
        f"p95 Modell-Inferenzzeit: {inference_p95_ms:.2f} ms",
        f"Maximale Modell-Inferenzzeit: {inference_max_ms:.2f} ms",
        f"Ausgelöste Bite-Ereignisse: {event_count}",
        f"Bite-Abstände unter {CLOSE_BITE_INTERVAL_S:.1f} s: {short_bite_intervals}",
        f"Noch offener positiver Run beim Beenden: {pending_run_note}",
        "",
        "=== EINGEFRORENE V4-PARAMETER ===",
        f"Fenstergröße: {window_size} Samples",
        f"Schrittweite als Mindestfortschritt: {step_size} Samples",
        f"Threshold: {threshold:.2f}",
        f"Min. positive Live-Inferenzen: {min_consecutive}",
        f"Sperrzeit: {refractory_period_s:.2f} s",
        f"Grenzwert kurzer Bite-Abstand: < {CLOSE_BITE_INTERVAL_S:.1f} s",
        "",
        f"Rohdaten: {raw_path}",
        f"Wahrscheinlichkeiten: {probabilities_path}",
        f"Events: {events_path}",
    ]
    if serial_error:
        summary_lines.extend(["", f"Serieller Fehler: {serial_error[0]}"])

    summary = "\n".join(summary_lines)
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
