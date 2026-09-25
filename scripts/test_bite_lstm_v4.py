from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

MODEL_DIR = Path("models/bite_detector/v4_validated")
MODEL_PATH = MODEL_DIR / "bite_lstm_best.keras"
CONFIG_PATH = MODEL_DIR / "deployment_config.json"
NORMALIZATION_PATH = MODEL_DIR / "bite_normalization_stats.csv"

RAW_WINDOWS_PATH = Path("data/processed/bite_windows_50_step_5.npz")
METADATA_PATH = Path("data/processed/bite_windows_50_step_5_metadata.csv")
OUTPUT_DIR = Path("models/bite_detector/v4_test")

MATCH_TOLERANCE_S = 0.50


def require_columns(frame: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {source_name}: {', '.join(missing)}")


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision_bite": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_bite": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_bite": float(f1_score(y_true, prediction, zero_division=0)),
    }


def detect_events(
    session: pd.DataFrame,
    threshold: float,
    min_consecutive_windows: int,
    refractory_period_s: float,
) -> list[dict]:
    session = session.sort_values("window_start_time_s").reset_index(drop=True)
    probabilities = session["bite_probability"].to_numpy(dtype=float)
    starts = session["window_start_time_s"].to_numpy(dtype=float)
    ends = session["window_end_time_s"].to_numpy(dtype=float)
    above = probabilities >= threshold

    candidates = []
    index = 0
    while index < len(session):
        if not above[index]:
            index += 1
            continue

        run_start = index
        while index < len(session) and above[index]:
            index += 1
        run_end = index

        if run_end - run_start < min_consecutive_windows:
            continue

        run_probabilities = probabilities[run_start:run_end]
        peak_index = run_start + int(np.argmax(run_probabilities))
        candidates.append(
            {
                "run_start_time_s": float(starts[run_start]),
                "run_end_time_s": float(ends[run_end - 1]),
                "run_windows": int(run_end - run_start),
                "peak_time_s": float((starts[peak_index] + ends[peak_index]) / 2.0),
                "peak_probability": float(probabilities[peak_index]),
            }
        )

    events = []
    for candidate in candidates:
        if not events:
            events.append(candidate)
            continue

        previous = events[-1]
        if candidate["peak_time_s"] - previous["peak_time_s"] < refractory_period_s:
            if candidate["peak_probability"] > previous["peak_probability"]:
                events[-1] = candidate
        else:
            events.append(candidate)
    return events


def bite_intervals(session: pd.DataFrame) -> list[dict]:
    bite_rows = session.loc[session["actual_label"] == "bite"].copy()
    if bite_rows.empty:
        return []

    bite_rows = bite_rows.sort_values("window_start_time_s").reset_index(drop=True)
    intervals = []
    current_start = float(bite_rows.loc[0, "window_start_time_s"])
    current_end = float(bite_rows.loc[0, "window_end_time_s"])

    for row in bite_rows.iloc[1:].itertuples(index=False):
        start = float(row.window_start_time_s)
        end = float(row.window_end_time_s)
        if start <= current_end + 0.15:
            current_end = max(current_end, end)
        else:
            intervals.append(
                {
                    "bite_start_time_s": current_start,
                    "bite_end_time_s": current_end,
                    "bite_center_time_s": (current_start + current_end) / 2.0,
                }
            )
            current_start, current_end = start, end

    intervals.append(
        {
            "bite_start_time_s": current_start,
            "bite_end_time_s": current_end,
            "bite_center_time_s": (current_start + current_end) / 2.0,
        }
    )
    return intervals


def event_matches_interval(event: dict, interval: dict) -> bool:
    event_start = event["run_start_time_s"]
    event_end = event["run_end_time_s"]
    interval_start = interval["bite_start_time_s"] - MATCH_TOLERANCE_S
    interval_end = interval["bite_end_time_s"] + MATCH_TOLERANCE_S
    return event_end >= interval_start and event_start <= interval_end


def match_events(events: list[dict], intervals: list[dict]) -> tuple[list[dict], set[int], set[int]]:
    candidates = []
    for event_index, event in enumerate(events):
        for interval_index, interval in enumerate(intervals):
            if event_matches_interval(event, interval):
                distance = abs(event["peak_time_s"] - interval["bite_center_time_s"])
                candidates.append((distance, event_index, interval_index))

    pairs = []
    used_events = set()
    used_intervals = set()
    for distance, event_index, interval_index in sorted(candidates):
        if event_index in used_events or interval_index in used_intervals:
            continue
        used_events.add(event_index)
        used_intervals.add(interval_index)
        pairs.append(
            {
                "event_index": event_index,
                "bite_interval_index": interval_index,
                "time_error_s": events[event_index]["peak_time_s"]
                - intervals[interval_index]["bite_center_time_s"],
                "absolute_time_error_s": distance,
            }
        )
    return pairs, used_events, used_intervals


def main() -> None:
    for required_path in [MODEL_PATH, CONFIG_PATH, NORMALIZATION_PATH, RAW_WINDOWS_PATH, METADATA_PATH]:
        if not required_path.exists():
            raise FileNotFoundError(f"Benötigte Datei nicht gefunden: {required_path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    required_config = [
        "window_size_samples",
        "step_size_samples",
        "sampling_rate_hz",
        "feature_order",
        "bite_probability_threshold",
        "min_consecutive_positive_windows",
        "refractory_period_s",
    ]
    missing_config = [key for key in required_config if key not in config]
    if missing_config:
        raise ValueError("Fehlende Config-Werte: " + ", ".join(missing_config))

    threshold = float(config["bite_probability_threshold"])
    min_consecutive = int(config["min_consecutive_positive_windows"])
    refractory_period_s = float(config["refractory_period_s"])
    expected_features = [str(name) for name in config["feature_order"]]

    data = np.load(RAW_WINDOWS_PATH, allow_pickle=False)
    required_keys = {"X", "y", "feature_names", "class_names", "window_size", "step_size"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise ValueError("Fehlende NPZ-Keys: " + ", ".join(sorted(missing_keys)))

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = [str(name) for name in data["feature_names"]]
    class_names = [str(name) for name in data["class_names"]]

    if feature_names != expected_features:
        raise ValueError(f"Feature-Reihenfolge stimmt nicht: {feature_names} vs. {expected_features}")
    if class_names != ["not_bite", "bite"]:
        raise ValueError(f"Unerwartete Klassen: {class_names}")
    if int(data["window_size"]) != int(config["window_size_samples"]):
        raise ValueError("Fenstergröße der Daten stimmt nicht mit deployment_config.json überein.")
    if int(data["step_size"]) != int(config["step_size_samples"]):
        raise ValueError("Schrittweite der Daten stimmt nicht mit deployment_config.json überein.")

    metadata = pd.read_csv(METADATA_PATH)
    require_columns(
        metadata,
        ["file", "split", "binary_label", "window_start_time_s", "window_end_time_s"],
        METADATA_PATH.name,
    )
    if len(metadata) != len(X) or len(y) != len(X):
        raise ValueError("X, y und Metadaten haben unterschiedliche Längen.")

    test_mask = metadata["split"].astype(str).to_numpy() == "test"
    if not test_mask.any():
        raise ValueError("Keine Test-Fenster gefunden. Prüfe splits.csv und erstelle die Bite-Windows neu.")

    X_test_raw = X[test_mask]
    y_test = y[test_mask]
    meta_test = metadata.loc[test_mask].reset_index(drop=True).copy()

    stats = pd.read_csv(NORMALIZATION_PATH)
    require_columns(stats, ["feature", "train_mean", "train_std"], NORMALIZATION_PATH.name)
    stats = stats.set_index("feature").reindex(expected_features)
    if stats[["train_mean", "train_std"]].isna().any().any():
        raise ValueError("Normalisierungsstatistik enthält fehlende Features oder Werte.")

    mean = stats["train_mean"].to_numpy(dtype=np.float32)
    std = stats["train_std"].to_numpy(dtype=np.float32)
    if np.any(std <= 0):
        raise ValueError("Normalisierungsstandardabweichung muss positiv sein.")

    X_test = ((X_test_raw - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)

    model = tf.keras.models.load_model(MODEL_PATH)
    probability = model.predict(X_test, batch_size=64, verbose=0).reshape(-1)
    prediction = (probability >= threshold).astype(np.int64)

    window_metrics = binary_metrics(y_test, probability, threshold)
    window_metrics["roc_auc"] = float(roc_auc_score(y_test, probability)) if len(np.unique(y_test)) == 2 else np.nan
    window_metrics["average_precision"] = (
        float(average_precision_score(y_test, probability)) if len(np.unique(y_test)) == 2 else np.nan
    )
    cm = confusion_matrix(y_test, prediction, labels=[0, 1])

    predictions = meta_test.copy()
    predictions["actual_label_id"] = y_test
    predictions["actual_label"] = np.where(y_test == 1, "bite", "not_bite")
    predictions["bite_probability"] = probability
    predictions["threshold"] = threshold
    predictions["predicted_label_id"] = prediction
    predictions["predicted_label"] = np.where(prediction == 1, "bite", "not_bite")

    event_rows = []
    ground_truth_rows = []
    session_rows = []

    for file_name, session in predictions.groupby("file", sort=True):
        events = detect_events(session, threshold, min_consecutive, refractory_period_s)
        intervals = bite_intervals(session)
        pairs, matched_event_indices, matched_interval_indices = match_events(events, intervals)
        pair_by_event = {pair["event_index"]: pair for pair in pairs}
        pair_by_interval = {pair["bite_interval_index"]: pair for pair in pairs}

        for event_index, event in enumerate(events):
            pair = pair_by_event.get(event_index)
            event_rows.append(
                {
                    "file": file_name,
                    "event_index": event_index,
                    **event,
                    "matched": event_index in matched_event_indices,
                    "matched_bite_interval_index": pair["bite_interval_index"] if pair else pd.NA,
                    "time_error_s": pair["time_error_s"] if pair else np.nan,
                    "absolute_time_error_s": pair["absolute_time_error_s"] if pair else np.nan,
                }
            )

        for interval_index, interval in enumerate(intervals):
            pair = pair_by_interval.get(interval_index)
            ground_truth_rows.append(
                {
                    "file": file_name,
                    "bite_interval_index": interval_index,
                    **interval,
                    "matched": interval_index in matched_interval_indices,
                    "matched_event_index": pair["event_index"] if pair else pd.NA,
                    "time_error_s": pair["time_error_s"] if pair else np.nan,
                    "absolute_time_error_s": pair["absolute_time_error_s"] if pair else np.nan,
                }
            )

        tp = len(pairs)
        fp = len(events) - tp
        fn = len(intervals) - tp
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        errors = [pair["absolute_time_error_s"] for pair in pairs]
        session_rows.append(
            {
                "file": file_name,
                "true_bite_events": len(intervals),
                "predicted_events": len(events),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "event_precision": precision,
                "event_recall": recall,
                "event_f1": f1,
                "mean_absolute_time_error_s": float(np.mean(errors)) if errors else np.nan,
                "median_absolute_time_error_s": float(np.median(errors)) if errors else np.nan,
            }
        )

    events_table = pd.DataFrame(event_rows)
    ground_truth_table = pd.DataFrame(ground_truth_rows)
    session_summary = pd.DataFrame(session_rows)

    total_tp = int(session_summary["true_positives"].sum())
    total_fp = int(session_summary["false_positives"].sum())
    total_fn = int(session_summary["false_negatives"].sum())
    total_true_events = int(session_summary["true_bite_events"].sum())
    total_predicted_events = int(session_summary["predicted_events"].sum())
    event_precision = safe_ratio(total_tp, total_tp + total_fp)
    event_recall = safe_ratio(total_tp, total_tp + total_fn)
    event_f1 = safe_ratio(2 * event_precision * event_recall, event_precision + event_recall)
    matched_errors = events_table.loc[events_table["matched"], "absolute_time_error_s"].dropna()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "bite_lstm_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        cm,
        index=["actual_not_bite", "actual_bite"],
        columns=["pred_not_bite", "pred_bite"],
    ).to_csv(OUTPUT_DIR / "bite_lstm_test_confusion_matrix.csv", encoding="utf-8-sig")
    events_table.to_csv(OUTPUT_DIR / "bite_event_test_predictions.csv", index=False, encoding="utf-8-sig")
    ground_truth_table.to_csv(OUTPUT_DIR / "bite_event_test_ground_truth.csv", index=False, encoding="utf-8-sig")
    session_summary.to_csv(OUTPUT_DIR / "bite_event_test_session_summary.csv", index=False, encoding="utf-8-sig")

    report = classification_report(
        y_test,
        prediction,
        labels=[0, 1],
        target_names=["not_bite", "bite"],
        digits=4,
        zero_division=0,
    )
    summary_lines = [
        "=== FINALER BITE-DETEKTOR-TEST ===",
        "Keine Trainings- oder Parameteranpassung wurde in diesem Skript vorgenommen.",
        "",
        "=== EINGEFRORENE DEPLOYMENT-PARAMETER ===",
        f"Modell: {MODEL_PATH}",
        f"Window size: {config['window_size_samples']} Samples",
        f"Step size: {config['step_size_samples']} Samples",
        f"Sampling rate: {config['sampling_rate_hz']} Hz",
        f"Threshold: {threshold:.2f}",
        f"Min. consecutive positive windows: {min_consecutive}",
        f"Refractory period: {refractory_period_s:.2f} s",
        f"Event matching tolerance (evaluation only): ±{MATCH_TOLERANCE_S:.2f} s",
        "",
        "=== TEST-FENSTER ===",
        f"Anzahl Test-Fenster: {len(y_test)}",
        f"Not-bite-Fenster: {int(np.sum(y_test == 0))}",
        f"Bite-Fenster: {int(np.sum(y_test == 1))}",
        f"ROC-AUC: {window_metrics['roc_auc']:.4f}",
        f"Average Precision / PR-AUC: {window_metrics['average_precision']:.4f}",
        f"Accuracy: {window_metrics['accuracy']:.4f}",
        f"Bite Precision: {window_metrics['precision_bite']:.4f}",
        f"Bite Recall: {window_metrics['recall_bite']:.4f}",
        f"Bite F1: {window_metrics['f1_bite']:.4f}",
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        np.array2string(cm),
        "",
        "Classification report:",
        report,
        "",
        "=== TEST-EREIGNISSE ===",
        f"Echte Bite-Ereignisse: {total_true_events}",
        f"Vorhergesagte Ereignisse: {total_predicted_events}",
        f"True Positives: {total_tp}",
        f"False Positives: {total_fp}",
        f"False Negatives: {total_fn}",
        f"Event Precision: {event_precision:.4f}",
        f"Event Recall: {event_recall:.4f}",
        f"Event F1: {event_f1:.4f}",
        f"Mittlerer absoluter Zeitfehler: {matched_errors.mean():.3f} s" if len(matched_errors) else "Mittlerer absoluter Zeitfehler: n/a",
        f"Median absoluter Zeitfehler: {matched_errors.median():.3f} s" if len(matched_errors) else "Median absoluter Zeitfehler: n/a",
        "",
        "=== EREIGNISSE PRO TEST-SESSION ===",
        session_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
    ]
    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "bite_lstm_test_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print("\nGespeicherte Test-Artefakte:")
    for name in [
        "bite_lstm_test_summary.txt",
        "bite_lstm_test_confusion_matrix.csv",
        "bite_lstm_test_predictions.csv",
        "bite_event_test_session_summary.csv",
        "bite_event_test_predictions.csv",
        "bite_event_test_ground_truth.csv",
    ]:
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
