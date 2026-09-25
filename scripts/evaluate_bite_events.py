from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_PATH = Path("models/bite_detector/v4_validated/bite_lstm_val_predictions.csv")
OUTPUT_DIR = Path("models/bite_detector/v4_validated")

# Event-Regel für die Live-Anwendung und die Event-basierte Validation.
THRESHOLD = 0.9  # None = den beim Training gespeicherten Schwellenwert verwenden.
MIN_CONSECUTIVE_WINDOWS = 3
REFRACTORY_PERIOD_S = 1.20
MATCH_TOLERANCE_S = 0.50


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {name}: {', '.join(missing)}")


def detect_events(session: pd.DataFrame, threshold: float) -> list[dict]:
    session = session.sort_values("window_start_time_s").reset_index(drop=True).copy()
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

        if run_end - run_start < MIN_CONSECUTIVE_WINDOWS:
            continue

        run_probabilities = probabilities[run_start:run_end]
        peak_offset = int(np.argmax(run_probabilities))
        peak_index = run_start + peak_offset
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
        if candidate["peak_time_s"] - previous["peak_time_s"] < REFRACTORY_PERIOD_S:
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
    pairs = []
    used_events = set()
    used_intervals = set()

    possible_pairs = []
    for event_index, event in enumerate(events):
        for interval_index, interval in enumerate(intervals):
            if event_matches_interval(event, interval):
                distance = abs(event["peak_time_s"] - interval["bite_center_time_s"])
                possible_pairs.append((distance, event_index, interval_index))

    for distance, event_index, interval_index in sorted(possible_pairs):
        if event_index in used_events or interval_index in used_intervals:
            continue
        used_events.add(event_index)
        used_intervals.add(interval_index)
        pair = {
            "event_index": event_index,
            "bite_interval_index": interval_index,
            "time_error_s": events[event_index]["peak_time_s"] - intervals[interval_index]["bite_center_time_s"],
            "absolute_time_error_s": distance,
        }
        pairs.append(pair)

    return pairs, used_events, used_intervals


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions-Datei nicht gefunden: {PREDICTIONS_PATH}")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    require_columns(
        predictions,
        [
            "file",
            "window_start_time_s",
            "window_end_time_s",
            "actual_label",
            "bite_probability",
            "threshold",
        ],
        PREDICTIONS_PATH.name,
    )

    threshold = float(predictions["threshold"].iloc[0]) if THRESHOLD is None else float(THRESHOLD)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Ungültiger Threshold: {threshold}")

    all_event_rows = []
    all_interval_rows = []
    session_rows = []

    for file_name, session in predictions.groupby("file", sort=True):
        events = detect_events(session, threshold)
        intervals = bite_intervals(session)
        pairs, matched_event_indices, matched_interval_indices = match_events(events, intervals)
        pair_by_event = {pair["event_index"]: pair for pair in pairs}
        pair_by_interval = {pair["bite_interval_index"]: pair for pair in pairs}

        for event_index, event in enumerate(events):
            pair = pair_by_event.get(event_index)
            all_event_rows.append(
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
            all_interval_rows.append(
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

        true_positives = len(pairs)
        false_positives = len(events) - true_positives
        false_negatives = len(intervals) - true_positives
        precision = safe_ratio(true_positives, true_positives + false_positives)
        recall = safe_ratio(true_positives, true_positives + false_negatives)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        errors = [pair["absolute_time_error_s"] for pair in pairs]

        session_rows.append(
            {
                "file": file_name,
                "true_bite_events": len(intervals),
                "predicted_events": len(events),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "event_precision": precision,
                "event_recall": recall,
                "event_f1": f1,
                "mean_absolute_time_error_s": float(np.mean(errors)) if errors else np.nan,
                "median_absolute_time_error_s": float(np.median(errors)) if errors else np.nan,
            }
        )

    event_table = pd.DataFrame(all_event_rows)
    interval_table = pd.DataFrame(all_interval_rows)
    session_summary = pd.DataFrame(session_rows)

    total_tp = int(session_summary["true_positives"].sum())
    total_fp = int(session_summary["false_positives"].sum())
    total_fn = int(session_summary["false_negatives"].sum())
    total_true = int(session_summary["true_bite_events"].sum())
    total_predicted = int(session_summary["predicted_events"].sum())
    precision = safe_ratio(total_tp, total_tp + total_fp)
    recall = safe_ratio(total_tp, total_tp + total_fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    matched_errors = event_table.loc[event_table["matched"], "absolute_time_error_s"].dropna()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_table.to_csv(OUTPUT_DIR / "bite_event_predictions.csv", index=False, encoding="utf-8-sig")
    interval_table.to_csv(OUTPUT_DIR / "bite_event_ground_truth.csv", index=False, encoding="utf-8-sig")
    session_summary.to_csv(OUTPUT_DIR / "bite_event_session_summary.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "=== EVENT-BASIERTE BITE-VALIDATION ===",
        f"Threshold: {threshold:.2f}",
        f"Min. aufeinanderfolgende positive Fenster: {MIN_CONSECUTIVE_WINDOWS}",
        f"Sperrzeit: {REFRACTORY_PERIOD_S:.2f} s",
        f"Matching-Toleranz: ±{MATCH_TOLERANCE_S:.2f} s",
        "",
        f"Echte Bite-Ereignisse: {total_true}",
        f"Vorhergesagte Ereignisse: {total_predicted}",
        f"True Positives: {total_tp}",
        f"False Positives: {total_fp}",
        f"False Negatives: {total_fn}",
        f"Event Precision: {precision:.4f}",
        f"Event Recall: {recall:.4f}",
        f"Event F1: {f1:.4f}",
        f"Mittlerer absoluter Zeitfehler: {matched_errors.mean():.3f} s" if len(matched_errors) else "Mittlerer absoluter Zeitfehler: n/a",
        f"Median absoluter Zeitfehler: {matched_errors.median():.3f} s" if len(matched_errors) else "Median absoluter Zeitfehler: n/a",
        "",
        "=== PRO SESSION ===",
        session_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
    ]
    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "bite_event_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print("\nGespeichert:")
    print(OUTPUT_DIR / "bite_event_summary.txt")
    print(OUTPUT_DIR / "bite_event_session_summary.csv")
    print(OUTPUT_DIR / "bite_event_predictions.csv")
    print(OUTPUT_DIR / "bite_event_ground_truth.csv")


if __name__ == "__main__":
    main()
