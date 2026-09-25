from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_FILE = Path("data/metadata/splits.csv")
OUTPUT_DIR = Path("data/processed")

FEATURE_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
CLASS_NAMES = np.array(["not_bite", "bite"])

WINDOW_SIZE = 50
STEP_SIZE = 5
MIN_LABEL_PURITY = 0.80
TIME_TOLERANCE_S = 0.10


def read_splits() -> pd.DataFrame:
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"Split-Datei nicht gefunden: {SPLITS_FILE}")

    splits = pd.read_csv(SPLITS_FILE)
    required_columns = [
        "file",
        "path",
        "source",
        "split",
        "annotation_file",
        "annotation_path",
    ]
    missing_columns = [
        column for column in required_columns if column not in splits.columns
    ]
    if missing_columns:
        raise ValueError(
            "Fehlende Spalten in splits.csv: " + ", ".join(missing_columns)
        )

    splits = splits.copy()
    for column in required_columns:
        splits[column] = splits[column].fillna("").astype(str).str.strip()

    invalid_splits = sorted(set(splits["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError("Ungültige Split-Namen: " + ", ".join(invalid_splits))

    if splits["file"].duplicated().any():
        duplicate_files = sorted(
            splits.loc[splits["file"].duplicated(keep=False), "file"].unique()
        )
        raise ValueError("Dateien kommen mehrfach vor: " + ", ".join(duplicate_files))

    if (splits["source"] != "continuous").any():
        invalid_sources = sorted(
            splits.loc[splits["source"] != "continuous", "source"].unique()
        )
        raise ValueError(
            "Dieses Skript verarbeitet nur source=continuous. Gefunden: "
            + ", ".join(invalid_sources)
        )

    return splits


def read_sensor(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sensor-CSV nicht gefunden: {path}")

    sensor = pd.read_csv(path)
    required_columns = ["timestamp_ms", "sample"] + FEATURE_COLUMNS
    missing_columns = [
        column for column in required_columns if column not in sensor.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Fehlende Sensor-Spalten in {path.name}: "
            + ", ".join(missing_columns)
        )

    sensor = sensor.copy()
    for column in required_columns:
        sensor[column] = pd.to_numeric(sensor[column], errors="coerce")

    if sensor[required_columns].isna().any().any():
        invalid_columns = sensor[required_columns].columns[
            sensor[required_columns].isna().any()
        ].tolist()
        raise ValueError(
            f"Ungültige Sensorwerte in {path.name}: " + ", ".join(invalid_columns)
        )

    if len(sensor) < WINDOW_SIZE:
        raise ValueError(
            f"Zu kurze Session in {path.name}: {len(sensor)} Samples"
        )

    if not sensor["timestamp_ms"].is_monotonic_increasing:
        raise ValueError(f"Zeitstempel sind nicht aufsteigend: {path.name}")

    return sensor


def read_annotations(path: Path, sensor_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotationsdatei fehlt für {sensor_name}: {path}")

    annotations = pd.read_csv(path)
    required_columns = ["start_time_s", "end_time_s", "label"]
    missing_columns = [
        column for column in required_columns if column not in annotations.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Fehlende Annotationsspalten in {path.name}: "
            + ", ".join(missing_columns)
        )

    annotations = annotations.copy()
    annotations["start_time_s"] = pd.to_numeric(
        annotations["start_time_s"], errors="coerce"
    )
    annotations["end_time_s"] = pd.to_numeric(
        annotations["end_time_s"], errors="coerce"
    )
    annotations["label"] = (
        annotations["label"].fillna("").astype(str).str.strip().str.lower()
    )

    if annotations.empty:
        raise ValueError(f"Annotationsdatei ist leer: {path.name}")
    if annotations[["start_time_s", "end_time_s"]].isna().any().any():
        raise ValueError(f"Ungültige Zeitwerte in {path.name}")
    if (annotations["end_time_s"] <= annotations["start_time_s"]).any():
        raise ValueError(f"Endzeit <= Startzeit in {path.name}")

    valid_labels = {"bite", "scoop", "cut", "rest", "other"}
    invalid_labels = sorted(set(annotations["label"]) - valid_labels)
    if invalid_labels:
        raise ValueError(
            f"Ungültige Labels in {path.name}: " + ", ".join(invalid_labels)
        )

    return annotations.sort_values(["start_time_s", "end_time_s"]).reset_index(drop=True)


def assign_sample_labels(
    timestamps_s: np.ndarray,
    annotations: pd.DataFrame,
    sensor_name: str,
) -> np.ndarray:
    sample_labels = np.full(len(timestamps_s), "", dtype=object)

    for row_index, row in enumerate(annotations.itertuples(index=False)):
        is_last = row_index == len(annotations) - 1
        if is_last:
            mask = (
                (timestamps_s >= row.start_time_s)
                & (timestamps_s <= row.end_time_s + TIME_TOLERANCE_S)
            )
        else:
            mask = (
                (timestamps_s >= row.start_time_s)
                & (timestamps_s < row.end_time_s)
            )

        if np.any(sample_labels[mask] != ""):
            raise ValueError(
                f"Überlappende Annotationen beim Zuordnen: {sensor_name}"
            )

        sample_labels[mask] = row.label

    missing_count = int((sample_labels == "").sum())
    if missing_count:
        missing_times = timestamps_s[sample_labels == ""]
        raise ValueError(
            f"{sensor_name}: {missing_count} Samples haben kein Label. "
            f"Zeitbereich: {missing_times.min():.3f}–{missing_times.max():.3f} s"
        )

    return sample_labels


def classify_window(labels: np.ndarray) -> tuple[str | None, float]:
    bite_ratio = float(np.mean(labels == "bite"))
    non_bite_ratio = 1.0 - bite_ratio

    if bite_ratio >= MIN_LABEL_PURITY:
        return "bite", bite_ratio
    if non_bite_ratio >= MIN_LABEL_PURITY:
        return "not_bite", non_bite_ratio
    return None, max(bite_ratio, non_bite_ratio)


def main() -> None:
    splits = read_splits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_labels = []
    metadata_rows = []
    session_rows = []
    skipped_mixed_windows = 0

    for session in splits.itertuples(index=False):
        sensor_path = Path(session.path)
        annotation_path = Path(session.annotation_path)

        sensor = read_sensor(sensor_path)
        annotations = read_annotations(annotation_path, sensor_path.name)

        timestamps_s = sensor["timestamp_ms"].to_numpy(dtype=np.float64)
        timestamps_s = (timestamps_s - timestamps_s[0]) / 1000.0
        values = sensor[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        samples = sensor["sample"].to_numpy(dtype=np.int64)
        sample_labels = assign_sample_labels(
            timestamps_s, annotations, sensor_path.name
        )

        accepted = 0
        skipped = 0
        starts = range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE)

        for start_index in starts:
            end_index = start_index + WINDOW_SIZE
            binary_label, purity = classify_window(sample_labels[start_index:end_index])

            if binary_label is None:
                skipped += 1
                skipped_mixed_windows += 1
                continue

            label_id = 1 if binary_label == "bite" else 0
            all_windows.append(values[start_index:end_index])
            all_labels.append(label_id)

            metadata_rows.append(
                {
                    "window_id": len(metadata_rows),
                    "file": session.file,
                    "path": str(sensor_path),
                    "source": session.source,
                    "split": session.split,
                    "annotation_file": session.annotation_file,
                    "binary_label": binary_label,
                    "label_id": label_id,
                    "label_purity": round(purity, 4),
                    "window_start_index": start_index,
                    "window_end_index_exclusive": end_index,
                    "window_start_sample": int(samples[start_index]),
                    "window_end_sample_exclusive": int(samples[end_index - 1] + 1),
                    "window_start_time_s": round(float(timestamps_s[start_index]), 4),
                    "window_end_time_s": round(float(timestamps_s[end_index - 1]), 4),
                }
            )
            accepted += 1

        session_rows.append(
            {
                "file": session.file,
                "split": session.split,
                "accepted_windows": accepted,
                "skipped_mixed_windows": skipped,
            }
        )

    if not all_windows:
        raise ValueError("Es wurden keine Bite-Detektor-Fenster erzeugt.")

    X = np.stack(all_windows).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)
    session_summary = pd.DataFrame(session_rows)

    npz_path = OUTPUT_DIR / f"bite_windows_{WINDOW_SIZE}_step_{STEP_SIZE}.npz"
    metadata_path = OUTPUT_DIR / f"bite_windows_{WINDOW_SIZE}_step_{STEP_SIZE}_metadata.csv"
    session_path = OUTPUT_DIR / f"bite_windows_{WINDOW_SIZE}_step_{STEP_SIZE}_session_summary.csv"

    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        feature_names=np.array(FEATURE_COLUMNS),
        class_names=CLASS_NAMES,
        window_size=np.array(WINDOW_SIZE),
        step_size=np.array(STEP_SIZE),
        min_label_purity=np.array(MIN_LABEL_PURITY),
    )
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    session_summary.to_csv(session_path, index=False, encoding="utf-8-sig")

    print("=== BITE-WINDOWING FERTIG ===")
    print(f"Fenstergröße: {WINDOW_SIZE} Samples")
    print(f"Schrittweite: {STEP_SIZE} Samples")
    print(f"Minimale Label-Reinheit: {MIN_LABEL_PURITY:.0%}")
    print(f"Klassen: {list(CLASS_NAMES)}")
    print(f"X-Form: {X.shape}")
    print(f"y-Form: {y.shape}")
    print(f"Verworfene Mischfenster: {skipped_mixed_windows}")

    print("\n=== FENSTER PRO BINARY LABEL UND SPLIT ===")
    print(
        metadata.groupby(["binary_label", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=CLASS_NAMES, fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
        .to_string()
    )

    print("\n=== FENSTER PRO SESSION ===")
    print(session_summary.to_string(index=False))

    print(f"\nNPZ-Datei gespeichert: {npz_path}")
    print(f"Metadaten gespeichert: {metadata_path}")
    print(f"Session-Zusammenfassung gespeichert: {session_path}")


if __name__ == "__main__":
    main()
