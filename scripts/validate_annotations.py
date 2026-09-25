from pathlib import Path

import numpy as np
import pandas as pd

CONTINUOUS_DIR = Path("data/raw/continuous")
ANNOTATIONS_DIR = Path("data/annotations/continuous")
OUTPUT_FILE = Path("data/metadata/annotation_overview.csv")

CLASS_NAMES = ["bite", "scoop", "cut", "rest", "other"]
OTHER_TYPES = [
    "pickup",
    "returnplate",
    "putdown",
    "reposition",
    "gesture",
    "hold",
    "transition",
]

SPLIT_NAMES = {"train", "val", "test"}
SENSOR_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
REQUIRED_SENSOR_COLUMNS = ["timestamp_ms", "sample"] + SENSOR_COLUMNS
REQUIRED_ANNOTATION_COLUMNS = ["start_time_s", "end_time_s", "label"]
OPTIONAL_ANNOTATION_COLUMNS = ["other_type", "notes"]
TIME_TOLERANCE_S = 0.10


def split_for_sensor(sensor_path: Path) -> str | None:
    try:
        relative = sensor_path.relative_to(CONTINUOUS_DIR)
    except ValueError:
        return None

    if len(relative.parts) < 2:
        return None

    split = relative.parts[0]
    return split if split in SPLIT_NAMES else None


def annotation_path_for(sensor_path: Path) -> Path | None:
    split = split_for_sensor(sensor_path)
    if split is None:
        return None

    relative = sensor_path.relative_to(CONTINUOUS_DIR / split)
    return ANNOTATIONS_DIR / split / relative.parent / (
        f"{sensor_path.stem}_annotations.csv"
    )


def base_result(
    sensor_path: Path,
    split: str | None,
    annotation_path: Path | None,
) -> dict:
    return {
        "split": split or "",
        "sensor_file": sensor_path.name,
        "sensor_path": str(sensor_path),
        "annotation_file": annotation_path.name if annotation_path else "",
        "annotation_path": str(annotation_path) if annotation_path else "",
        "sensor_duration_s": np.nan,
        "segment_count": 0,
        "annotated_duration_s": 0.0,
        "coverage_ratio": 0.0,
        "gap_count": 0,
        "overlap_count": 0,
        "status": "OK",
        "details": "",
    }


def read_sensor_duration(sensor_path: Path) -> tuple[float | None, str | None]:
    try:
        sensor = pd.read_csv(sensor_path)
    except Exception as error:
        return None, f"SENSOR_READ_ERROR: {error}"

    missing = [
        column for column in REQUIRED_SENSOR_COLUMNS if column not in sensor.columns
    ]
    if missing:
        return None, "SENSOR_MISSING_COLUMNS: " + ", ".join(missing)

    if len(sensor) < 2:
        return None, "SENSOR_TOO_FEW_ROWS"

    timestamps = pd.to_numeric(sensor["timestamp_ms"], errors="coerce")
    if timestamps.isna().any():
        return None, "SENSOR_INVALID_TIMESTAMPS"

    duration_s = float((timestamps.iloc[-1] - timestamps.iloc[0]) / 1000)
    if duration_s <= 0:
        return None, "SENSOR_INVALID_DURATION"

    return duration_s, None


def validate_annotation(
    sensor_path: Path,
    split: str,
    annotation_path: Path,
    duration_s: float,
) -> tuple[dict, pd.DataFrame | None]:
    result = base_result(sensor_path, split, annotation_path)
    result["sensor_duration_s"] = round(duration_s, 3)

    try:
        annotations = pd.read_csv(annotation_path)
    except Exception as error:
        result["status"] = "ANNOTATION_READ_ERROR"
        result["details"] = str(error)
        return result, None

    missing = [
        column
        for column in REQUIRED_ANNOTATION_COLUMNS
        if column not in annotations.columns
    ]
    if missing:
        result["status"] = "ANNOTATION_MISSING_COLUMNS"
        result["details"] = ", ".join(missing)
        return result, None

    if annotations.empty:
        result["status"] = "ANNOTATION_EMPTY"
        result["details"] = "Keine Segmentzeilen vorhanden."
        return result, None

    annotations = annotations.copy()
    for column in OPTIONAL_ANNOTATION_COLUMNS:
        if column not in annotations.columns:
            annotations[column] = ""

    annotations["start_time_s"] = pd.to_numeric(
        annotations["start_time_s"], errors="coerce"
    )
    annotations["end_time_s"] = pd.to_numeric(
        annotations["end_time_s"], errors="coerce"
    )
    annotations["label"] = (
        annotations["label"].fillna("").astype(str).str.strip().str.lower()
    )
    annotations["other_type"] = (
        annotations["other_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    annotations["notes"] = annotations["notes"].fillna("").astype(str).str.strip()

    errors = []

    if annotations[["start_time_s", "end_time_s"]].isna().any().any():
        errors.append("Ungültige oder fehlende Start-/Endzeit.")

    if (annotations["end_time_s"] <= annotations["start_time_s"]).any():
        errors.append("Mindestens ein Segment hat end_time_s <= start_time_s.")

    invalid_labels = sorted(set(annotations["label"]) - set(CLASS_NAMES))
    if invalid_labels:
        errors.append("Ungültige Labels: " + ", ".join(invalid_labels))

    other_rows = annotations["label"] == "other"
    if (other_rows & (annotations["other_type"] == "")).any():
        errors.append("Mindestens ein other-Segment hat keinen other_type.")

    invalid_other_types = sorted(
        set(annotations.loc[other_rows, "other_type"]) - set(OTHER_TYPES) - {""}
    )
    if invalid_other_types:
        errors.append(
            "Ungültige other_type-Werte: " + ", ".join(invalid_other_types)
        )

    if ((~other_rows) & (annotations["other_type"] != "")).any():
        errors.append("other_type ist nur bei label=other erlaubt.")

    if errors:
        result["status"] = "ANNOTATION_INVALID"
        result["details"] = " | ".join(errors)
        return result, annotations

    annotations = annotations.sort_values(
        ["start_time_s", "end_time_s"]
    ).reset_index(drop=True)

    gaps = 0
    overlaps = 0
    first_start = float(annotations.loc[0, "start_time_s"])
    if first_start > TIME_TOLERANCE_S:
        gaps += 1

    previous_end = float(annotations.loc[0, "end_time_s"])
    for row_index in range(1, len(annotations)):
        start = float(annotations.loc[row_index, "start_time_s"])
        end = float(annotations.loc[row_index, "end_time_s"])

        if start > previous_end + TIME_TOLERANCE_S:
            gaps += 1
        elif start < previous_end - TIME_TOLERANCE_S:
            overlaps += 1

        previous_end = max(previous_end, end)

    if previous_end < duration_s - TIME_TOLERANCE_S:
        gaps += 1

    if (annotations["start_time_s"] < -TIME_TOLERANCE_S).any():
        errors.append("Mindestens eine Startzeit liegt vor 0 Sekunden.")

    if (annotations["end_time_s"] > duration_s + TIME_TOLERANCE_S).any():
        errors.append("Mindestens eine Endzeit liegt nach dem Sensorende.")

    annotated_duration_s = float(
        (annotations["end_time_s"] - annotations["start_time_s"]).sum()
    )
    result["segment_count"] = len(annotations)
    result["annotated_duration_s"] = round(annotated_duration_s, 3)
    result["coverage_ratio"] = round(annotated_duration_s / duration_s, 4)
    result["gap_count"] = gaps
    result["overlap_count"] = overlaps

    if errors:
        result["status"] = "ANNOTATION_OUT_OF_RANGE"
        result["details"] = " | ".join(errors)
    elif overlaps:
        result["status"] = "ANNOTATION_OVERLAPS"
        result["details"] = f"Überlappungen: {overlaps}"
    elif gaps:
        result["status"] = "ANNOTATION_GAPS"
        result["details"] = f"Lücken: {gaps}"

    return result, annotations


def main() -> None:
    if not CONTINUOUS_DIR.exists():
        raise FileNotFoundError(
            f"Continuous-Datenordner nicht gefunden: {CONTINUOUS_DIR}"
        )

    sensor_files = sorted(CONTINUOUS_DIR.rglob("*.csv"))
    if not sensor_files:
        print(f"Keine Continuous-Sensor-CSVs gefunden: {CONTINUOUS_DIR}")
        return

    results = []
    label_durations = {label: 0.0 for label in CLASS_NAMES}
    other_type_durations = {other_type: 0.0 for other_type in OTHER_TYPES}

    print("=== ANNOTATIONS-VALIDIERUNG ===")

    for sensor_path in sensor_files:
        split = split_for_sensor(sensor_path)
        annotation_path = annotation_path_for(sensor_path)

        if split is None or annotation_path is None:
            result = base_result(sensor_path, split, annotation_path)
            result["status"] = "INVALID_DIRECTORY_STRUCTURE"
            result["details"] = (
                "Sensor-CSV muss unter data/raw/continuous/train, /val oder /test liegen."
            )
            results.append(result)
            print(f"FEHLER: {sensor_path} -> ungültige Ordnerstruktur")
            continue

        if not annotation_path.exists():
            result = base_result(sensor_path, split, annotation_path)
            result["status"] = "ANNOTATION_FILE_MISSING"
            result["details"] = "Keine passende Annotationsdatei gefunden."
            results.append(result)
            print(f"FEHLER: {sensor_path.name} [{split}] -> Annotation fehlt")
            continue

        duration_s, sensor_error = read_sensor_duration(sensor_path)
        if sensor_error:
            result = base_result(sensor_path, split, annotation_path)
            result["status"] = sensor_error
            results.append(result)
            print(f"FEHLER: {sensor_path.name} [{split}] -> {sensor_error}")
            continue

        result, annotations = validate_annotation(
            sensor_path=sensor_path,
            split=split,
            annotation_path=annotation_path,
            duration_s=duration_s,
        )
        results.append(result)

        print(
            f"{result['status']}: {sensor_path.name} [{split}] | "
            f"Segmente={result['segment_count']} | "
            f"Abdeckung={result['coverage_ratio']:.1%}"
        )
        if result["details"]:
            print(f"  {result['details']}")

        if result["status"] == "OK" and annotations is not None:
            annotations = annotations.copy()
            annotations["duration_s"] = (
                annotations["end_time_s"] - annotations["start_time_s"]
            )

            for label, duration in annotations.groupby("label")["duration_s"].sum().items():
                label_durations[label] += float(duration)

            other_annotations = annotations[annotations["label"] == "other"]
            for other_type, duration in other_annotations.groupby("other_type")[
                "duration_s"
            ].sum().items():
                other_type_durations[other_type] += float(duration)

    overview = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n=== SESSIONSSTATUS ===")
    print(overview.groupby(["split", "status"]).size().rename("sessions").to_string())

    print("\n=== ANNOTIERTE DAUER PRO HAUPTLABEL ===")
    for label in CLASS_NAMES:
        print(f"{label}: {label_durations[label]:.2f} s")

    print("\n=== ANNOTIERTE OTHER-DAUER PRO UNTERTYP ===")
    for other_type in OTHER_TYPES:
        print(f"{other_type}: {other_type_durations[other_type]:.2f} s")

    print(f"\nÜbersicht gespeichert: {OUTPUT_FILE}")
    ok_count = int((overview["status"] == "OK").sum())
    print(f"Gültige Sessions: {ok_count}/{len(overview)}")


if __name__ == "__main__":
    main()
