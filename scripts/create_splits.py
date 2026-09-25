from pathlib import Path

import pandas as pd

OVERVIEW_FILE = Path("data/metadata/annotation_overview.csv")
OUTPUT_FILE = Path("data/metadata/splits.csv")

VALID_SPLITS = ["train", "val", "test"]


def main() -> None:
    if not OVERVIEW_FILE.exists():
        raise FileNotFoundError(
            f"Annotation-Übersicht nicht gefunden: {OVERVIEW_FILE}"
        )

    overview = pd.read_csv(OVERVIEW_FILE)

    required_columns = [
        "split",
        "sensor_file",
        "sensor_path",
        "annotation_file",
        "annotation_path",
        "status",
    ]
    missing_columns = [
        column for column in required_columns if column not in overview.columns
    ]
    if missing_columns:
        raise ValueError(
            "Fehlende Spalten in annotation_overview.csv: "
            + ", ".join(missing_columns)
        )

    overview["split"] = overview["split"].fillna("").astype(str).str.strip()
    overview["status"] = overview["status"].fillna("").astype(str).str.strip()

    invalid_splits = sorted(set(overview["split"]) - set(VALID_SPLITS))
    if invalid_splits:
        raise ValueError(
            "Ungültige Split-Namen in annotation_overview.csv: "
            + ", ".join(invalid_splits)
        )

    if overview["sensor_file"].duplicated().any():
        duplicate_files = sorted(
            overview.loc[
                overview["sensor_file"].duplicated(keep=False), "sensor_file"
            ].astype(str).unique()
        )
        raise ValueError(
            "Sensor-Dateien kommen mehrfach vor: " + ", ".join(duplicate_files)
        )

    invalid_annotated = overview[
        overview["split"].isin(["train", "val"])
        & (overview["status"] != "OK")
    ].copy()

    if not invalid_annotated.empty:
        details = "\n".join(
            f"- {row.sensor_file} [{row.split}]: {row.status}"
            for row in invalid_annotated.itertuples()
        )
        raise ValueError(
            "Train- und Val-Sessions müssen vor dem Split-Erzeugen vollständig "
            "und gültig annotiert sein:\n"
            + details
        )

    invalid_test = overview[
        (overview["split"] == "test")
        & ~overview["status"].isin(["OK", "ANNOTATION_EMPTY"])
    ].copy()

    if not invalid_test.empty:
        details = "\n".join(
            f"- {row.sensor_file}: {row.status}"
            for row in invalid_test.itertuples()
        )
        raise ValueError(
            "Test-Sessions dürfen nur OK oder bewusst ANNOTATION_EMPTY sein:\n"
            + details
        )

    included = overview[
        (overview["status"] == "OK")
        & overview["split"].isin(VALID_SPLITS)
    ].copy()

    if included.empty:
        raise ValueError("Keine gültigen annotierten Sessions für Splits gefunden.")

    splits = pd.DataFrame(
        {
            "file": included["sensor_file"],
            "path": included["sensor_path"],
            "source": "continuous",
            "split": included["split"],
            "annotation_file": included["annotation_file"],
            "annotation_path": included["annotation_path"],
        }
    )

    split_order = {"train": 0, "val": 1, "test": 2}
    splits["_split_order"] = splits["split"].map(split_order)
    splits = (
        splits.sort_values(["_split_order", "file"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("=== CONTINUOUS-SPLITS ERSTELLT ===")
    print(
        splits.groupby("split")
        .size()
        .reindex(VALID_SPLITS, fill_value=0)
        .rename("sessions")
        .to_string()
    )

    print("\n=== AUSGESCHLOSSENE SESSIONS ===")
    excluded = overview[overview["status"] != "OK"].copy()
    if excluded.empty:
        print("Keine.")
    else:
        print(
            excluded[["split", "sensor_file", "status", "details"]]
            .sort_values(["split", "sensor_file"])
            .to_string(index=False)
        )

    print(f"\nSplit-Datei gespeichert: {OUTPUT_FILE}")
    print("Hinweis: Noch leere Test-Annotationen werden bewusst nicht aufgenommen.")


if __name__ == "__main__":
    main()
