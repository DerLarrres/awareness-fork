import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import serial


BAUDRATE = 115200
EXPECTED_SENSOR_COLUMNS = 8

CSV_HEADER = [
    "timestamp_ms",
    "sample",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]

ANNOTATION_HEADER = [
    "start_time_s",
    "end_time_s",
    "label",
    "notes",
]


def build_output_paths(session_name: str) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = session_name.strip().replace(" ", "_")

    if not safe_name:
        raise ValueError("--name darf nicht leer sein.")

    stem = f"continuous_{safe_name}_{timestamp}"

    data_folder = Path("data/raw/continuous")
    annotation_folder = Path("data/annotations")

    data_folder.mkdir(parents=True, exist_ok=True)
    annotation_folder.mkdir(parents=True, exist_ok=True)

    data_path = data_folder / f"{stem}.csv"
    annotation_path = annotation_folder / f"{stem}_annotations.csv"

    return data_path, annotation_path


def parse_sensor_line(line: str) -> list[str] | None:
    values = line.split(",")

    if len(values) != EXPECTED_SENSOR_COLUMNS:
        return None

    try:
        [float(value) for value in values]
    except ValueError:
        return None

    return values


def create_annotation_template(annotation_path: Path) -> None:
    with annotation_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(ANNOTATION_HEADER)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Nimmt eine unlabelled kontinuierliche ESP32-IMU-Session auf "
            "und erstellt eine leere Segment-Annotationsdatei."
        )
    )
    parser.add_argument(
        "--port",
        required=True,
        help="COM-Port des ESP32, z. B. COM3",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Kurzname, z. B. train_01, val_01 oder test_01",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Aufnahmedauer in Sekunden (Standard: 30)",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds muss größer als 0 sein.")

    data_path, annotation_path = build_output_paths(args.name)

    valid_rows = 0
    invalid_rows = 0

    print(f"Öffne {args.port} bei {BAUDRATE} Baud ...")
    print(f"Session: {args.name}")
    print(f"Aufnahmedauer: {args.seconds:.1f} Sekunden")
    print(f"Sensordaten: {data_path}")
    print(f"Annotationen: {annotation_path}")

    try:
        with serial.Serial(args.port, BAUDRATE, timeout=1) as serial_port:
            time.sleep(2)
            serial_port.reset_input_buffer()

            print("\nAufnahme gestartet. Führe jetzt den Ablauf aus ...")
            print("Merke dir grob die Zeitpunkte der Bewegungswechsel.")

            start_time = time.monotonic()

            with data_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(CSV_HEADER)

                while time.monotonic() - start_time < args.seconds:
                    raw_line = serial_port.readline().decode(
                        "utf-8",
                        errors="replace",
                    ).strip()

                    if not raw_line:
                        continue

                    values = parse_sensor_line(raw_line)

                    if values is None:
                        invalid_rows += 1
                        continue

                    writer.writerow(values)
                    valid_rows += 1

    except serial.SerialException as error:
        print(f"\nSerieller Fehler: {error}")
        return

    create_annotation_template(annotation_path)

    expected_samples = args.seconds * 50

    print("\nAufnahme beendet.")
    print(f"Gültige Messzeilen: {valid_rows}")
    print(f"Übersprungene Zeilen: {invalid_rows}")
    print(f"Erwartungswert bei 50 Hz: ungefähr {expected_samples:.0f} Messzeilen")
    print(f"Sensordaten gespeichert: {data_path}")
    print(f"Leere Annotation gespeichert: {annotation_path}")
    print("\nNächster Schritt: Video ansehen und die Annotationsdatei ausfüllen.")


if __name__ == "__main__":
    main()
