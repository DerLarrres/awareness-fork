import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ACCEL_COLUMNS = ["ax", "ay", "az"]
GYRO_COLUMNS = ["gx", "gy", "gz"]

COLORS = {
    "ax": "#e74c3c",
    "ay": "#2ecc71",
    "az": "#3498db",
    "gx": "#e74c3c",
    "gy": "#2ecc71",
    "gz": "#3498db",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erstellt Diagramme für eine ESP32-IMU-CSV-Session."
    )
    parser.add_argument(
        "csv_file",
        help="Pfad zur CSV-Datei, z. B. data/raw/rest/rest_20260910_001344.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"FEHLER: Datei nicht gefunden: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as error:
        print(f"FEHLER beim Einlesen der CSV: {error}")
        return

    required_columns = ["timestamp_ms"] + ACCEL_COLUMNS + GYRO_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(f"FEHLER: Fehlende Spalten: {missing_columns}")
        return

    if len(df) < 2:
        print("FEHLER: Zu wenige Messzeilen zum Plotten.")
        return

    # Zeit relativ zum ersten Messpunkt, in Sekunden.
    time_s = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000

    # Label für den Diagrammtitel bestimmen.
    if "label" in df.columns and df["label"].nunique() == 1:
        label = str(df["label"].iloc[0])
    else:
        label = "unbekannt"

    # Zwei Diagramme untereinander.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 9),
        sharex=True,
    )

    # Beschleunigung
    for column in ACCEL_COLUMNS:
        axes[0].plot(
            time_s,
            df[column],
            label=column,
            color=COLORS[column],
            linewidth=1.1,
        )

    axes[0].set_title(f"Beschleunigung – Label: {label}")
    axes[0].set_ylabel("Beschleunigung [m/s²]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Gyro
    for column in GYRO_COLUMNS:
        axes[1].plot(
            time_s,
            df[column],
            label=column,
            color=COLORS[column],
            linewidth=1.1,
        )

    axes[1].set_title("Drehrate")
    axes[1].set_xlabel("Zeit seit Aufnahmestart [s]")
    axes[1].set_ylabel("Gyro [rad/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"IMU-Session: {csv_path.name} | {len(df)} Messpunkte",
        fontsize=14,
    )

    fig.tight_layout()

    # PNG-Ausgabeordner erstellen und Bild speichern.
    output_folder = Path("data/plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{csv_path.stem}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    print(f"Plot gespeichert: {output_path}")

    # Fenster mit Diagramm öffnen.
    plt.show()


if __name__ == "__main__":
    main()