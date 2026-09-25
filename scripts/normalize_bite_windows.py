from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/processed/bite_windows_50_step_5.npz")
OUTPUT_DIR = Path("data/processed")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {INPUT_PATH}")

    data = np.load(INPUT_PATH, allow_pickle=False)
    required_keys = {"X", "y", "feature_names", "class_names", "window_size", "step_size"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise ValueError("Fehlende NPZ-Keys: " + ", ".join(sorted(missing_keys)))

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = data["feature_names"]
    class_names = data["class_names"]
    window_size = data["window_size"]
    step_size = data["step_size"]
    min_label_purity = data["min_label_purity"] if "min_label_purity" in data.files else None

    if X.ndim != 3:
        raise ValueError(f"X muss 3D sein, erhalten: {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"Ungleiche Anzahl X/y: {len(X)} vs. {len(y)}")

    metadata_path = OUTPUT_DIR / "bite_windows_50_step_5_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadaten nicht gefunden: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(X):
        raise ValueError(
            f"Metadaten und X haben unterschiedliche Länge: {len(metadata)} vs. {len(X)}"
        )
    if "split" not in metadata.columns:
        raise ValueError("Spalte 'split' fehlt in den Metadaten.")

    train_mask = metadata["split"].to_numpy() == "train"
    if not train_mask.any():
        raise ValueError("Keine Train-Fenster gefunden.")

    train_values = X[train_mask].reshape(-1, X.shape[-1])
    mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_values.std(axis=0, dtype=np.float64).astype(np.float32)

    if np.any(std <= 0):
        bad_features = feature_names[std <= 0].tolist()
        raise ValueError("Standardabweichung <= 0 für: " + ", ".join(bad_features))

    X_normalized = ((X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_npz = OUTPUT_DIR / "bite_windows_50_step_5_normalized.npz"
    output_stats = OUTPUT_DIR / "bite_normalization_stats.csv"

    save_data = {
        "X": X_normalized,
        "y": y,
        "feature_names": feature_names,
        "class_names": class_names,
        "window_size": window_size,
        "step_size": step_size,
        "mean": mean,
        "std": std,
    }
    if min_label_purity is not None:
        save_data["min_label_purity"] = min_label_purity

    np.savez_compressed(output_npz, **save_data)

    stats = pd.DataFrame(
        {
            "feature": feature_names,
            "train_mean": mean,
            "train_std": std,
            "train_windows": int(train_mask.sum()),
            "train_samples_per_feature": int(train_values.shape[0]),
        }
    )
    stats.to_csv(output_stats, index=False, encoding="utf-8-sig")

    print("=== BITE-NORMALISIERUNG FERTIG ===")
    print(f"Eingabe: {INPUT_PATH}")
    print(f"Ausgabe: {output_npz}")
    print(f"Train-Fenster für Statistik: {int(train_mask.sum())}")
    print(f"X-Form: {X_normalized.shape}")
    print("\n=== TRAIN-STATISTIK ===")
    print(stats.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
