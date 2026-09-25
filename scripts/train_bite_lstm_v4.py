from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_PATH = Path("data/processed/bite_windows_50_step_5_normalized.npz")
METADATA_PATH = Path("data/processed/bite_windows_50_step_5_metadata.csv")
OUTPUT_DIR = Path("models/bite_detector/v4_validated")

SEED = 42
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision_bite": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_bite": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_bite": float(f1_score(y_true, prediction, zero_division=0)),
    }


def select_threshold(y_val: np.ndarray, probability: np.ndarray) -> tuple[float, dict]:
    thresholds = np.unique(np.concatenate(([0.05], np.arange(0.10, 0.91, 0.01), [0.95])))
    candidates = [binary_metrics(y_val, probability, float(t)) for t in thresholds]
    best = max(
        candidates,
        key=lambda item: (item["f1_bite"], item["recall_bite"], item["precision_bite"]),
    )
    return best["threshold"], best


def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="imu_window")
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.15)(inputs)
    x = tf.keras.layers.LSTM(32, dropout=0.15)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="bite_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bite_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
        ],
    )
    return model


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Normalisierte Daten nicht gefunden: {DATA_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadaten nicht gefunden: {METADATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = data["feature_names"].astype(str)
    class_names = data["class_names"].astype(str)

    if list(class_names) != ["not_bite", "bite"]:
        raise ValueError(f"Unerwartete Klassenreihenfolge: {class_names.tolist()}")
    if X.ndim != 3 or len(X) != len(y):
        raise ValueError(f"Ungültige Datenform: X={X.shape}, y={y.shape}")

    metadata = pd.read_csv(METADATA_PATH)
    if len(metadata) != len(X):
        raise ValueError(
            f"Metadaten und X haben unterschiedliche Länge: {len(metadata)} vs. {len(X)}"
        )

    split = metadata["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Train- oder Val-Split fehlt.")
    if test_mask.any():
        print("Hinweis: Test-Fenster werden beim Training nicht verwendet.")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    meta_val = metadata.loc[val_mask].reset_index(drop=True)

    train_counts = np.bincount(y_train, minlength=2)
    if np.any(train_counts == 0):
        raise ValueError(f"Eine Klasse fehlt im Training: {train_counts.tolist()}")

    total_train = len(y_train)
    class_weight = {
        class_id: float(total_train / (len(train_counts) * count))
        for class_id, count in enumerate(train_counts)
    }

    print("=== BITE-LSTM TRAINING ===")
    print(f"Train: X={X_train.shape}, Klassen={train_counts.tolist()}")
    print(f"Val:   X={X_val.shape}, Klassen={np.bincount(y_val, minlength=2).tolist()}")
    print(f"Class weights: {class_weight}")

    model = build_model((X.shape[1], X.shape[2]))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "bite_lstm_best.keras"),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / "bite_lstm_history.csv")),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    model = tf.keras.models.load_model(OUTPUT_DIR / "bite_lstm_best.keras")
    val_probability = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).reshape(-1)

    threshold, threshold_metrics = select_threshold(y_val, val_probability)
    val_prediction = (val_probability >= threshold).astype(np.int64)
    cm = confusion_matrix(y_val, val_prediction, labels=[0, 1])

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_val, val_probability
    )
    best_index = int(np.argmax(2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-12)))

    global_metrics = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_class_counts": {"not_bite": int(train_counts[0]), "bite": int(train_counts[1])},
        "val_class_counts": {
            "not_bite": int(np.sum(y_val == 0)),
            "bite": int(np.sum(y_val == 1)),
        },
        "class_weight": {"not_bite": class_weight[0], "bite": class_weight[1]},
        "val_roc_auc": float(roc_auc_score(y_val, val_probability)),
        "val_average_precision": float(average_precision_score(y_val, val_probability)),
        "selected_threshold": float(threshold),
        "selected_threshold_metrics": threshold_metrics,
        "best_pr_curve_index": best_index,
        "epochs_completed": int(len(history.history["loss"])),
    }

    model.save(OUTPUT_DIR / "bite_lstm_final.keras")
    with open(OUTPUT_DIR / "bite_lstm_metrics.json", "w", encoding="utf-8") as file:
        json.dump(global_metrics, file, ensure_ascii=False, indent=2)

    pd.DataFrame(
        cm,
        index=["actual_not_bite", "actual_bite"],
        columns=["pred_not_bite", "pred_bite"],
    ).to_csv(OUTPUT_DIR / "bite_lstm_val_confusion_matrix.csv", encoding="utf-8-sig")

    predictions = meta_val.copy()
    predictions["actual_label_id"] = y_val
    predictions["actual_label"] = np.where(y_val == 1, "bite", "not_bite")
    predictions["bite_probability"] = val_probability
    predictions["threshold"] = threshold
    predictions["predicted_label_id"] = val_prediction
    predictions["predicted_label"] = np.where(val_prediction == 1, "bite", "not_bite")
    predictions.to_csv(OUTPUT_DIR / "bite_lstm_val_predictions.csv", index=False, encoding="utf-8-sig")

    report = classification_report(
        y_val,
        val_prediction,
        labels=[0, 1],
        target_names=["not_bite", "bite"],
        digits=4,
        zero_division=0,
    )
    with open(OUTPUT_DIR / "bite_lstm_val_summary.txt", "w", encoding="utf-8") as file:
        file.write("=== BITE-LSTM VALIDATION ===\n")
        file.write(f"Selected threshold: {threshold:.2f}\n")
        file.write(f"ROC-AUC: {global_metrics['val_roc_auc']:.4f}\n")
        file.write(f"Average precision / PR-AUC: {global_metrics['val_average_precision']:.4f}\n")
        file.write(f"Accuracy: {threshold_metrics['accuracy']:.4f}\n")
        file.write(f"Bite precision: {threshold_metrics['precision_bite']:.4f}\n")
        file.write(f"Bite recall: {threshold_metrics['recall_bite']:.4f}\n")
        file.write(f"Bite F1: {threshold_metrics['f1_bite']:.4f}\n\n")
        file.write("Confusion matrix (rows=actual, columns=predicted):\n")
        file.write(np.array2string(cm))
        file.write("\n\nClassification report:\n")
        file.write(report)

    print("\n=== VALIDATION ===")
    print(f"Ausgewählter Bite-Threshold: {threshold:.2f}")
    print(f"ROC-AUC: {global_metrics['val_roc_auc']:.4f}")
    print(f"PR-AUC / Average Precision: {global_metrics['val_average_precision']:.4f}")
    print(f"Bite Precision: {threshold_metrics['precision_bite']:.4f}")
    print(f"Bite Recall: {threshold_metrics['recall_bite']:.4f}")
    print(f"Bite F1: {threshold_metrics['f1_bite']:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(cm)
    print("\n" + report)
    print(f"\nModell und Ergebnisse gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
