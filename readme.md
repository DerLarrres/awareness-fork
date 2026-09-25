
# Awareness-Fork

A sensor-based prototype that uses an inertial measurement unit (IMU), an ESP32
microcontroller, and an LSTM-based machine-learning model to detect likely
fork-to-mouth movements in real time.

The prototype provides an awareness prompt when the interval between two
confirmed bite events is shorter than ten seconds. It is designed to support
reflection on eating pace and is not a medical device.

## Project Overview

The physical prototype consists of a standard fork with a GY-91 IMU sensor and
an ESP32 microcontroller attached to it. The ESP32 transmits six motion signals
to a computer through a USB serial connection:

```text
ax, ay, az → linear acceleration
gx, gy, gz → angular velocity
```

A Python application processes the incoming data, normalises the sensor window,
and uses an LSTM-based binary classifier to distinguish between:

```text
1 = bite
0 = not_bite
```

A bite event is created only when the live model output meets the configured
event-detection conditions.

## Hardware
- ESP32 Dev Module / ESP-WROOM-32
- GY-521 mit MPU-6050
- USB-C-Verbindung zum Windows-PC
- I2C: VCC → 3V3, GND → GND, SDA → GPIO21/D21, SCL → GPIO22/D22

Both the GY-91 sensor and ESP32 microcontroller are attached to the fork so
that the measured motion represents the movement of the utensil itself.

## Logger

- Sampling rate: 50 Hz
- Sample interval: 20 ms
- I²C clock speed: 100 kHz
- Accelerometer range: ±8 g
- Gyroscope range: ±500°/s
- Serial baud rate: 115200

## Repository Structure

awareness-fork-public/
├── data/
│   ├── examples/
│   │   ├── raw_examples
│   │   └── annotation_examples
│   └── processed/
│   └── metadata/
│
├── firmware/esp32_imu_logger
│   └── esp32_imu_logger.ino
│
├── models/bite_detector/
│   ├── v4_validated
│   │   ├── bite_lstm_v4_final.keras
│   │   ├── bite_normalization_stats.csv
│   │   ├── bite_lstm_metrics.json
│   │   ├── deployment_config.json
│   │   ├── bite_lstm_val_summary.text     
│   │   └── bite_windows_50_step_5_normalized
│   └── v4_test
│       ├── bite_event_test_ground_truth.csv
│       ├── bite_event_test_predictions.csv
│       ├── bite_event_test_session_summary.csv
│       ├── bite_lstm_test_confusion_matrix.csv
│       ├── bite_lstm_test_predictions.csv
│       └── bite_lstm_test_summary.txt
│
├── scripts/
│   ├── validate_annotations.py
│   ├── create_splits.py
│   ├── create_bite_windows.py
│   ├── normalize_bite_events.py
│   ├── train_bite_lstm.py
│   ├── evaluate_bite_events.py
│   ├── test_bite_events.py
│   ├── replay_bite_detector.py
│   ├── live_bite_detector.py
│   └── live_bite_detector_v4.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Data and Model Pipeline

The project uses a reproducible pipeline to prepare sensor data, train the model and evaluate bite events:

validate_annotations.py
        ↓
create_splits.py
        ↓
create_bite_windows.py
        ↓
normalize_bite_events.py
        ↓
train_bite_lstm.py
        ↓
evaluate_bite_events.py
        ↓
test_bite_events.py

The raw data was recorded in approximately 30-second sessions. Each session has
a raw IMU CSV file and a corresponding video-derived annotation CSV file.

The initial annotation categories are:

bite, scoop, cut, rest, other

For model training, they are converted into the binary classes `bite` and
`not_bite`.

### Installation

### 1. Clone the repository

```bash
git clone https://github.com/DerLarrres/awareness-fork-public.git
cd awareness-fork
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Live Detection

Connect the ESP32 to the computer using USB and ensure that the serial port in
the deployment configuration matches the connected device. Windows power mode should be "Best performance".

Run the V4 live detector from the repository root:

python scripts/live_bite_detector_v4.py

The live detector loads:

models/bite_lstm_v4_final.keras
models/v4_normalization_statistics.csv
config/deployment_config_v4.json

### V4 Event Configuration

The final V4 deployment configuration uses:

| Parameter | Value |
|---|---:|
| Bite-probability threshold | 0.9 |
| Minimum consecutive positive windows | 3 |
| Refractory period | 1.2 seconds |
| Bite-interval awareness threshold | 18 seconds |

The detector evaluates overlapping sensor windows approximately every 0.1
seconds. A detected bite event is created only after three consecutive positive
model outputs. The refractory period prevents one physical fork-to-mouth
movement from being counted multiple times.

## Offline Replay

`replay_bite_detector.py` sends a recorded raw IMU CSV through the same
preprocessing, LSTM inference, and event-detection logic used by the live
detector. This allows repeatable testing without a connected ESP32.

Example:


python scripts/replay_bite_detector.py data/examples/example_raw_session.csv


Adjust the command if your script expects command-line options or a different
file path.

## Final V4 Results

The final V4 model was evaluated on separate validation and test datasets.

| Dataset | Windows | Bite windows | Accuracy | Bite precision | Bite recall | Bite F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 2,426 | 113 | 99.34% | 93.69% | 92.04% | 92.86% |
| Test | 2,117 | 136 | 97.83% | 84.62% | 80.88% | 82.71% |

These figures describe window-level binary classification. Real-time behaviour
also depends on thresholding, consecutive-positive confirmation, and the
refractory period.

## Data Availability and Privacy

Complete raw IMU recordings, annotation files, processed data, and recording
videos are not included in this repository. These materials contain behavioural
data and, in the case of videos, may contain personally identifiable
information.

The repository includes anonymised example CSV files to document the expected
input and annotation formats. The data-processing, training, evaluation, and
inference code is included, but the full research dataset is retained locally
and is not publicly released.

## Limitations

- The prototype detects motion patterns associated with fork-to-mouth
  movements; it does not prove that food was consumed.
- Performance may vary across people, hands, fork orientations, food types,
  and eating styles.
- The system requires a USB-connected computer and is not yet a standalone
  embedded device.
- The ten-second interval threshold is a project-defined awareness setting,
  not a medical recommendation or diagnosis.
- The prototype is not a medical device and must not be used for clinical,
  employment, insurance, surveillance, or other high-stakes decisions.

## Ethical Use

The system is intended only as a voluntary, user-controlled awareness tool.
Raw sensor data, bite histories, and videos should not be stored by default or
used for monitoring another person. The user should be able to disable feedback
at any time.

## License

All rights reserved. This repository is provided for academic assessment and
inspection. 

