#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// Verkabelung des GY-521 am ESP32
const int SDA_PIN = 21;
const int SCL_PIN = 22;

// 20 ms = 50 Messungen pro Sekunde
const unsigned long SAMPLE_INTERVAL_MS = 20;

// Zeitplanung und Zähler
unsigned long nextSampleTime = 0;
unsigned long sampleCount = 0;

void setup() {
  Serial.begin(115200);

  // I2C: SDA = GPIO21, SCL = GPIO22
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);  // 100 kHz

  // GY-521 / MPU-6050 an Adresse 0x68 initialisieren
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("ERROR,MPU6050_NOT_FOUND");

    while (true) {
      delay(1000);
    }
  }

  // Diese Sensorbereiche für alle späteren Aufnahmen unverändert lassen
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Erste Messung ungefähr 20 ms nach Abschluss des Setups
  nextSampleTime = millis() + SAMPLE_INTERVAL_MS;

  // Kopfzeile: exakt 8 Spalten, passend zum Python-Recorder
  Serial.println("timestamp_ms,sample,ax,ay,az,gx,gy,gz");
}

void loop() {
  const unsigned long now = millis();

  // Ist die nächste Messung noch nicht fällig?
  if ((long)(now - nextSampleTime) < 0) {
    return;
  }

  // Nächsten Soll-Zeitpunkt setzen:
  // Stabiler Langzeit-Takt statt "20 ms nach Ende der letzten Schleife".
  nextSampleTime += SAMPLE_INTERVAL_MS;

  sensors_event_t acceleration;
  sensors_event_t rotation;
  sensors_event_t temperature;

  // Sensor auslesen
  const bool readOk = mpu.getEvent(&acceleration, &rotation, &temperature);

  // Fehlerhafte Messwerte sichtbar ausgeben und nicht als CSV-Feature schreiben
  if (!readOk ||
      isnan(acceleration.acceleration.x) ||
      isnan(acceleration.acceleration.y) ||
      isnan(acceleration.acceleration.z) ||
      isnan(rotation.gyro.x) ||
      isnan(rotation.gyro.y) ||
      isnan(rotation.gyro.z)) {
    Serial.println("ERROR,INVALID_SENSOR_VALUE");
    return;
  }

  // Eine Messung = eine CSV-Zeile
  Serial.print(now);
  Serial.print(",");
  Serial.print(sampleCount);
  Serial.print(",");

  Serial.print(acceleration.acceleration.x, 5);
  Serial.print(",");
  Serial.print(acceleration.acceleration.y, 5);
  Serial.print(",");
  Serial.print(acceleration.acceleration.z, 5);
  Serial.print(",");

  Serial.print(rotation.gyro.x, 5);
  Serial.print(",");
  Serial.print(rotation.gyro.y, 5);
  Serial.print(",");
  Serial.println(rotation.gyro.z, 5);

  sampleCount++;
}
