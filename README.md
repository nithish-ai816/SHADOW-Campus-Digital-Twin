# SHADOW — Campus Energy Digital Twin & External Automatic Shutdown Control

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65%2B-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SHADOW** is an intelligent Campus Energy Digital Twin designed to autonomously detect unnecessary electrical consumption in classrooms, estimate wasted wattage, alert authorized faculty, and interface with physical electronic control devices for automated power cutoff.

---

## ⚡ Core Features

1. **Camera-Based Non-Intrusive AI Monitoring**
   * Uses computer vision to estimate people occupancy, fan motion, and lighting status.
   * Renders high-security CCTV surveillance bounding boxes (`Identified` green badges, `Unknown` red badges, face/head targets, and HUD telemetry).
   * Strictly non-intrusive estimation — no direct electrical energy sensors or biometric storage.

2. **Automatic Energy-Waste Estimation & Timetable Alignment**
   * Correlates real-time camera occupancy against classroom academic schedules.
   * Detects empty rooms with active loads (fans/lights) and calculates estimated waste in kW and hourly cost (₹).

3. **Faculty Directory & Automatic Alert Routing**
   * Assign faculty members to classrooms with a dedicated directory and `➕ Add Faculty` registration.
   * Routes automated alert emails directly to the assigned faculty member's inbox.

4. **10-State Countdown & Automatic Shutdown Automation**
   * Configurable countdown timer (e.g. 60s) before issuing isolation commands.
   * Continuous condition reassessment allows manual override ("Keep Power ON") or condition cancellation.

5. **External Hardware Device Integration**
   * SHADOW acts as the decision-making brain; separate microcontroller hardware (ESP32 / Raspberry Pi) acts as the electrical actuator.
   * HMAC-authenticated JSON protocol isolates non-critical circuits via 4-channel optocoupler relays and AC contactors.
   * Hard-locked safety loops protect emergency exit lighting and critical safety circuits.

6. **Secure Surveillance Footage Storage**
   * 10 pre-loaded classroom surveillance scenarios in browser-playable WebP animations and MP4.
   * Event-linked video evidence archive with camera credentials authorization.

---

## 🔌 Hardware Wiring & Architecture

```
┌─────────────────────────────────┐
│     SHADOW Software Server      │  (Decision-Making Brain)
│  (Camera AI & Countdown Logic)  │
└────────────────┬────────────────┘
                 │ Authenticated JSON via REST POST / MQTT
                 │ {"action": "SHUTDOWN_CIRCUITS", "auth_token": "..."}
                 ▼
┌─────────────────────────────────┐
│   ESP32 Microcontroller (Wi-Fi) │  (Electrical Control System)
│   GPIO 18 ──► Relay 1 (Lights)  │
│   GPIO 19 ──► Relay 2 (Fans)    │
│   GPIO 21 ──► Relay 3 (AC)      │
└────────────────┬────────────────┘
                 │ 5V Logic Control
                 ▼
┌─────────────────────────────────┐
│ 4-Channel Optocoupler Relay     │  (Galvanic Isolation)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ AC Power Contactor (25A / 230V) │  (Mounted in Distribution Board)
└─────────────────────────────────┘  ──► Safely Disconnects Appliances
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Run the Upgraded Dashboard
```bash
streamlit run app_shadow_ai.py
```
Open **http://localhost:8501** in your browser.

**Default Login:**
* **Username:** `admin`
* **Password:** `admin123`

### 3. Run the Local SMTP Test Receiver (Optional)
To test alert email delivery offline without credentials:
```bash
python local_email_receiver.py
```

### 4. Run Automated Test Verification Suite
Validates all 12 system requirements:
```bash
python -c "from test_suite import ShadowTestSuite; suite = ShadowTestSuite(); print('Passed:', sum(1 for r in suite.run_all_tests() if r['status'] == 'PASS'), '/ 12')"
```

---

## 📂 Project Structure

```
NEW_SHADOW/
├── app_new.py                  # Original baseline SHADOW application
├── app_shadow_ai.py            # Upgraded SHADOW system with Camera AI & Relays
├── camera_ai.py                # Computer Vision engine (HUD & Bounding Boxes)
├── camera_sources.py           # Video file, Webcam, RTSP, and Synthetic feeds
├── energy_waste.py             # Timetable cross-referencing & waste estimation
├── device_integration.py       # External hardware manager & HMAC security
├── test_suite.py               # 12-test automated verification suite
├── local_email_receiver.py     # Local offline SMTP debug receiver
├── shadow_ai_data.json         # Digital twin campus data store
├── shadow_users.json           # User credentials and role-based access
├── assets/                     # Campus maps, brand logos, sample MP4 & WebP footage
└── .streamlit/
    └── config.toml             # Custom dark theme configuration
```
