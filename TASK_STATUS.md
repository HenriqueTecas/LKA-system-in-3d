# Lab 1 Task Status

Context from `lab_1.pdf` (due 12-12-2025, 23:59:59).

## Task Tracker
- **Task 1 – Joystick/keyboard steering:** Implemented in `src/main.py` (pygame event handling); car responds to left/right input.
- **Task 2a – Car model:** Implemented in `src/car.py` as a kinematic Ackermann model with steering rate limiting and wheel animation; enhanced with weight transfer, simple pitch/roll suspension, and an HL friction-limited tire model (μ + downforce cap instead of Pacejka).
- **Task 2b – Sensor models:** Implemented via lane-detection camera (`src/realistic_camera.py`; older `camera_sensor.py` noted in README). Provides lane boundaries and confidence.
- **Task 3 – Simulation environment & manual lane keeping:** Implemented. `src/main.py` builds the track (`src/track.py`), renderer (`src/renderer.py`), minimap/HUD, and allows manual driving to stay inside a lane.
- **Task 4 – Time-series plot of lane detections:** Completed. Added `LaneDetectionLogger` (see `src/lane_logger.py`) that records lane detections at each physics step, streams them to timestamped CSV files in `logs/`, and renders a live time-history plot on the HUD showing left/right boundaries and lane center offsets; HUD displays active CSV filename.
- **Task 5 – LTA controller:** Implemented. Hybrid LKA with 3 modos (`src/hybrid_controller.py`) integrado em `src/main.py` (1=manual, 2=warnings, 3=assist com controlo de direção/velocidade). Pure Pursuit/MPC mantidos como legado (sem teclas ativas).
- **Task 6 – Mathematical effectiveness proof:** Not yet covered in code; belongs in the report (no analytical proof included).

## Deliverables Checklist
- Report with techniques/results and individual roles: **Pending** (not in repo).
- Software with install/run instructions: Partially covered (README in `src/` and `requirements.txt` at repo root).
