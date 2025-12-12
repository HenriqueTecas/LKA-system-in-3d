# Robotics Lab 1 – Lane Keeping Assist (LTA) Simulator

Implements the Lab 1 tasks from `files/lab_1.pdf` using the ideas from the theory slides. The simulator models a car on a multi-lane highway, detects lane boundaries with an onboard camera, and runs both manual and automatic lane keeping controllers.

## Lab Tasks ↔ Implementation
- **Task 1 – Joystick/keyboard steering:** `src/main.py` input handling (`A/D` steering, `W/S` accel/brake).
- **Task 2a – Car geometric model:** `src/car.py` Ackermann kinematics/dynamics (wheelbase/track consistent with the theory slides).
- **Task 2b – Sensors:** `src/realistic_camera.py` pinhole + homography camera; distance-based confidence and noise.
- **Task 3 – Manual lane keeping demo:** Drive in-sim and stay inside a lane on the São Paulo F1 track (`src/track.py`); HUD shows lane detection status.
- **Task 4 – Lane detection time-series plot:** `src/lane_logger.py` streams detections to `logs/` CSVs and draws a rolling plot on the HUD.
- **Task 5 – LTA controller:** Hybrid LKA with 3 modos (`src/hybrid_controller.py`) substituting Pure Pursuit/MPC toggles. Assiste direção/velocidade em modo 3 e mostra avisos em modo 2. Se a curvatura/safe speed não puder ser estimada, o modo 3 devolve o controlo de velocidade ao condutor (evita “speed NaN”).
- **Task 6 – Analytical effectiveness proof:** *Not in code*; add to the report. Suggested approach: Lyapunov-style error analysis for pure-pursuit tracking or linearized unicycle tracking as in the theory slides.

## Setup
```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows use PowerShell or cmd instead of `source`:

PowerShell (recommended):
```powershell
# From project root
python -m venv .venv
# Activate in the current PowerShell session
.\.venv\Scripts\Activate.ps1
# If execution of scripts is blocked, allow it for this session only:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m pip install -r requirements.txt
```

Command Prompt (cmd.exe):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

If you are using Git Bash or WSL, the original `source` command works inside those environments:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running
```bash
python3 -m src.main
```
- A parameter menu appears in the console to tweak steering/brake ramps and stability caps.
- The sim opens a 3D view plus a HUD/minimap overlay. CSV logs are written under `logs/` (timestamped).
- If your environment lacks GLU, either install it (e.g., Mesa GLU) or switch the renderer to use `glu_fallback` helpers (included).

## Controls
- `W/S`: accelerate / brake (or reverse at low speed)
- `A/D`: steer left / right (manual)
- `1`: Hybrid MANUAL (sem assistência)
- `2`: Hybrid WARNING (avisos apenas)
- `3`: Hybrid ASSIST (deixa o condutor dirigir; assume temporariamente se houver risco de saída de faixa/velocidade insegura)
- `C`: alterna visão da câmara (chase ↔ lane POV)
- `ESC`: quit

## Key Components (src/)
- `car.py`: Ackermann model with force limits, steering rate limits, and an HL-style friction-limited tire model (μ + downforce, no Pacejka).
- `realistic_camera.py`: camera geometry, homography-based ground projection, distance-dependent confidence + noise.
- `hybrid_controller.py`: controlador híbrido (modes 1/2/3) com direção por tangente única-linha + controlo de velocidade preditivo.
- `lka_controller.py` / `mpc_controller.py`: controladores legados mantidos apenas para compatibilidade (não mapeados em teclas).
- `lane_logger.py`: detection logging + HUD plot.
- `track.py`: São Paulo F1 circuit, scaled to meters.
- `hud.py`, `renderer.py`, `minimap.py`: visualization.

## Logs and Plots
- CSVs: `logs/lane_detections_*.csv` (left/center/right offsets, confidences, point counts per physics step).
- HUD plot: shows recent lane offsets and active CSV filename; helpful for Task 4 evidence.

## Reporting (deliverable reminder)
- Include a short report describing:
  - Models used (Ackermann, camera IPM).
  - LTA controller chosen (pure pursuit) and why it tracks the reference lane (keep-lane policy).
  - Empirical evidence (screenshots, logged plots) showing Task 3–5 results.
  - Roles of each author (required by the pdf).
  - Mathematical effectiveness (Task 6): add your Lyapunov/linearized tracking argument here—code does not provide it.

## Notes
- Theory alignment: the pure-pursuit law matches the trajectory-tracking view from the slides; keep-lane enforcement treats the chosen lane as the reference path and returns to it if the camera classification flips.
- Tires: HL friction-limited model (μ-limited traction/lateral grip with downforce) replaces the Pacejka Magic Formula used in the merge build.
- Modes: Mode 2 now warns when current speed exceeds safe curve speed; Mode 3 keeps the driver in control and only takes over temporarily to prevent lane exit/overspeed, then releases back to manual.
- Display issues: if running headless/SSH, use X11 forwarding or `xvfb-run` for OpenGL.
