# CARLA Integration (Stub for Future Labs)

This folder holds a lightweight CARLA client to let you experiment with lane-keeping in the CARLA simulator without touching the 3D OpenGL sim. It connects to a running CARLA server, spawns an ego car with a front RGB camera, and runs a simple pure-pursuit-style steering loop that follows the lane center using CARLA waypoints.

> CARLA is not installed in this repo and cannot be vendorized here. Follow the setup steps below on a machine with CARLA available.

## Setup
1) Install CARLA 0.9.14+ (binary or from source) and start the server, e.g.:
```bash
# in your CARLA root
./CarlaUE4.sh -quality-level=Low -opengl  # Linux/OpenGL; adjust for your platform
```

2) Create a venv for the client and install `carla`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install carla==0.9.14 pygame
```

3) From this folder, run the client:
```bash
python lane_follow_client.py --host 127.0.0.1 --port 2000 --town Town04
```

## Script: `lane_follow_client.py`
- Connects to CARLA, loads/preserves the chosen town.
- Spawns an ego vehicle (Tesla Model 3 if available).
- Attaches a front RGB camera for visualization (press `C` to toggle camera display).
- Runs a simple waypoint-based lane-follow controller:
  - Uses CARLA waypoint API to grab a lookahead point on the same lane.
  - Computes a steering command from heading error (pure pursuit–style).
  - Fixed throttle with basic brake when slow.
- Cleans up actors on exit (Ctrl+C).

## Notes
- This is intentionally minimal to keep it portable. For richer control (PID speed, MPC, etc.), drop your own logic in `compute_control()`.
- Traffic/obstacles: disabled by default; enable CARLA traffic manager separately if needed.
- Visualizations: the script doesn’t record logs yet; add CSV/plot emissions similar to `src/lane_logger.py` if you need them.
