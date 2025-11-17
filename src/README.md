# Source Modules

This directory contains the modular implementation of the 3D Robotics Lab simulation.

## Module Structure

```
src/
├── __init__.py           # Package initialization
├── config.py             # Configuration constants
├── main.py               # Main simulation loop
├── car.py                # Car class with Ackermann kinematics
├── camera_sensor.py      # CameraSensor for lane detection
├── lka_controller.py     # PurePursuitLKA lane keeping controller
├── track.py              # SaoPauloTrack circuit layout
├── renderer.py           # Renderer3D for OpenGL 3D scene
├── minimap.py            # Minimap 2D top-down view
└── hud.py                # HUD heads-up display
```

## Module Descriptions

### `config.py`
- Screen dimensions (WIDTH, HEIGHT, MINIMAP_SIZE)
- Frame rate (FPS)
- Color constants (RGB tuples)

### `car.py`
- **Car class**: Ackermann steering kinematics
- 3D rendering with animated wheels
- Wheel rotation based on distance traveled
- **Realistic steering rate limiting for LKA**
- ~415 lines

### `camera_sensor.py`
- **CameraSensor class**: Lane boundary detection
- Uniform sampling at configurable intervals
- Field of view: 100° (increased for better coverage)
- ~223 lines

### `lka_controller.py`
- **PurePursuitLKA class**: Lane keeping assist
- **Lane-specific tracking**: Only uses current lane boundaries (left or right)
- Dense lane center path generation
- Bidirectional pairing and interpolation
- Pure Pursuit steering algorithm
- **NEW: Predictive path extension using curvature estimation**
  - Transforms lane points to ego (car-centric) frame
  - Fits polynomial to estimate curvature: y(x) = ax² + bx + c
  - Projects path forward along constant-curvature arc
  - Realistic approach used in production AVs (Tesla, Waymo)
  - Solves tight turn problem without unrealistic "memory"
- ~341 lines

### `track.py`
- **SaoPauloTrack class**: F1 circuit layout
- 3D road surface and terrain rendering
- Lane markings and scenery
- Track features (checkpoints, signs, buildings)
- ~590 lines

### `renderer.py`
- **Renderer3D class**: OpenGL 3D scene management
- Lighting and camera setup
- Lane marker visualization
- Lookahead point rendering
- ~130 lines

### `minimap.py`
- **Minimap class**: 2D top-down view
- Shows full track, car position, camera FOV
- Lane detection points
- LKA lookahead visualization
- ~220 lines

### `hud.py`
- **HUD class**: Heads-up display
- FPS counter with performance indicators
- LKA status
- Speed and steering telemetry
- Lane detection status
- ~110 lines

### `main.py`
- Display initialization with fallback options
- Component creation and setup
- Main simulation loop
- Event handling
- 3D and 2D rendering coordination
- ~260 lines

## Running

From the project root:
```bash
python3 main.py
```

Or run directly:
```bash
python3 -m src.main
```

## Dependencies

See `requirements.txt` in project root:
- pygame >= 2.0.0
- PyOpenGL >= 3.1.0
- PyOpenGL_accelerate >= 3.1.0
- numpy >= 1.20.0
