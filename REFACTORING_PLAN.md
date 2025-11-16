# Code Refactoring Plan

## Current Structure
```
robotics_lab_3d.py (2132 lines)
├── Imports & Setup
├── Car class (lines 124-520)
├── CameraSensor class (lines 521-731)
├── PurePursuitLKA class (lines 732-898)
├── SaoPauloTrack class (lines 899-1488)
├── Renderer3D class (lines 1489-1619)
├── Minimap class (lines 1620-1839)
├── HUD class (lines 1840-1950)
└── main() function

```

## Proposed Modular Structure
```
LKA/
├── main.py                  # Entry point
├── robotics_lab_3d.py      # Original (keep for reference)
├── requirements.txt         # Dependencies
├── README_ROBOTICS_3D.md   # Documentation
└── src/
    ├── __init__.py
    ├── config.py            # Constants (WIDTH, HEIGHT, COLORS, FPS)
    ├── car.py               # Car class
    ├── camera_sensor.py     # CameraSensor class
    ├── lka_controller.py    # PurePursuitLKA class
    ├── track.py             # SaoPauloTrack class  
    ├── renderer.py          # Renderer3D class
    ├── minimap.py           # Minimap class
    └── hud.py               # HUD class
```

## Benefits of Refactoring
- ✅ Better code organization
- ✅ Easier to maintain and debug
- ✅ Reusable components
- ✅ Clear separation of concerns
- ✅ Easier to add new features
- ✅ Better for collaboration

## Implementation Status
- [x] Created main.py entry point
- [x] Created src/ directory structure
- [x] Created config.py with constants
- [ ] Extract Car class to car.py
- [ ] Extract CameraSensor to camera_sensor.py
- [ ] Extract PurePursuitLKA to lka_controller.py
- [ ] Extract SaoPauloTrack to track.py
- [ ] Extract Renderer3D to renderer.py
- [ ] Extract Minimap to minimap.py
- [ ] Extract HUD to hud.py
- [ ] Update imports and test
- [ ] Create requirements.txt
- [ ] Update README with new structure

## Note
For now, keeping robotics_lab_3d.py as the main working file.
The modular structure is prepared for future refactoring when needed.
