# Robotics Lab 3D - OpenGL Conversion

## Overview

This is a 3D OpenGL conversion of the robotics lab simulation that preserves all the original Ackermann kinematics and Pure Pursuit lane-keeping logic while adding immersive 3D visualization.

## Features

### Latest Enhancements ✨

- **✅ Performance Optimized** - Fixed freezing issues (reduced scenery by 50%, simplified rendering)
- **✅ Fixed Steering Controls** - A/D keys now work correctly for 3D hood camera view
- **✅ Fixed Collision Detection** - More forgiving boundaries, no more unexpected wall hits
- **🔧 Minimap Scaling (IN PROGRESS)** - Working on showing entire track scaled to fit (debug mode active)
- **✅ 3D Grass Texture** - Added alternating stripe and checkerboard patterns to terrain for depth perception
- **✅ Road Texture** - Subtle 3-tone gray pattern on road surface for better visual feedback
- **✅ Visual Track Features** - Checkpoints, sector markers, direction arrows, and start/finish line
- **✅ Optimized Scenery** - Trees, distance signs, and buildings for spatial awareness:
  - 🌲 **Trees**: Green foliage on brown trunks (optimized - every 8 points)
  - 🚏 **Distance Signs**: Orange markers at key corners (3 total)
  - 🏢 **Buildings**: 2 landmark buildings at strategic positions

### 3D Rendering
- **First-person hood camera view** - Experience the simulation from the driver's perspective
- **3D track visualization** - São Paulo F1 circuit rendered with realistic road surface and lane markings
- **Elevated terrain** - Green terrain walls around the track to clearly distinguish the drivable area
- **3D lane detection markers** - Visual spheres showing detected lane points in the 3D world
- **Track markers** - Checkpoint poles, sector numbers, and direction arrows

### Minimap (Top-Right Corner)
- **Full 2D simulation view** - 500x500px minimap showing the ENTIRE track scaled to fit
- Shows track layout, car position, camera FOV, detected lanes, and LKA lookahead point
- **Enhanced visibility** - Dark background, double border, and increased size
- **DEBUG MODE ACTIVE**: Currently showing grid lines, red bounds rectangle, scale factor, and coordinate labels to verify scaling is working correctly

### Preserved Logic
- ✅ **Ackermann steering kinematics** - Exact same physics model
- ✅ **Camera sensor model** - Same field of view, range, and detection logic
- ✅ **Pure Pursuit LKA controller** - Identical lane-keeping algorithm
- ✅ **All control parameters** - Speed, steering, lookahead distances unchanged

## Controls

- **W** - Accelerate
- **S** - Brake/Reverse
- **A** - Steer LEFT (deactivates LKA)
- **D** - Steer RIGHT (deactivates LKA)
- **F** - Toggle Lane Keeping Assist (LKA) on/off
- **ESC** - Exit simulation

**Note:** Steering controls are optimized for 3D hood camera perspective. From the driver's seat, A turns the wheel left and D turns it right, which feels natural in first-person view.

## Requirements

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate numpy
```

## Running the Simulation

### Option 1: Direct Run (if you have a display)
```bash
python3 robotics_lab_3d.py
```

### Option 2: Using the Helper Script (handles display setup)
```bash
./run_robotics_3d.sh
```
This script automatically detects your environment and uses Xvfb if needed.

### Option 3: For Headless/SSH Environments
```bash
# Install Xvfb if not already installed
sudo apt-get install xvfb

# Run with virtual display
xvfb-run -s "-screen 0 1920x1080x24" python3 robotics_lab_3d.py
```

## Troubleshooting

If you encounter display or OpenGL errors, see **[TROUBLESHOOTING_3D.md](TROUBLESHOOTING_3D.md)** for detailed solutions.

Common quick fixes:
```bash
# For "Could not get EGL display" error
export SDL_VIDEO_X11_FORCE_EGL=0
python3 robotics_lab_3d.py

# For hardware acceleration issues
export LIBGL_ALWAYS_SOFTWARE=1
python3 robotics_lab_3d.py

# For SSH sessions
ssh -X user@host
python3 robotics_lab_3d.py
```

## Technical Details

### Architecture
- **Pygame + PyOpenGL hybrid** - Combines Pygame for event handling and 2D overlays with OpenGL for 3D rendering
- **Modern OpenGL** - Uses OpenGL fixed pipeline for simplicity and compatibility
- **Efficient rendering** - Separates 3D scene rendering from 2D HUD overlay

### Key Components

1. **Car Class** - Ackermann steering model with 3D rendering methods
2. **CameraSensor Class** - Lane detection with original FOV and range logic
3. **PurePursuitLKA Class** - Pure Pursuit algorithm for autonomous lane keeping
4. **SaoPauloTrack Class** - Track layout with 3D rendering (flat road + elevated terrain)
5. **Renderer3D Class** - OpenGL 3D scene management and lighting
6. **Minimap Class** - 2D top-down view reusing original visualization code
7. **HUD Class** - Heads-up display showing telemetry and status

### Visual Elements

#### 3D Scene
- **Road Surface**: Dark gray asphalt with white lane boundaries and yellow dashed centerline
- **Terrain Walls**: Green elevated walls (30-unit height) clearly marking track boundaries
- **Lane Detection Markers**:
  - Red spheres for left lane boundary detections
  - Cyan spheres for right lane boundary detections
  - Yellow spheres for center dotted line detections
  - Large yellow sphere for LKA lookahead point
- **Track Features**:
  - **Checkpoint Markers**: Cyan poles with spheres at track sides (every 6 points)
  - **Sector Numbers**: Colored floating spheres above track indicating sector/segment
  - **Direction Arrows**: Yellow arrows on track surface showing driving direction
  - **Start/Finish Line**: Red and white tall poles marking the start/finish
- **Scenery Elements** (OPTIMIZED):
  - **Trees**: Green spherical foliage on brown trunks, placed every 8 points alternating sides
  - **Distance Signs**: Orange posts with colored spheres at 3 key positions
  - **Buildings**: 2 landmark buildings with different colors (brown, red-gray)
    - Varying heights (30-45 units) for easy identification
    - Windows for realism
    - Positioned at strategic corners (8, 18)
  - All scenery optimized for smooth 60 FPS performance
- **Collision Detection**: Invisible walls at track boundaries prevent off-track driving

#### Minimap (400x400px) - Exact Match of 2D Implementation
- Track outline with lane markings and dashed centerline
- Camera FOV cone (semi-transparent green) with edge lines
- Camera position marker (green circle)
- Front wheel positions (orange left wheel, cyan right wheel)
- Detected lane points:
  - Red circles for left lane boundary
  - Cyan circles for right lane boundary
  - Dark blue circles for center dotted line
  - Orange vectors from left wheel to left lane points
  - Cyan vectors from right wheel to right lane points
- Car representation with main axis and heading indicator
- LKA lookahead point and path (when active)

#### HUD
- LKA status (ACTIVE in green / OFF in red)
- Speed display
- Steering angle display
- Lane detection status (current lane, left/right detection)
- Control hints at bottom

## Differences from Original

### Added
- 3D first-person hood camera view
- 3D terrain visualization
- 3D lane detection markers
- Perspective projection and lighting
- Minimap in corner showing original 2D view

### Unchanged
- All physics calculations
- All control logic
- All detection algorithms
- All parameters and gains

## Performance

- Target: 60 FPS
- Optimized for real-time interaction
- Efficient OpenGL rendering with lighting and depth testing

## Future Enhancements (Optional)

- Add elevation changes to the road surface itself
- Implement multiple camera views (chase cam, top-down, etc.)
- Add more detailed car model
- Include track-side objects (barriers, signs, trees)
- Add motion blur or other visual effects
- Implement picture-in-picture camera view showing raw sensor feed

## Credits

Based on the original 2D pygame simulation with Ackermann steering kinematics and Pure Pursuit lane keeping controller.
