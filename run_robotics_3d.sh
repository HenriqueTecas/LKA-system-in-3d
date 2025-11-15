#!/bin/bash
# Helper script to run the 3D robotics simulation with proper OpenGL setup

echo "=== Robotics Lab 3D - OpenGL Simulation ==="
echo ""

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "⚠ No DISPLAY environment variable found!"
    echo ""
    echo "Attempting to use virtual display (Xvfb)..."

    # Check if Xvfb is installed
    if command -v Xvfb &> /dev/null; then
        echo "✓ Xvfb found, starting virtual display..."
        xvfb-run -s "-screen 0 1920x1080x24" python3 robotics_lab_3d.py
    else
        echo "✗ Xvfb not installed!"
        echo ""
        echo "Please install Xvfb:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install xvfb"
        echo ""
        echo "Or run from a system with a display."
        exit 1
    fi
else
    echo "✓ Display found: $DISPLAY"
    echo ""

    # Set OpenGL environment variables for better compatibility
    export SDL_VIDEO_X11_FORCE_EGL=0
    export PYOPENGL_PLATFORM=glx

    # Optional: Use software rendering if hardware acceleration fails
    # Uncomment the line below if you have issues
    # export LIBGL_ALWAYS_SOFTWARE=1

    echo "Starting simulation..."
    python3 robotics_lab_3d.py
fi
