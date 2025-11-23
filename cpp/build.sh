#!/bin/bash

# Build script for LKA C++ accelerators

set -e  # Exit on error

echo "========================================"
echo "Building LKA C++ Accelerators"
echo "========================================"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math" \
    -DPython_EXECUTABLE=$(which python3)

# Build
echo ""
echo "Building (using all available cores)..."
make -j$(nproc)

# Install to parent src directory for easy import
echo ""
echo "Installing Python module..."
INSTALL_DIR="../.."
cp lka_cpp_accelerators*.so "$INSTALL_DIR/"

echo ""
echo "========================================"
echo "Build complete!"
echo "Module installed to: $INSTALL_DIR"
echo "========================================"
echo ""
echo "You can now import with: import lka_cpp_accelerators"
