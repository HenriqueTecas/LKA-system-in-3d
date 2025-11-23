# LKA C++ Accelerators

High-performance C++ implementations of performance-critical LKA system components with OpenMP parallelization and SIMD optimizations.

## Performance Improvements

| Component | Python | C++ (Optimized) | Speedup |
|-----------|--------|-----------------|---------|
| Lane Detection | ~15-20ms | ~1-2ms | **10-20x** |
| MPC Controller | ~8-10ms | ~1-2ms | **5-10x** |
| Hybrid Controller | ~3-5ms | ~0.5-1ms | **3-5x** |
| Physics Simulation | ~2-3ms | ~0.5-1ms | **3-5x** |

**Overall System Performance**: Expected improvement from **~10 FPS → 100+ FPS**

## Architecture

### Parallelization Strategy

1. **Lane Detection** (`lane_detection.cpp`)
   - Left/right boundary processing with `#pragma omp sections`
   - Parallel distance computation with `#pragma omp parallel for`
   - Fast homography with Eigen matrix operations

2. **MPC Controller** (`mpc_controller.cpp`)
   - Parallel evaluation of 7 steering candidates with `#pragma omp parallel for`
   - Each candidate simulates 15-step trajectory independently
   - Dynamic scheduling for load balancing

3. **Hybrid Controller** (`hybrid_controller.cpp`)
   - Fast polynomial fitting with Eigen LDLT decomposition
   - Optimized with `-ffast-math` compiler flag
   - Rolling median filter for smooth steering

4. **Physics Engine** (`physics.cpp`)
   - SIMD-accelerated vector operations
   - Ackermann geometry computations
   - Simplified Pacejka tire model

### Compiler Optimizations

- `-O3`: Maximum optimization level
- `-march=native`: CPU-specific SIMD instructions (SSE/AVX)
- `-ffast-math`: Fast floating-point math (relaxed IEEE 754)
- OpenMP: Multi-core parallelization

## Building

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libeigen3-dev python3-dev

# macOS
brew install cmake eigen python3

# OpenMP (if not included with compiler)
# Ubuntu/Debian: included with gcc
# macOS: brew install libomp
```

### Build Instructions

```bash
cd cpp
./build.sh
```

The build script will:
1. Configure CMake with release optimizations
2. Compile all C++ sources with OpenMP and Eigen
3. Build Python bindings with pybind11
4. Install the module to the parent directory

### Manual Build

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp lka_cpp_accelerators*.so ../..
```

## Usage

### Drop-in Replacement

The C++ accelerators are designed as drop-in replacements for Python implementations:

```python
# Before (Pure Python)
from realistic_camera import RealisticCamera
from mpc_controller import MPCController
from hybrid_controller import HybridController

# After (C++ Accelerated)
from cpp_accelerators import CppLaneDetector, CppMPCController, CppHybridController

# Same API, 10x faster!
detector = CppLaneDetector(width, height, cam_height, pitch, fov)
left, right = detector.detect_lanes(track, car_pos, car_forward, ...)
```

### Integration with Existing Code

```python
# Check if C++ accelerators are available
from cpp_accelerators import CPP_AVAILABLE

if CPP_AVAILABLE:
    from cpp_accelerators import CppLaneDetector as LaneDetector
    print("Using C++ accelerated lane detection")
else:
    from realistic_camera import RealisticCamera as LaneDetector
    print("Using Python lane detection")
```

### Benchmarking

```python
from cpp_accelerators import PerformanceBenchmark

# Compare performance
PerformanceBenchmark.compare_lane_detection(iterations=1000)
PerformanceBenchmark.compare_mpc_controller(iterations=1000)
```

## Implementation Details

### Lane Detection Pipeline

**Python (Original)**:
- Nested loops for boundary interpolation
- Sequential point processing
- NumPy operations without parallelization

**C++ (Optimized)**:
```cpp
#pragma omp parallel sections
{
    #pragma omp section
    { /* Process left boundary */ }

    #pragma omp section
    { /* Process right boundary */ }
}
```

### MPC Controller

**Python (Original)**:
- Sequential evaluation of 7 steering candidates
- 105 forward simulations (7 × 15 steps)
- Pure Python loops

**C++ (Optimized)**:
```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < num_candidates; ++i) {
    costs[i] = evaluate_trajectory(candidates[i]);
}
```

### Memory Management

- Pre-allocated vectors with `reserve()`
- Minimal heap allocations in hot loops
- Copy elision and move semantics
- Stack allocation for small objects

### Threading

- OpenMP thread pool (persistent threads)
- Work-stealing scheduler for load balancing
- No GIL (Python Global Interpreter Lock) during C++ execution
- Thread count: `OMP_NUM_THREADS` environment variable (default: # of cores)

## Troubleshooting

### Module Not Found

```bash
# Ensure the module is built
cd cpp && ./build.sh

# Check if .so file exists
ls -la ../lka_cpp_accelerators*.so
```

### OpenMP Warnings

```bash
# Set thread count explicitly
export OMP_NUM_THREADS=4

# Disable OpenMP warnings
export OMP_DISPLAY_ENV=FALSE
```

### Performance Not as Expected

```bash
# Check CPU frequency scaling
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Should be "performance", not "powersave"

# Set to performance mode
sudo cpupower frequency-set -g performance
```

### Build Errors

```bash
# Missing Eigen
sudo apt-get install libeigen3-dev

# Missing OpenMP
sudo apt-get install libomp-dev

# Python headers not found
sudo apt-get install python3-dev
```

## Performance Tuning

### Thread Count

```bash
# Benchmark different thread counts
for n in 1 2 4 8; do
    export OMP_NUM_THREADS=$n
    echo "Threads: $n"
    python3 main.py --benchmark
done
```

### CPU Affinity

```bash
# Pin threads to specific cores
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

### SIMD Intrinsics

The code uses `-march=native` to enable all SIMD instructions supported by your CPU:
- **Intel**: SSE, SSE2, SSE3, SSSE3, SSE4, AVX, AVX2, AVX-512
- **AMD**: Same as Intel, plus AMD-specific optimizations
- **ARM**: NEON (if supported)

## Development

### Adding New Accelerated Functions

1. Add C++ implementation in `src/`
2. Add header in `include/`
3. Add Python binding in `bindings/bindings.cpp`
4. Rebuild with `./build.sh`

### Debugging

```bash
# Build with debug symbols
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Use gdb
gdb python3
(gdb) run main.py
```

### Profiling

```bash
# CPU profiling with perf
perf record -g python3 main.py
perf report

# Memory profiling with valgrind
valgrind --tool=massif python3 main.py
```

## License

Same as parent LKA system project.

## Credits

- **Eigen**: Linear algebra library
- **pybind11**: Python bindings
- **OpenMP**: Parallel programming
