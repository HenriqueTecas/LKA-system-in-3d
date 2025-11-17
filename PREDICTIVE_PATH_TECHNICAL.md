# Predictive Path Extension - Technical Details

## Overview

This document explains the **curvature-based predictive path extension** implemented in the LKA controller. This approach is used by production autonomous vehicles (Tesla, Waymo, MobilEye) to handle situations where lane markings temporarily leave the camera's field of view.

## The Problem

During tight turns, lane markings can move outside the camera's FOV before the turn is complete. Without prediction, the LKA system loses its target points and can't steer properly.

## Why Not Use "Memory"?

**Simple memory is unrealistic** because:
- Real cameras only see what's currently in frame
- Without continuous visual feedback, you can't reliably track where old points are
- Assumes perfect odometry (knowing exactly how you've moved)
- Not representative of real autonomous vehicle challenges

## The Realistic Solution: Geometric Prediction

Instead of remembering old points, we **predict where the lane should go** based on the **shape** (curvature) of the currently visible lane.

## Algorithm Steps

### Step 1: Transform to Ego Frame

Convert world coordinates to the car's local frame (ego frame):
- **x-axis**: Forward along car heading
- **y-axis**: Left/right lateral position

```python
dx = world_x - car_x
dy = world_y - car_y

cos_th = np.cos(-car_theta)
sin_th = np.sin(-car_theta)

x_ego = dx * cos_th - dy * sin_th
y_ego = dx * sin_th + dy * cos_th
```

This is standard in robotics (Frenet-Serret frames) and exactly what real LKA systems do internally.

### Step 2: Estimate Curvature

We use **polynomial fitting** (Option B from the original design):

**Lane Model**: `y(x) = ax² + bx + c`

Where:
- `x` = longitudinal distance ahead
- `y` = lateral offset from centerline
- `a, b, c` = polynomial coefficients

**Curvature Estimation**:
```python
# Fit quadratic to lane points
coeffs = np.polyfit(x_points, y_points, deg=2)
a = coeffs[0]  # x² coefficient

# Curvature at x=0 (car position)
κ = 2 * a
```

**Physical Meaning**:
- `κ > 0`: Lane curves to the left
- `κ < 0`: Lane curves to the right
- `κ ≈ 0`: Lane is nearly straight

### Step 3: Project Path Forward

Using the estimated polynomial, generate predicted points by **directly evaluating the polynomial**:

**Direct Polynomial Evaluation** (CORRECT approach):
```python
# Fit polynomial: y(x) = ax² + bx + c
coeffs = np.polyfit(x_points, y_points, deg=2)
a, b, c = coeffs

# Find furthest visible point
furthest_x = max([x for x, y in forward_points])

# Generate predictions by evaluating polynomial forward
for x_pred in range(furthest_x, furthest_x + horizon, step):
    y_pred = a * x_pred**2 + b * x_pred + c
    predicted_points.append((x_pred, y_pred))
```

This is **correct** because:
- ✅ Smoothly continues the fitted curve
- ✅ No coordinate frame confusion
- ✅ Simple and mathematically sound
- ✅ Predictions lie exactly on the fitted polynomial### Step 4: Transform Back to World Frame

Convert predicted ego-frame points back to world coordinates:

```python
cos_th = np.cos(car_theta)
sin_th = np.sin(car_theta)

for x_ego, y_ego in predicted_points:
    # Rotate to world frame
    dx = x_ego * cos_th - y_ego * sin_th
    dy = x_ego * sin_th + y_ego * cos_th

    # Translate to world position
    world_x = car_x + dx
    world_y = car_y + dy
```

### Step 5: Combine with Detected Points

Merge the predicted points with the actual detected lane points:

```python
extended_points = detected_points + predicted_points
```

The Pure Pursuit controller then uses all available points (detected + predicted) to select the best lookahead target.

## Why This Is Realistic

✅ **Only uses current observations**: No arbitrary "memory"
✅ **Geometric reasoning**: Assumes smooth road curvature (physically valid)
✅ **Graceful degradation**: Predictions update as new data arrives
✅ **Production-proven**: Tesla, Waymo, MobilEye use similar approaches
✅ **Physically plausible**: Roads have continuous curvature
✅ **Sensor-appropriate**: Works within camera limitations

## Configuration Parameters

Located in `src/lka_controller.py`:

```python
self.enable_prediction = True           # Toggle prediction on/off
self.prediction_horizon = 150.0         # How far to predict (pixels)
self.prediction_step = 10.0             # Spacing between points (pixels)
self.curvature_sample_points = 8        # Points used for curvature fit
```

## Comparison to Alternatives

| Approach | Realistic? | Pros | Cons |
|----------|-----------|------|------|
| **Memory** | ❌ No | Simple | Violates sensor constraints, assumes perfect odometry |
| **Curvature Prediction** | ✅ Yes | Realistic, production-proven | Requires math, assumes smooth roads |
| **Model Predictive Control** | ✅ Yes | Very advanced | Complex, computationally expensive |
| **Wider FOV** | ✅ Yes | Dead simple | Still fails on very tight turns |

## Real-World Analogy

Think of driving at night with headlights:
- **Memory**: "I remember there was a curve 50 meters back, so I'll steer based on that" (unrealistic - you've moved!)
- **Prediction**: "The road is curving left based on what I can see now, so it probably continues curving left ahead" (realistic!)

## Mathematical Background

This approach is based on:
- **Frenet-Serret frames**: Standard in robotics path planning
- **Clothoid curves**: Roads are designed with continuous curvature
- **Polynomial approximation**: Common in computer vision lane detection
- **Arc length parameterization**: Natural way to describe curves

## References

- Tesla Autopilot: Uses polynomial lane models + predictive extension
- Waymo: Multi-hypothesis path prediction with geometric priors
- MobilEye: EyeQ chips implement curvature-based lane tracking
- Academic: "A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles" (IEEE, 2016)

## Testing the Feature

Run the simulation and observe:
1. Yellow dots show detected lane points
2. Additional yellow dots appear ahead during curves (predicted points)
3. LKA stays active even when lane markings temporarily leave FOV
4. Steering remains smooth through tight turns

Toggle prediction on/off in `src/lka_controller.py`:
```python
self.enable_prediction = False  # Disable to see the difference
```

## Conclusion

This implementation provides **realistic**, **sensor-appropriate** path prediction that maintains the simulation's fidelity to real autonomous vehicle behavior. It solves the tight turn problem without violating the constraints of a vision-based system.
