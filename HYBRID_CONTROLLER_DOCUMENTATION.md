# Hybrid Lane Controller Documentation

## Overview

The HybridLaneController class (src/hybrid_controller_clean.py) implements a **minimal, production-ready lane-keeping assist system** with three operating modes:

1. **MANUAL** - Controller inactive, driver has full control
2. **WARNING** - Visual/audible warnings only, no intervention
3. **ASSIST** - Active intervention with "green zone" driver-friendly logic

**Philosophy:** In ASSIST mode, the system only intervenes when necessary (outside the "green zone" or when warnings are active), allowing natural driver control when centered and aligned.

---

## Control Modes

### Mode 0: MANUAL
- Controller completely inactive
- All warnings suppressed
- No steering/throttle/brake commands issued

### Mode 1: WARNING
- Monitors lane position and speed
- Shows warnings when hazards detected:
  - `lane_departure` - lateral offset > 1.2m
  - `speed_too_high` - velocity > safe_speed  1.05
  - `time_to_crossing` - time to cross lane edge < 1.0s
- **Green zone applies:** Suppresses lane/time warnings when centered (offset < 1.2m, heading < 0.8 rad)
- Speed warnings always shown regardless of green zone
- **No control commands issued**

### Mode 2: ASSIST
- Same warnings as WARNING mode
- **Actively intervenes** when:
  - Outside green zone AND warnings present
  - Speed warning overrides green zone (immediate braking)
- Returns control to driver when:
  - Inside green zone AND no warnings active
- **Control outputs:** Steering, throttle, and brake commands

---

## Green Zone Logic

The "green zone" is a **comfort region** where the car is well-centered and pointing straight. When inside this zone, ASSIST mode gives control back to the driver.

### Green Zone Boundaries (Tunable)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `assist_offset_deadband` | 1.2 m | Lateral tolerance before assist engages |
| `assist_heading_deadband` | 0.8 rad | Heading error tolerance before assist engages |

**Inside green zone if:**

UTF8
|\text{lane\_offset}| < 1.2 \text{ m} \quad \text{AND} \quad |\text{heading\_error}| < 0.8 \text{ rad}
UTF8

**When in green zone:**
- `intervening = False`
- Lane departure and time-to-crossing warnings suppressed
- Speed warnings still shown (safety-critical)
- No steering/throttle/brake commands issued

**Speed warning exception:** If `speed_too_high` is active, the system **forces exit from green zone** and applies braking immediately, even if position/heading are centered.

---

## Steering Control (Pure Pursuit)

### Algorithm: Adaptive Lookahead Pure Pursuit

The controller uses a **robust pure-pursuit** algorithm with several enhancements:

#### 1. Centerline Construction

**If both lane boundaries visible:**

UTF8
\text{center}[i] = \left( \frac{\text{left}[i] + \text{right}[i]}{2} \right)
UTF8

**If only one boundary visible:**
- Fit 2nd-order polynomial to boundary
- Offset by half lane width (2.0m) perpendicular to curve
- Creates virtual centerline

#### 2. Curvature-Adaptive Lookahead

Base lookahead distance:

UTF8
L_{base} = \text{clamp}(v \times 0.5, \ 5.0, \ 22.0) \quad \text{[meters]}
UTF8

**Curvature proxy** (triangle area method):

UTF8
\kappa_{proxy} = \frac{\text{Area}(\text{p1}, \text{p2}, \text{p3})}{\text{baseline}^3}
UTF8

**Lookahead adjustment:**
- High curvature (`κ > 0.002`): `L = max(4.0, L_base  0.30)`
- Medium curvature (`κ > 0.001`): `L = max(5.0, L_base  0.50)`
- Low curvature: `L = L_base`

Tighter curves  shorter lookahead  more responsive steering.

#### 3. Target Point Selection

Find the point along the centerline at arc length `L` from the car:

1. Start from nearest forward point
2. Walk along centerline segments
3. Interpolate when remaining distance < segment length
4. If centerline ends before `L`, use last point

#### 4. Steering Angle Calculation

**Lateral error** (perpendicular distance to target):

UTF8
e_{lat} = \Delta x \cdot (-\sin\theta) + \Delta y \cdot \cos\theta
UTF8

**Heading error** (angle to target):

UTF8
e_{heading} = \arctan2(\Delta y, \Delta x) - \theta
UTF8

**Combined error** (weighted):

UTF8
e_{combined} = e_{lat} + e_{heading} \cdot h_{img} \cdot w_{heading}
UTF8

Where:
- `h_img = 14.0` (image height equivalent in meters)
- `w_heading = 0.25` (heading correction weight)

**Raw steering** (geometric pure pursuit):

UTF8
\delta_{raw} = 90 - \arctan\left(\frac{h_{img}}{e_{combined}}\right)
UTF8

#### 5. Smoothing (Median + Low-Pass Filter)

To prevent oscillations:

**Median filter** (window = 6 frames):

UTF8
\delta_{med} = \text{median}(\delta_{raw}, \delta_{t-1}, \ldots, \delta_{t-5})
UTF8

**Low-pass filter** (`α = 0.1`):

UTF8
\delta_{filtered} = 0.1 \cdot \delta_{med} + 0.9 \cdot \delta_{prev}
UTF8

Final steering command is clamped to `max_steering_angle`.

---

## Speed Control (Predictive Curve Braking)

### Algorithm: Safe Speed Planning with Curvature Prediction

The controller predicts upcoming curve radius and computes a safe speed, then brakes if needed.

#### 1. Curve Radius Estimation

**Primary method:** 2nd-order polynomial fit to lane centerline:

UTF8
y = ax^2 + bx + c
UTF8

**Curvature** at midpoint:

UTF8
\kappa = \frac{|d^2y/dx^2|}{(1 + (dy/dx)^2)^{3/2}} = \frac{2|a|}{(1 + (2ax_{mid} + b)^2)^{3/2}}
UTF8

**Radius:**

UTF8
R = \frac{1}{\kappa}
UTF8

**Fallback methods** (if polynomial fit fails):
1. Use stored centerline from steering calculation
2. 3-point circle fit:

UTF8
R = \frac{|AB \times AC|}{2 \cdot |BC|}
UTF8

3. Conservative default: `R = 120 m`

#### 2. Safe Speed Calculation

**Lateral acceleration limit:**

UTF8
a_{lat,limit} = 0.2g = 1.962 \text{ m/s}^2
UTF8

**Safe speed from physics:**

UTF8
v_{safe} = \sqrt{a_{lat,limit} \cdot R} \times \text{safe\_speed\_scale}
UTF8

Where `safe_speed_scale = 0.8` (global multiplier for more conservative speed).

Clamped to `[0, max_velocity]`.

#### 3. Braking Decision

**Trigger braking if:**

UTF8
R < 650 \text{ m} \quad \text{AND} \quad v_{current} > v_{safe} \times 0.8
UTF8

**Brake command intensity:**

UTF8
\text{brake} = \text{clamp}\left(\frac{v_{current} - v_{safe} \times 0.8}{v_{safe}}, \ 0.0, \ 1.0\right)
UTF8

**Acceleration decision:**

If `v_current < v_safe  0.8  1.0`, allow throttle:

UTF8
\text{throttle} = \text{clamp}\left(\frac{v_{safe} - v_{current}}{v_{safe}}, \ 0.0, \ 1.0\right)
UTF8

---

## Warning System

### Warning Triggers

| Warning | Condition | Description |
|---------|-----------|-------------|
| `speed_too_high` | `v > v_safe  1.05` | Exceeding safe curve speed |
| `lane_departure` | `|offset| > 1.2 m` OR `TTC < 1.0 s` | Drifting out of lane |
| `time_to_crossing` | `TTC < 1.0 s` | About to cross lane edge |
| `assist_on` | `mode == ASSIST` | Assist mode active |
| `assist_intervening` | `mode == ASSIST` AND `intervening == True` | Computer is controlling |

**Time-to-crossing (TTC):**

UTF8
\text{TTC} = \frac{|\text{lane\_offset}|}{v \cdot |\tan(\delta)|}
UTF8

Where lateral velocity is approximated as `v_lat = v  tan(δ)`.

### Green Zone Warning Suppression

**In WARNING mode:**
- If inside green zone: suppress `lane_departure` and `time_to_crossing`
- Speed warnings always shown

**In ASSIST mode:**
- Same suppression logic
- If `speed_too_high` is active, force exit from green zone (`centered = False`)

---

## Intervention Logic (ASSIST Mode Only)

### Engagement Conditions

**Immediate engagement:**
- Speed planner recommends braking (`action == "brake"`)
- Any warning active (speed, lane departure, time to crossing)

**Hysteresis (stability counter):**
- Hazard must be present for `3` consecutive frames before engaging
- Prevents jitter from noisy detections

### Release Conditions

**Must be stable for `5` consecutive frames:**

UTF8
|\text{lane\_offset}| < 0.25 \text{ m} \quad \text{AND} \quad |\text{heading\_error}| < 0.09 \text{ rad} \quad \text{AND} \quad v < v_{safe} \times 1.08
UTF8

Release thresholds are **tighter than engagement** to prevent rapid engage/disengage cycles.

### Control Commands When Intervening

#### Steering
Always applied when intervening (pure pursuit target).

#### Throttle/Brake
**Priority 1: Speed warning or brake recommendation**

If `speed_too_high` OR `speed_cmd.action == "brake"`:

UTF8
\text{throttle} = 0.0
UTF8

UTF8
\text{brake} = \max\left(\text{brake}_{overspeed}, \ \text{brake}_{speed\_cmd}\right)
UTF8

Where:

UTF8
\text{brake}_{overspeed} = \text{clamp}\left(\frac{v - v_{safe} \times 0.8}{v_{safe}}, \ 0.0, \ 1.0\right)
UTF8

**Priority 2: Lane departure only**

If only `lane_departure` or `time_to_crossing` warnings:

UTF8
\text{throttle} = \texttt{None} \quad \text{(driver controls speed)}
UTF8

UTF8
\text{brake} = \texttt{None}
UTF8

System applies **steering correction only**, letting driver control pedals.

**Priority 3: No intervention**

If `intervening == False` and `speed_cmd.action == "accelerate"`:

UTF8
\text{throttle} = \text{speed\_cmd.throttle}
UTF8

UTF8
\text{brake} = \texttt{None}
UTF8

---

## Tunable Parameters

### Green Zone Size

`python
self.assist_offset_deadband = 1.2  # meters - increase for larger green zone
self.assist_heading_deadband = 0.8  # radians - increase for more heading tolerance
`

### Speed Planning Aggressiveness

`python
self.safe_speed_scale = 0.8  # [0.6-1.0] - lower = slower in curves
self.lateral_accel_limit = 0.2 * 9.81  # [0.15-0.25]g - lower = slower
self.curve_entry_threshold = 650.0  # meters - higher = treat gentler curves as hazards
self.comfort_margin = 0.8  # [0.7-0.9] - lower = earlier braking
self.brake_threshold = 0.8  # [0.7-0.9] - lower = earlier braking trigger
`

### Warning Thresholds

`python
self.warning_lateral_offset = 1.2  # meters - distance before lane departure warning
self.warning_speed_margin = 1.05  # multiplier - speed margin before warning
`

### Intervention Engagement

`python
self.intervention_lane_offset = 0.5  # meters - tighter than warning threshold
self.heading_target_window_enter = 0.18  # rad - heading error to engage
self.heading_target_window_release = 0.09  # rad - heading error to release (tighter)
self.speed_overshoot_margin = 1.08  # multiplier - speed margin before engage
self.hazard_stable_frames = 3  # frames - require stable hazard before engaging
self.release_stable_frames = 5  # frames - require stable safe state before releasing
`

### Steering Smoothing

`python
self.rolling_median_window = 6  # frames - more = smoother but slower response
self.steering_lpf_alpha = 0.1  # [0.05-0.3] - lower = smoother
`

---

## Key Methods

| Method | Purpose |
|--------|---------|
| `set_mode(mode)` | Switch between MANUAL/WARNING/ASSIST |
| `calculate_control(track)` | Main loop - returns `(steering, throttle, brake, warnings, intervening)` |
| `_calculate_steering_direction(...)` | Pure pursuit steering with adaptive lookahead |
| `_calculate_speed_control(...)` | Predictive curve braking |
| `_update_warnings(...)` | Evaluate warning conditions |
| `_update_intervention_state(...)` | Decide when to engage/disengage in ASSIST mode |
| `_estimate_lane_offset()` | Perpendicular distance from car to centerline |
| `_predict_curve_radius_from_lane(...)` | Polynomial fit to predict curve radius |

---

## Algorithm Summary

`

  1. Detect lane boundaries (camera)                 
  2. Construct centerline (average or virtual)       
  3. Compute steering (pure pursuit + smoothing)     
  4. Predict curve radius (polynomial fit)           
  5. Calculate safe speed (v = sqrt(a_lat  R))      
  6. Check warnings (speed, lane departure, TTC)     
  7. Green zone check (centered?)                    
  8. [ASSIST only] Update intervention state         
  9. [ASSIST only] Apply steering/throttle/brake     

`

---

## Safety Features

 **Green zone prevents over-intervention** - Driver comfort in normal conditions  
 **Speed warnings always shown** - Safety-critical even when centered  
 **Hysteresis on engage/release** - Prevents rapid oscillations  
 **Selective throttle control** - Only cuts throttle on speed warnings, not lane drift  
 **Fallback curve radius** - Conservative default (120m) when prediction fails  
 **Multi-layer smoothing** - Median + LPF prevents steering oscillations  
 **Tighter release thresholds** - Ensures stable return to manual control  

---

## Typical Use Cases

### Use Case 1: Normal Driving (Centered)
- Mode: ASSIST
- Car centered, aligned with lane
- **Result:** `intervening = False`, no commands issued, driver has full control

### Use Case 2: Drifting Slightly
- Mode: ASSIST
- Car offset 0.8m, still within green zone
- **Result:** `intervening = False`, lane warning suppressed, driver has control

### Use Case 3: Excessive Drift
- Mode: ASSIST
- Car offset 1.5m, outside green zone
- **Result:** `lane_departure = True`, `intervening = True`, steering correction applied, throttle/brake returned to driver

### Use Case 4: Entering Curve Too Fast
- Mode: ASSIST
- Curve radius = 200m, speed = 25 m/s, safe speed = 20 m/s
- **Result:** `speed_too_high = True`, `intervening = True`, steering correction + braking applied, throttle cut to 0

### Use Case 5: Warning Only
- Mode: WARNING
- Car drifting or speeding
- **Result:** Warnings shown on HUD, no control commands, driver maintains full control

---

This system balances **safety** (intervening when needed) with **driver autonomy** (staying out of the way when not needed).
