# Car Model Documentation

## Overview

The Car class (src/car.py) implements a physics-based vehicle simulator using **Ackermann steering kinematics** combined with realistic longitudinal and lateral dynamics. The model uses standard automotive engineering formulas from vehicle dynamics theory.

**Key Features:**
- Ackermann steering geometry (bicycle model)
- Realistic longitudinal forces (drive, brake, drag, rolling resistance)
- Speed-dependent handling with tire grip limits
- Aerodynamic downforce increasing grip at high speeds
- Simple 2-DOF suspension dynamics (pitch and roll)
- First-order actuator dynamics for throttle, brake, and steering

---

## Physics Model

### 1. Coordinate System and Units

**All internal calculations use SI units:**
- Position: meters (m)
- Velocity: meters per second (m/s)
- Angle: radians (rad)
- Force: Newtons (N)
- Mass: kilograms (kg)

**Rendering uses pixels** (converted via `PIXELS_PER_METER = 12`).

### 2. Vehicle State Variables

| Variable | Unit | Description |
|----------|------|-------------|
| `x, y` | m | Position in world coordinates |
| `theta` | rad | Heading angle (yaw) |
| `velocity` | m/s | Longitudinal velocity |
| `steering_angle` | rad | Front wheel steering angle |
| `pitch` | rad | Body pitch angle (nose up/down) |
| `roll` | rad | Body roll angle (lean in turns) |

---

## Kinematics Formulas

### Low-Speed Ackermann Steering (`velocity < 5 m/s`)

Classic **bicycle model** for perfect no-slip turning:

-Force
\omega = \frac{v \cdot \tan(\delta)}{L}
-Force

Where:
- `ω` = yaw rate (rad/s)
- `v` = velocity (m/s)
- `δ` = steering angle (rad)
- `L` = wheelbase (2.7 m)

**Position update** (Euler integration):

-Force
x_{t+1} = x_t + v \cdot \cos(\theta) \cdot dt
-Force

-Force
y_{t+1} = y_t + v \cdot \sin(\theta) \cdot dt
-Force

-Force
\theta_{t+1} = \theta_t + \omega \cdot dt
-Force

---

### High-Speed Lateral Dynamics (`velocity  5 m/s`)

At higher speeds, tires can slip. The model computes **centripetal acceleration** and limits it by available grip:

**Turn radius from steering geometry:**

-Force
R = \frac{L}{|\tan(\delta)|}
-Force

**Lateral acceleration required:**

-Force
a_{lat} = \frac{v^2}{R}
-Force

**Maximum grip available** (with downforce):

-Force
a_{lat,max} = \min\left( \mu \cdot \frac{F_{normal}}{m}, \ 0.2g \right)
-Force

Where:
- `μ = 1.0` (tire friction coefficient)
- `F_normal = mg + F_downforce` (weight + downforce)
- Stability cap at `0.2g` lateral acceleration

**If demanded `a_lat > a_lat,max`**, the effective steering is reduced:

-Force
\delta_{eff} = \delta \cdot \frac{a_{lat,max}}{a_{lat}}
-Force

This simulates **understeer** (car doesn't turn as sharply as steering input suggests).

---

## Longitudinal Dynamics

### Forces Applied

Net longitudinal force:

-Force
F_{net} = F_{drive} + F_{brake} + F_{drag} + F_{rolling} + F_{engine\_braking} + F_{cornering\_drag}
-Force

#### 1. **Drive Force** (rear-wheel drive with traction limit)

-Force
F_{drive} = F_{rear\_normal} \cdot \mu \cdot \tanh\left(\frac{\text{throttle} \cdot F_{max\_drive}}{F_{rear\_normal} \cdot \mu}\right)
-Force

- Saturates smoothly using `tanh()` to prevent wheel spin
- `F_max_drive = 5000 N`

#### 2. **Brake Force**

-Force
F_{brake} = -\text{brake} \cdot F_{max\_brake} \cdot \text{sign}(v)
-Force

- `F_max_brake = 8000 N`
- Applied in direction opposing motion

#### 3. **Aerodynamic Drag**

-Force
F_{drag} = -\frac{1}{2} \rho \cdot C_d \cdot A \cdot v^2
-Force

- `ρ = 1.225 kg/m` (air density)
- `C_d = 0.3` (drag coefficient)
- `A = 2.2 m` (frontal area)

#### 4. **Rolling Resistance**

-Force
F_{rolling} = -C_{rr} \cdot m \cdot g \cdot \text{sign}(v)
-Force

- `C_rr = 0.015` (rolling resistance coefficient)

#### 5. **Engine Braking** (when coasting, `|throttle| < 0.05`)

-Force
F_{engine\_brake} = -(300 + 30 |v|) \cdot \text{sign}(v)
-Force

Increases with speed to simulate compression braking.

#### 6. **Cornering Drag** (energy lost in tire slip angles)

-Force
F_{cornering} = -0.15 \cdot m \cdot |\delta| \cdot v^2
-Force

Resists motion proportionally to steering angle and speed squared.

---

### Acceleration and Velocity Update

-Force
a = \frac{F_{net}}{m}
-Force

-Force
v_{t+1} = v_t + a \cdot dt
-Force

Velocity is clamped to `[-MAX_VELOCITY/2, MAX_VELOCITY]`.

---

## Aerodynamic Downforce

Downforce increases tire normal forces at speed, improving grip:

-Force
F_{downforce} = \frac{1}{2} \rho \cdot C_L \cdot A_{top} \cdot v^2
-Force

- `C_L = 0.2` (lift coefficient, negative for downforce)
- `A_top = 4.5 m` (top-down reference area)

**Normal force distribution:**
- Front: `40%` of downforce
- Rear: `60%` of downforce

Total normal force per axle:

-Force
F_{normal} = \frac{mg \cdot l_{opposite}}{L} + F_{downforce\_share}
-Force

Where `l_opposite` is distance from CG to opposite axle (weight transfer).

---

## Suspension Dynamics (2-DOF)

Simple rotational spring-damper system for pitch and roll.

### Pitch Dynamics (longitudinal acceleration)

-Force
\tau_{pitch} = m \cdot a_x \cdot h_{cg}
-Force

-Force
\alpha_{pitch} = \frac{\tau_{pitch} - k_{pitch} \cdot \theta_{pitch} - c_{pitch} \cdot \dot{\theta}_{pitch}}{I_{pitch}}
-Force

-Force
\dot{\theta}_{pitch,t+1} = \dot{\theta}_{pitch,t} + \alpha_{pitch} \cdot dt
-Force

-Force
\theta_{pitch,t+1} = \theta_{pitch,t} + \dot{\theta}_{pitch} \cdot dt
-Force

### Roll Dynamics (lateral acceleration)

-Force
\tau_{roll} = m \cdot a_{lat} \cdot h_{cg} \cdot \frac{w_{track}}{2}
-Force

Same spring-damper update as pitch.

**Parameters:**
- `k_pitch = 45000 Nm/rad` (pitch stiffness)
- `c_pitch = 5500 Nms/rad` (pitch damping)
- `k_roll = 55000 Nm/rad` (roll stiffness)
- `c_roll = 6000 Nms/rad` (roll damping)

---

## Actuator Dynamics

Real vehicles don't have instant response. Throttle, brake, and steering use **first-order lag filters**:

-Force
\text{actuator}_{t+1} = \text{actuator}_t + \frac{\text{command} - \text{actuator}_t}{\tau} \cdot dt
-Force

**Time constants (`τ`):**
- Throttle: `0.15 s`
- Brake: `0.10 s`
- Steering: `0.12 s`

Additionally, **rate limits** prevent actuators from changing faster than physically possible:
- Steering: `1.5 rad/s`
- Input shaping also applies deadbands and ramp rates for stability.

---

## Control Inputs

### Manual Control (Keyboard)

| Key | Action | Internal Command |
|-----|--------|------------------|
| `W` | Accelerate | `throttle = 1.0` |
| `S` | Brake (or reverse if stopped) | `brake = 1.0` or `throttle = -1.0` |
| `A` | Steer left | `steering_angle  +max` |
| `D` | Steer right | `steering_angle  -max` |

### LKA Override

If `lka_steering` is provided, it overrides manual steering with first-order tracking + rate limit.

### Hybrid ASSIST Override

`override_throttle` and `override_brake` allow the hybrid controller to take full control of pedals.

---

## Vehicle Parameters (BMW E36 Reference)

| Parameter | Value | Unit |
|-----------|-------|------|
| Mass | 1500 | kg |
| Wheelbase | 2.7 | m |
| Track width | 1.7 | m |
| CG height | 0.55 | m |
| Max steering angle | 30 | (0.524 rad) |
| Max velocity | 60 | m/s (~216 km/h) |
| Max drive force | 5000 | N |
| Max brake force | 8000 | N |
| Tire friction (μ) | 1.0 | (static) |
| Lateral accel limit | 0.2g | (stability cap) |

---

## Key Methods

| Method | Purpose |
|--------|---------|
| `update(dt, keys, lka_steering, override_throttle, override_brake)` | Main physics loop - updates all forces and state |
| `get_x_pixels() / get_y_pixels()` | Convert position to pixels for rendering |
| `get_front_axle_position()` | Returns front axle center (for camera mounting) |
| `get_hood_camera_position()` | Returns first-person camera position and look-at point |
| `is_on_track(track)` | Collision detection with track boundaries |
| `handle_collision()` | Revert to previous position and stop on collision |

---

## Summary

The car model uses **industry-standard vehicle dynamics equations**:

 **Ackermann steering** for low-speed kinematics  
 **Centripetal acceleration** with grip limits for high-speed handling  
 **Newtonian mechanics** for all forces (`F = ma`)  
 **Aerodynamic drag and downforce** formulas from fluid dynamics  
 **Spring-damper** suspension model  
 **First-order actuator dynamics** for realism  

This provides a **realistic yet computationally efficient** vehicle simulation suitable for lane-keeping control research and testing.
