"""
Configuration constants for the 3D robotics simulation.
All physical quantities use SI units (meters, kg, seconds, radians).
"""

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
# Screen dimensions
WIDTH = 1600
HEIGHT = 900
MINIMAP_SIZE = 500
FPS = 60

# Colors (RGB tuples)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# ============================================================================
# UNIT CONVERSION
# ============================================================================
# Pixels to meters conversion. Rendering uses pixels, physics uses meters.
# 12 pixels = 1 meter for São Paulo F1 circuit scale
# Track width: 120 pixels = 10 meters (realistic F1 track)
# Lane width: 60 pixels = 5 meters per lane
PIXELS_PER_METER = 12

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
# Fixed physics timestep (seconds). Use e.g. 0.01 for 100 Hz physics updates.
PHYSICS_DT = 0.01

# ============================================================================
# VEHICLE GEOMETRY (SI Units)
# ============================================================================
# Mass (kg)
VEHICLE_MASS = 1500.0
# Moment of inertia around vertical axis (kg·m²)
# Calculated using: I_z = (1/12) * m_body * (L² + W²) + m_wheels * r_wheels²
# For BMW E36: ~2697 (body) + ~252 (wheels) ≈ 2950 kg·m²
VEHICLE_INERTIA_Z = 2949.0
# Wheelbase (m) - distance between front and rear axles
VEHICLE_WHEELBASE = 2.7
# Track width (m) - distance between left and right wheels
VEHICLE_TRACK_WIDTH = 1.7
# Center of gravity position (m from rear axle)
VEHICLE_CG_TO_FRONT = 1.2
VEHICLE_CG_TO_REAR = 1.5

# ============================================================================
# TIRE PARAMETERS
# ============================================================================
# Cornering stiffness front/rear (N/rad) - linear model
TIRE_CORNERING_STIFFNESS_FRONT = 80000.0
TIRE_CORNERING_STIFFNESS_REAR = 80000.0

# ============================================================================
# FORCES AND RESISTANCES
# ============================================================================
# Gravitational acceleration (m/s²)
GRAVITY = 9.81

# Aerodynamics
AIR_DENSITY = 1.225  # kg/m³
AERO_CD = 0.3  # Drag coefficient (dimensionless)
AERO_AREA = 2.2  # Frontal area (m²)
AERO_CL = 0.2  # Downforce coefficient (0.0=road car, 0.5-1.5=sports car, 2.0-4.0=race car)
AERO_DOWNFORCE_AREA = 4.5  # Top-down reference area (m²)

# Rolling resistance
ROLLING_RESISTANCE_COEFF = 0.015  # Dimensionless coefficient

# ============================================================================
# DRIVETRAIN AND ACTUATORS
# ============================================================================
# Engine/drivetrain forces
MAX_DRIVE_FORCE = 6000.0  # N (maximum drive force at wheels)
MAX_BRAKE_FORCE = 12000.0  # N (maximum braking force at wheels)

# Actuator time constants (seconds) - first-order response
THROTTLE_TAU = 0.12
BRAKE_TAU = 0.08
STEERING_TAU = 0.2

# ============================================================================
# STEERING SYSTEM
# ============================================================================
MAX_STEERING_ANGLE = 0.61  # radians (~35 degrees)
MAX_STEERING_RATE = 1.05  # rad/s (~60 deg/s)

# ============================================================================
# PERFORMANCE LIMITS
# ============================================================================
MAX_VELOCITY = 33.3  # m/s (approximately 120 km/h)

# ============================================================================
# CAMERA SENSOR MODEL
# ============================================================================
CAMERA_FRAME_RATE = 30  # Hz (frames per second)
CAMERA_LATENCY_MS = 50  # milliseconds (sensor latency)
CAMERA_NOISE_STD = 0.05  # meters (spatial noise standard deviation, 5cm)

# ============================================================================
# LKA CONTROLLER TUNING
# ============================================================================
# Exponential smoothing factor for lookahead point (0=no smoothing, 1=instant)
LKA_LOOKAHEAD_SMOOTHING_ALPHA = 0.2
# Minimum Euclidean change (meters) required to snap immediately to new lookahead
LKA_LOOKAHEAD_SNAP_THRESHOLD = 0.1  # 10cm

