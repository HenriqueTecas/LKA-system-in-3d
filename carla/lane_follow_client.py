"""
Minimal CARLA lane-following client for future labs.

Connects to a running CARLA server, spawns an ego car with a front RGB camera,
and runs a simple waypoint-based steering loop to stay in lane.

Tested version: carla==0.9.14 (not shipped here).
"""

import argparse
import math
import sys
import weakref

try:
    import carla
except ImportError:
    print("carla module not found. Install with: pip install carla==0.9.14")
    sys.exit(1)

try:
    import pygame
except ImportError:
    pygame = None


def get_blueprint(bp_lib, filter_str):
    bps = bp_lib.filter(filter_str)
    if not bps:
        raise RuntimeError(f"No blueprint matches filter: {filter_str}")
    return bps[0]


def world_density(world):
    """Return default vehicle spawn points; fallback to random location."""
    spawns = world.get_map().get_spawn_points()
    if spawns:
        return spawns
    raise RuntimeError("No spawn points available in this map.")


def yaw_to_heading(yaw_deg):
    """Yaw (deg) to unit heading vector in world frame."""
    rad = math.radians(yaw_deg)
    return math.cos(rad), math.sin(rad)


def angle_diff(a, b):
    """Shortest signed difference between two angles (radians)."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def compute_control(vehicle, lookahead=12.0, throttle_base=0.35):
    """
    Compute a simple pure-pursuit-like control to follow lane center using waypoints.
    Returns carla.VehicleControl.
    """
    world = vehicle.get_world()
    amap = world.get_map()
    transform = vehicle.get_transform()
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # m/s

    # Current lane waypoint
    waypoint = amap.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return carla.VehicleControl(throttle=0.0, brake=1.0)

    # Lookahead waypoint along the same lane
    wps = waypoint.next(lookahead)
    if not wps:
        wps = waypoint.next(5.0)
    target_wp = wps[0]

    # Heading vectors
    hx, hy = yaw_to_heading(transform.rotation.yaw)
    tx, ty = yaw_to_heading(target_wp.transform.rotation.yaw)

    # Vector from ego to target
    dx = target_wp.transform.location.x - transform.location.x
    dy = target_wp.transform.location.y - transform.location.y
    target_heading = math.atan2(dy, dx)
    ego_heading = math.atan2(hy, hx)

    # Steering from heading error
    heading_error = angle_diff(target_heading, ego_heading)
    steer_cmd = max(-1.0, min(1.0, heading_error / 0.35))  # scale for moderate gain

    # Simple speed policy
    desired_speed = 12.0  # m/s (~43 km/h)
    throttle = throttle_base
    brake = 0.0
    if speed > desired_speed + 1.0:
        throttle = 0.0
        brake = min(1.0, (speed - desired_speed) * 0.1)
    elif speed < desired_speed - 2.0:
        throttle = min(0.6, throttle_base + 0.2)

    return carla.VehicleControl(throttle=throttle, steer=steer_cmd, brake=brake)


class CameraManager:
    """Optional RGB camera display using pygame."""

    def __init__(self, vehicle, width=960, height=540):
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.sensor = None
        self.surface = None
        self.display = None

    def spawn(self, world):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", "90")
        transform = carla.Transform(
            carla.Location(x=1.5, z=1.4),
            carla.Rotation(pitch=-10),
        )
        self.sensor = world.spawn_actor(bp, transform, attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._process_image(weak_self, image))

    @staticmethod
    def _process_image(weak_self, image):
        self = weak_self()
        if not self or self.display is None:
            return
        array = image.raw_data
        if pygame:
            surf = pygame.image.frombuffer(array, (image.width, image.height), "RGB")
            self.surface = surf
            self.display.blit(self.surface, (0, 0))
            pygame.display.flip()

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="Optional town to load (e.g., Town04)")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera display")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    if args.town:
        world = client.load_world(args.town)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = get_blueprint(blueprint_library, "vehicle.tesla.model3")

    spawn_points = world_density(world)
    ego_transform = spawn_points[0]
    vehicle = world.spawn_actor(vehicle_bp, ego_transform)
    actor_list = [vehicle]

    cam_manager = None
    if not args.no_camera:
        cam_manager = CameraManager(vehicle)
        cam_manager.spawn(world)
        actor_list.append(cam_manager.sensor)
        if pygame:
            pygame.init()
            cam_manager.display = pygame.display.set_mode((cam_manager.width, cam_manager.height))
            pygame.display.set_caption("CARLA Lane Following Client")

    try:
        while True:
            world.tick()
            control = compute_control(vehicle)
            vehicle.apply_control(control)
            if pygame:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
    except KeyboardInterrupt:
        print("Exiting and cleaning up...")
    finally:
        if cam_manager:
            cam_manager.destroy()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        if pygame:
            pygame.quit()


if __name__ == "__main__":
    main()
