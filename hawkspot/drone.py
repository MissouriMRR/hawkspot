from typing import Optional, cast
import logging
import math

import dronekit
from pymavlink import mavutil
import time

class Drone:
    def __init__(self, vehicle: dronekit.Vehicle):
        self.vehicle = vehicle
        self.yaw=0.0

    def goto_local(self, point: dronekit.LocationLocal):
        #current_yaw = self.vehicle.attitude.yaw  # radians, already in NED frame

        self.vehicle._master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (not used)
            self.vehicle._master.target_system,  # target_system
            self.vehicle._master.target_component,  # target_component
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # ty: ignore[unresolved-attribute]  # frame
            0b0000101111111000,  # type_mask (only positions enabled)
            point.north,
            point.east,
            0,  # x, y, z positions in meters (NED: negative z is up)
            0,
            0,
            0,  # vx, vy, vz (ignored)
            0,
            0,
            0,  # afx, afy, afz (ignored)
            0,
            0,  # yaw, yaw_rate (ignored)
        )



    def set_relative_yaw_and_wait(
        self,
        yaw_change_deg,
        angular_speed_deg_s=30,
        tolerance_deg=2.0,
        timeout=15.0,
    ):
        """
        Rotate the vehicle by a relative yaw amount and block until reached.

        yaw_change_deg:      degrees to rotate (positive = clockwise,
                            negative = counter-clockwise)
        angular_speed_deg_s: rotation speed (deg/s)
        tolerance_deg:       how close to target before considered "reached"
        timeout:             max seconds to wait before giving up

        Returns True if yaw reached, False if timed out.
        """
        # MAV_CMD_CONDITION_YAW needs a positive magnitude + direction flag
        direction = 1 if yaw_change_deg >= 0 else -1
        magnitude = abs(yaw_change_deg)

        # Compute absolute target for the wait loop
        current_yaw_deg = math.degrees(self.vehicle.attitude.yaw) % 360
        absolute_target = (current_yaw_deg + yaw_change_deg) % 360

        # Send MAV_CMD_CONDITION_YAW
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,                                       # target system, component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,      # command id
            0,                                          # confirmation
            magnitude,                                  # param 1: angle (deg)
            angular_speed_deg_s,                        # param 2: angular speed
            direction,                                  # param 3: 1=CW, -1=CCW
            1,                                          # param 4: 1=relative
            0, 0, 0,                                    # params 5-7 unused
        )
        self.vehicle.send_mavlink(msg)

        # Block until heading is within tolerance, or timeout
        start = time.time()
        while time.time() - start < timeout:
            current = math.degrees(self.vehicle.attitude.yaw) % 360
            # Shortest angular distance, handles 0/360 wrap-around
            error = abs((current - absolute_target + 180) % 360 - 180)
            if error <= tolerance_deg:
                return True
            time.sleep(0.1)

        return False
        

    def send_landing_target(self, x: float, y: float):
        if not self.vehicle.location.global_relative_frame.alt:
            logging.warning("No altitude available, cannot send landing target")
            return
        z = self.vehicle.location.global_relative_frame.alt
        distance = math.sqrt(x**2 + y**2 + z**2)

        self.vehicle.message_factory.landing_target_send(
            0,  # time target data was processed, as close to sensor capture as possible
            self.vehicle._master.target_system,  # target num, not used
            mavutil.mavlink.MAV_FRAME_BODY_FRD,  # ty: ignore[unresolved-attribute]  # frame, not used
            0,  # X-axis angular offset, in radians
            0,  # Y-axis angular offset, in radians
            distance,  # distance, in meters
            0,  # Target x-axis size, in radians
            0,  # Target y-axis size, in radians
            y,  # forward
            x,  # right
            z,  # down
            (
                1,
                0,
                0,
                0,
            ),  # q	float[4]	Quaternion of landing target orientation (w, x, y, z order, zero-rotation is 1, 0, 0, 0)
            3,  # visual marker
            1,  # position_valid
        )

    def to_quaternion(self, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Convert degrees to quaternions
        """
        t0 = math.cos(math.radians(yaw * 0.5))
        t1 = math.sin(math.radians(yaw * 0.5))
        t2 = math.cos(math.radians(roll * 0.5))
        t3 = math.sin(math.radians(roll * 0.5))
        t4 = math.cos(math.radians(pitch * 0.5))
        t5 = math.sin(math.radians(pitch * 0.5))

        w = t0 * t2 * t4 + t1 * t3 * t5
        x = t0 * t3 * t4 - t1 * t2 * t5
        y = t0 * t2 * t5 + t1 * t3 * t4
        z = t1 * t2 * t4 - t0 * t3 * t5

        return [w, x, y, z]
