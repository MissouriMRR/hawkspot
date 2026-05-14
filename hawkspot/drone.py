import logging
import math

import dronekit
from pymavlink import mavutil


class Drone:
    def __init__(self, vehicle: dronekit.Vehicle):
        self.vehicle = vehicle

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
            self.yaw,
            0,  # yaw, yaw_rate (ignored)
        )

    def set_yaw(self, yaw: float):
        self.yaw=yaw
        """
        self.vehicle.message_factory.set_attitude_target_send(
            0,  # time_boot_ms
            self.vehicle._master.target_system,  # Target system
            self.vehicle._master.target_component,  # Target component
            0b00000111,
            self.to_quaternion(yaw=yaw),  # Quaternion
            0,  # Body roll rate in radian
            0,  # Body pitch rate in radian
            10,  # Body yaw rate in radian/second
            0.5,  # Thrust
        )
        """

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
