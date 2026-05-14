import logging
from dataclasses import dataclass
from typing import Optional, cast

import dronekit
import numpy as np

from hawkspot import (
    BaseDetector,
    BaseStream,
    BaseTracker,
    DetectionOffset,
    Drone,
    LocalOffset,
)


@dataclass
class HawkspotParams:
    # The forward offset in meters from the camera to the center of the drone
    forward_offset: float = 0.0

    # The right offset in meters from the camera to the center of the drone
    right_offset: float = 0.0

    # The yaw offset from the orientation of the detection to the orientation of the drone in radians. Positive values indicate the drone is rotated clockwise from the target, and negative values indicate the drone is rotated counterclockwise from the target.
    yaw_offset: float = 0.0

    # Camera focal length in millimeters
    camera_focal_length: float = 35.0

    # Camera sensor size in millimeters (width, height)
    camera_sensor_size: tuple[float, float] = (36.0, 24.0)

    # Maximum distance in meters the drone can be from the target
    # and still be considered on the target
    offset_threshold: float = 0.05

    # Maximum yaw offset in radians the drone can be off from the target to be considered acceptable
    yaw_threshold: float = 0.1

    # Number of consecutive frames the drone must be on the target
    # before it is considered to have reached the target
    target_threshold: int = 10

    # Maximum continous misdirection in meters before the drone considers
    # the target lost and must search for it again
    misdirection_threshold: float = 0.5

    # Altitude in meters to start going straight down
    land_decision_altitude: float = 1.0

    # Show live view of the drone's camera feed and detections
    show_stream: bool = False


class Hawkspot:
    def __init__(
        self,
        drone: Drone,
        stream: BaseStream,
        detector: BaseDetector,
        tracker: Optional[BaseTracker] = None,
        params: HawkspotParams = HawkspotParams(),
    ):
        self.drone = drone
        self.stream = stream
        self.detector = detector
        self.tracker = tracker
        self.params = params

        self._tracker_init = False

    def _calculate_offset(
        self, offset: DetectionOffset, frame_size: tuple[int, int]
    ) -> LocalOffset:
        # Calculate the offset from the center of the bounding box provided to the drone's current local position.
        # Returns the offset in meters, forward and right offsets in meters.
        altitude = self.drone.vehicle.location.global_relative_frame.alt
        drone_pos = np.array([0, 0, altitude])
        if altitude is None or altitude < 0:
            return LocalOffset(0, 0)

        fx = (
            self.params.camera_focal_length
            * frame_size[0]
            / self.params.camera_sensor_size[0]
        )
        fy = (
            self.params.camera_focal_length
            * frame_size[1]
            / self.params.camera_sensor_size[1]
        )
        r = np.array([offset.x / fx, offset.y / fy, 1.0])
        r = r / np.linalg.norm(r)

        r_cam_to_NED = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
            dtype=float,
        )

        r_NED = r_cam_to_NED @ r
        t = drone_pos[2] / -r_NED[2]

        return LocalOffset(r_NED[0] * t, r_NED[1] * t, offset.yaw)

    def _get_next_offset(self) -> LocalOffset:
        for frame in self.stream:
            if self.tracker is None or not self._tracker_init:
                detection = self.detector.detect(frame)
                if detection:
                    if self.tracker:
                        self.tracker.start_tracking(frame, detection.bbox)
                        self._tracker_init = True
                    return self._calculate_offset(detection, frame.shape)

            else:
                result = self.tracker.track(frame)
                if result is not None:
                    offset = self._calculate_offset(result, frame.shape)
                    return offset
        raise ValueError("Failed to find next offset")

    def _track_object(self, landing: bool) -> bool:
        misdirection: float = 0
        target_samples: int = 0
        last_delta: float | None = None

        while (
            cast(float, self.drone.vehicle.location.global_relative_frame.alt)
            > self.params.land_decision_altitude
            or not landing
        ):
            try:
                offset = self._get_next_offset()
                offset.yaw += self.params.yaw_offset
                logging.info(
                    f"offset | forward={offset.forward}, right={offset.right}, delta={offset.distance}, yaw={offset.yaw}"
                )

                # Check delta for misdirection
                if last_delta is not None and (offset.distance - last_delta) > 1:
                    misdirection += abs(offset.distance - last_delta)
                    if misdirection >= self.params.misdirection_threshold:
                        logging.info(f"Misdirection: {misdirection}")
                        # We want to reset the tracker if there is one
                        self._tracker_init = False

                else:
                    # We are heading in the right direction, reset misdirection
                    current_yaw: float = cast(float, self.drone.vehicle.attitude.yaw)
                    acceptable_yaw = (
                        abs(offset.yaw - current_yaw) < self.params.yaw_threshold
                        if offset.yaw is not None
                        else True
                    )
                    acceptable_offset = offset.distance < self.params.offset_threshold
                    if acceptable_yaw and acceptable_offset and not landing:
                        target_samples += 1
                        if target_samples >= self.params.target_threshold:
                            return True
                    else:
                        target_samples = 0
                    misdirection = 0
                last_delta = offset.distance

                
                # Start moving to point
                if not landing:
                    point = dronekit.LocationLocal(offset.forward, offset.right, 0)
                    for i in range(10):
                        self.drone.goto_local(point)
                

                # Send LANDING_TARGET message
                self.drone.send_landing_target(offset.right, offset.forward)

                # Adjust yaw if necessary
                if offset.yaw is not None:
                    logging.debug(
                        f"yaw | offset={offset.yaw}, current={self.drone.vehicle.attitude.yaw}"
                    )
                    self.drone.set_yaw(
                        cast(float, self.drone.vehicle.attitude.yaw) + offset.yaw 
                    )
                if(not self.drone.vehicle.armed):
                    logging.info("Drone disarmed, we assume it landed.")
                    return True

            except ValueError:
                return False

        # This will only be reached if the altitude is below the land decision altitude
        return True

    def find_object(self) -> dronekit.LocationGlobal:
        result = False
        while not result:
            result = self._track_object(False)

        logging.info("Object found @ %s", self.drone.vehicle.location.global_frame)
        return self.drone.vehicle.location.global_frame

    def land_on_object(self) -> dronekit.LocationGlobal:
        result = False
        while not result:
            logging.info("_______________here______________________")
            result = self._track_object(False)
            
        logging.info(f"Object found @ {self.drone.vehicle.location.global_frame}, landing...")
        self.drone.vehicle.mode = dronekit.VehicleMode("LAND")
        self.drone.vehicle.wait_for_mode("LAND")
        self._track_object(True)

        while self.drone.vehicle.armed:
            pass

        logging.info("Landed on object @ %s", self.drone.vehicle.location.global_frame)
        return self.drone.vehicle.location.global_frame
