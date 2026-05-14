import platform

from hawkspot.base import (
    BaseDetector,
    BaseStream,
    BaseTracker,
    BoundingBox,
    DetectionOffset,
    LocalOffset,
)

from hawkspot.drone import Drone
from hawkspot.hawkspot import Hawkspot

__all__ = [
    "Hawkspot",
    "BaseDetector",
    "BaseStream",
    "BaseTracker",
    "BoundingBox",
    "DetectionOffset",
    "Drone",
    "LocalOffset",
    "AprilTagDetector",
    "RTDETRV2_Detector",
    "AirsimCamera",
    "RPICamera",
    "SiyiCamera"
]
