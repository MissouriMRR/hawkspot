import platform

from hawkspot.base import (
    BaseDetector,
    BaseStream,
    BaseTracker,
    BoundingBox,
    DetectionOffset,
    LocalOffset,
)
from hawkspot.detectors.april_tag import AprilTagDetector
from hawkspot.detectors.rt_detr import RTDETRV2_Detector 
from hawkspot.streams.ogAirsim_camera import AirsimCamera
current_os = platform.system()
if current_os == "Linux":
    from hawkspot.streams.rpi_camera import RPICamera

from hawkspot.streams.siyi_camera import SiyiCamera
from hawkspot.detectors.april_tag import AprilTagDetector
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
