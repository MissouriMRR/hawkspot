from dataclasses import dataclass
from functools import total_ordering
from math import hypot
from typing import Optional

import cv2


@dataclass
@total_ordering
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __lt__(self, other) -> bool:
        raise NotImplementedError(
            "Subclasses must implement __lt__ method if ordering is needed"
        )

    def to_rect(self) -> cv2.typing.Rect:
        return [self.x, self.y, self.width, self.height]

    def to_pts(self) -> tuple[cv2.typing.Point, cv2.typing.Point]:
        return (self.x, self.y), (self.x + self.width, self.y + self.height)

    def middle(self) -> cv2.typing.Point:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def delta(self, point: cv2.typing.Point) -> cv2.typing.Point:
        middle = self.middle()
        return (-(point[0] - middle[0]), point[1] - middle[1])

    def offset(self, frame: cv2.typing.MatLike) -> cv2.typing.Point:
        return self.delta((frame.shape[1] // 2, frame.shape[0] // 2))


@dataclass
class DetectionOffset:
    """
    Represents the offset of a detection from the center of the image.
    Optionally can pass yaw if the detection method supports it.
    """

    bbox: BoundingBox
    x: float
    y: float
    yaw: Optional[float] = None


@dataclass
class LocalOffset:
    """
    Represents the offset of a detection from the drone's local position in meters.
    """

    forward: float
    right: float
    yaw: Optional[float] = None

    @property
    def distance(self) -> float:
        return hypot(self.forward, self.right)


class BaseStream:
    """
    A camera or video stream that provides frames for tracking.
    """

    frame: cv2.typing.MatLike

    def __init__(self, showStream=False):
        self.showStream = showStream

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.get_frame()
        except RuntimeError:
            raise StopIteration

    def get_frame(self) -> cv2.typing.MatLike:
        raise NotImplementedError("Subclasses must implement get_frame method")

    def show_frame(self):
        if self.showStream:
            cv2.imshow("Stream", self.frame)
            cv2.waitKey(1)

    def end(self):
        raise NotImplementedError("Subclasses must implement end method")


class BaseDetector:
    """
    Responsible for getting the initial region of interest in an image to track.
    In most cases, this will be responsible for detecting an object in an image that you
    want to track.
    """

    def detect(self, image: cv2.typing.MatLike) -> Optional[DetectionOffset]:
        raise NotImplementedError("Subclasses must implement detect method")


class BaseTracker:
    """
    This class is responsible for finding the new location of an object once it has been detected.
    It should output the new bounding box of the object.
    """

    def __init__(self, draw_result: bool = False):
        self.roi: cv2.typing.Rect | None = None

    def start_tracking(self, image: cv2.typing.MatLike, roi: BoundingBox) -> None:
        raise NotImplementedError("Subclasses must implement start_tracking method")

    def track(self, image: cv2.typing.MatLike) -> Optional[DetectionOffset]:
        raise NotImplementedError("Subclasses must implement track method")
