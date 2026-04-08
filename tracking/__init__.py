from dataclasses import dataclass
from functools import total_ordering
from typing import Sequence

import cv2


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


class BaseDetector:
    """
    Responsible for getting the initial region of interest in an image to track.
    In most cases, this will be responsible for detecting an object in an image that you
    want to track.
    """

    def detect(self, image: cv2.typing.MatLike) -> Sequence[BoundingBox]:
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

    def track(self, image: cv2.typing.MatLike) -> BoundingBox:
        raise NotImplementedError("Subclasses must implement track method")


class BaseFollower:
    def __init__(self, tracker: BaseTracker):
        self.tracker = tracker

    def follow(self, image):
        new_roi = self.tracker.track(image)
        return new_roi


class ObjectTracker:
    def __init__(
        self,
        stream: BaseStream,
        detector: BaseDetector,
        tracker: BaseTracker,
        follower: BaseFollower,
    ):
        self.stream = stream
        self.detector = detector
        self.tracker = tracker
        self.follower = follower

    def track_motion(self, image):
        detections = self.detector.detect(image)
        motion = self.tracker.track(image)
        return detections, motion


COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}
