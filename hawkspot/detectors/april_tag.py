try:
    from pupil_apriltags   import Detector
except ImportError:
    raise ImportError(
        "pupil_apriltags not installed. It is required for AprilTagDetector. Please install it with `pip install dt-apriltags`"
    )

import logging
from math import atan2, pi
from typing import Optional

import cv2
from cv2.typing import MatLike

from hawkspot import BaseDetector, BoundingBox, DetectionOffset


class AprilTagDetector(BaseDetector):
    def __init__(
        self,
        detector: Detector = Detector(
            
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        ),
        tag: int | None = None,
    ):

        self.detector = detector
        self.tag = tag

    def detect(self, image: MatLike) -> Optional[DetectionOffset]:
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = [
            result
            for result in self.detector.detect(gray_frame)
            if self.tag is None or result.tag_id == self.tag
        ]
        # print(results)

        if len(results) == 0:
            return None
        elif len(results) > 1:
            logging.warning(
                f"{len(results)} AprilTag results found. Returning the first one."
            )

        x1 = round(min(results[0].corners, key=lambda p: p[0])[0])
        y1 = round(min(results[0].corners, key=lambda p: p[1])[1])
        x2 = round(max(results[0].corners, key=lambda p: p[0])[0])
        y2 = round(max(results[0].corners, key=lambda p: p[1])[1])
        bbox = BoundingBox(x1, y1, x2 - x1, y2 - y1)

        dx = results[0].center[0] - image.shape[1] / 2
        dy = results[0].center[1] - image.shape[0] / 2

        top_left = results[0].corners[3]
        top_right = results[0].corners[2]
        vector = (top_right[0] - top_left[0], top_right[1] - top_left[1])
        yaw = atan2(vector[1], vector[0]) * 180 / pi

        return DetectionOffset(bbox, dx, dy, yaw)
