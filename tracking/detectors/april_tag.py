try:
    from dt_apriltags import Detector
except ImportError:
    raise ImportError(
        "dt_apriltags not installed. It is required for AprilTagDetector. Please install it with `pip install dt-apriltags`"
    )

from typing import Sequence

import cv2
from cv2.typing import MatLike

from tracking import BaseDetector, BoundingBox


class AprilTagDetector(BaseDetector):
    def __init__(
        self,
        detector: Detector = Detector(
            searchpath=["apriltags"],
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

    def detect(self, image: MatLike) -> Sequence[BoundingBox]:
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bboxes: list[BoundingBox] = []
        results = self.detector.detect(gray_frame)
        for result in results:
            print(result)
            if self.tag and result.tag_id != self.tag:
                continue
            x1 = round(min(result.corners, key=lambda p: p[0])[0])
            y1 = round(min(result.corners, key=lambda p: p[1])[1])
            x2 = round(max(result.corners, key=lambda p: p[0])[0])
            y2 = round(max(result.corners, key=lambda p: p[1])[1])
            bbox = BoundingBox(x1, y1, x2 - x1, y2 - y1)
            bboxes.append(bbox)
        return bboxes
