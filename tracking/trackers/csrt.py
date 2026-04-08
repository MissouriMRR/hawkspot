"""
Includes the CSRTTracker class, which uses the CSRT tracker from OpenCV to track objects.
"""

import cv2
from cv2.typing import MatLike

from tracking import BaseTracker, BoundingBox


class CSRTTracker(BaseTracker):
    """
    Uses the CSRT tracker from OpenCV to track objects.
    """

    def __init__(self):
        super().__init__()
        self.tracker = cv2.TrackerCSRT.create()

    def start_tracking(self, image: MatLike, roi: BoundingBox) -> None:
        self.roi = roi.to_rect()
        self.tracker.init(image, roi.to_rect())

    def track(self, image: MatLike) -> BoundingBox:
        success, roi = self.tracker.update(image)
        if success:
            return BoundingBox(*roi)
        raise ValueError("Tracking failed")
