"""
Tests that the AprilTagDetector can detect tags in a camera stream.
"""

import cv2

from tracking.detectors.april_tag import AprilTagDetector
from tracking.streams.camera import CameraStream
from tracking.utils import draw_bounding_box

detector = AprilTagDetector(tag=None)
stream = CameraStream(0, showStream=False)

for frame in stream:
    detections = detector.detect(frame)
    for detection in detections:
        print(detection)
        draw_bounding_box(frame, detection)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
