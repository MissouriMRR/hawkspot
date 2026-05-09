"""
Tests that the AprilTagDetector can detect tags in a camera stream.
"""

import cv2

from hawkspot.detectors.april_tag import AprilTagDetector
from hawkspot.streams.camera import CameraStream
from hawkspot.utils import draw_offset_line

detector = AprilTagDetector(tag=None)
stream = CameraStream(0, showStream=False)

for frame in stream:
    detection = detector.detect(frame)
    if detection:
        print(detection)
        draw_offset_line(frame, detection)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
