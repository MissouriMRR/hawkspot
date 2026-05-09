"""
Provides the SIYICamera class for streaming video from a SIYI camera onboard.
"""

try:
    from siyi_sdk.siyi_sdk import SIYISDK
except ImportError as exc:
    raise ImportError(
        "siyi_sdk is required for SIYICamera. Install it with: pip install siyi_sdk"
    ) from exc

import threading
from dataclasses import dataclass
from typing import Optional

from cv2 import VideoCapture  # pylint: disable=no-name-in-module
from cv2.typing import MatLike

from hawkspot import BaseStream


class SIYICamera(BaseStream):
    """
    Sets up a stream from a SIYI camera.
    The default IP is the IP of the camera over a Herelink hotspot.
    """

    @dataclass
    class Attitude:
        """
        Dataclass that represents the camera's attitude (roll, pitch, yaw).
        """

        roll: float
        pitch: float
        yaw: float

    def __init__(self, addr: str = "192.168.144.25:8554", showStream: bool = False):
        super().__init__(showStream)
        self._camera: SIYISDK = SIYISDK()
        if not self._camera.connect(maxRetries=10):
            raise TimeoutError("Failed to connect to SIYI camera after 10 attempts.")
        self._capture = VideoCapture(f"rtsp://{addr}/main.264")
        self._buffer: Optional[MatLike] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._reader)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise TimeoutError("Timed out waiting for frame")

    def _reader(self):
        while True:
            ret, frame = self._capture.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            with self._lock:
                self._buffer = frame
                self._ready.set()

    def get_frame(self):
        with self._lock:
            if self._buffer is None:
                raise RuntimeError("No frame available")
            self.frame = self._buffer
        super().show_frame()
        return self.frame

    def end(self):
        self._camera.disconnect()

    def get_camera_attitude(self):
        """
        Returns the current camera attitude (roll, pitch and yaw).

        Returns:
            Attitude: The current camera attitude.
        """
        gimbal_attitude = self._camera.getAttitude()
        roll_deg: float = gimbal_attitude[2]
        pitch_deg: float = gimbal_attitude[1]
        yaw_deg: float = gimbal_attitude[0]

        return SIYICamera.Attitude(roll_deg, pitch_deg, yaw_deg)

    def set_camera_attitude(self, attitude: Attitude):
        """
        Sets the desired camera attitude (yaw and pitch).
        Given roll is ignored.
        """
        self._camera.setGimbalRotation(attitude.yaw, attitude.pitch)
