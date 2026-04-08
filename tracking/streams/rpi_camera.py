"""
Provides the RPICamera class, which sets up a connected Raspberry Pi camera to be used as a stream.
"""

try:
    import picamera2
except ImportError as exc:
    raise ImportError(
        "picamera2 is required for RPICamera. It should be preinstalled on your pi, "
        "but if not, install it with: pip install picamera2"
    ) from exc

from tracking import BaseStream


class RPICamera(BaseStream):
    """
    Sets up a connected Raspberry Pi camera to be used as a stream.
    """

    def __init__(self, showStream: bool = False):

        super().__init__(showStream)
        self._camera = picamera2.Picamera2()
        self._camera.start()

    def get_frame(self):
        self.frame = self._camera.capture_array()
        super().show_frame()
        return self.frame

    def end(self):
        self._camera.stop_recording()
