import os
import threading
import typing

from cv2 import VideoCapture  # pylint: disable=no-name-in-module
from cv2.typing import MatLike

from tracking import BaseStream


class CameraStream(BaseStream):
    """
    Sets up a stream using an onboard camera, stream file, or IP stream with OpenCV's VideoCapture.
    """

    @typing.overload
    def __init__(self, source: int, showStream: bool = False) -> None: ...

    @typing.overload
    def __init__(
        self, source: str | os.PathLike[str], showStream: bool = False
    ) -> None: ...

    def __init__(
        self, source: int | str | os.PathLike[str], showStream: bool = False
    ) -> None:
        super().__init__(showStream)
        self.capturing = True
        self._capture = VideoCapture(source)
        self._buffer: typing.Optional[MatLike] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._reader)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise TimeoutError("Timed out waiting for frame")

    def _reader(self):
        while self.capturing:
            ret, frame = self._capture.read()
            if not ret:
                if not self.capturing:
                    break
                raise RuntimeError("Failed to capture frame")
            with self._lock:
                self._buffer = frame
                self._ready.set()

    def get_frame(self):
        """
        Retrieves the latest frame from the feed.
        """
        with self._lock:
            if self._buffer is None:
                raise RuntimeError("No frame available")
            self.frame = self._buffer
        super().show_frame()
        return self.frame

    def end(self):
        self.capturing = False
        self._capture.release()
