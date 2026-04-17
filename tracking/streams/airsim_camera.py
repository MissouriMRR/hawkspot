import airsim
import numpy as np
from cv2.typing import MatLike

from tracking import BaseStream


class AirsimCameraStream(BaseStream):
    """
    Connects to the camera stream inside of a Airsim instance.
    """

    def __init__(self, showStream: bool = False) -> None:
        super().__init__(showStream)
        self._client: airsim.MultirotorClient = airsim.MultirotorClient()

    def get_frame(self):
        img_data = self._client.simGetImage("bottom_center", airsim.ImageType.Scene)
        self.frame = np.fromstring(img_data, dtype=np.uint8)
        super().show_frame()
        return self.frame

    def end(self):
        pass
