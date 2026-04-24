from typing import Optional

from cv2.typing import MatLike
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import ImageType
from projectairsim.utils import unpack_image

from tracking import BaseStream


class AirsimCameraStream(BaseStream):
    """
    Connects to the camera stream inside of a Airsim instance.
    """

    def __init__(
        self,
        airsim_client: ProjectAirSimClient,
        airsim_world: World,
        airsim_drone: Drone,
        camera_name: str,
        showStream: bool = False,
    ) -> None:
        super().__init__(showStream)
        self._client: ProjectAirSimClient = airsim_client
        if not self._client.state:
            self._client.connect()

        self._world: World = airsim_world
        self._drone: Drone = airsim_drone
        self._frame: Optional[MatLike] = None
        self.camera_name: str = camera_name

        self._client.subscribe(
            self._drone.sensors["Camera"][camera_name], self._rgb_frame_callback
        )

    def _rgb_frame_callback(self, _, image_msg):
        self._frame = image_msg

    def get_frame(self):
        if self._frame is None:
            raise RuntimeError("No frame available")
        self.frame = unpack_image(self._frame)
        super().show_frame()
        return self.frame

    def update_focal_length(self, focal_length: float):
        self._drone.set_focal_length(self.camera_name, ImageType.SCENE, focal_length)

    def end(self):
        pass
