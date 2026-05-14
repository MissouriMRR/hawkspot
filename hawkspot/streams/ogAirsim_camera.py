try:
    import airsim
except ImportError as exc:
    raise ImportError(
        "Airsim library is not installed. Please install it to use the ogAirsim_camera stream."
    ) from exc

from hawkspot import BaseStream


from hawkspot import BaseStream


class AirsimCamera(BaseStream):
    """
    Sets up a connected Airsim camera to be used as a stream.
    """

    def __init__(self, showStream: bool = False, airsim_ip: str = "127.0.0.1", camera_name: str = "bottom_center"):

        super().__init__(showStream)
        self._client = airsim.VehicleClient()
        self._client.confirmConnection()

    def get_frame(self):
        self.frame = self._client.simGetImage("0", airsim.ImageType.Scene)
        #self.frame = self._camera.capture_array()
        super().show_frame()
        return self.frame

    def end(self):
        self._camera.stop_recording()
