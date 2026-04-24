from projectairsim import Drone, ProjectAirSimClient, World

from tracking.streams.airsim_camera import AirsimCameraStream


def test_airsim_camera_stream(
    address: str = "127.0.0.1",
    scene_config_name: str = "",
    sim_config_path: str = "sim_config/",
):
    client = ProjectAirSimClient(address=address)
    client.connect()

    world = World(client, scene_config_name, sim_config_path=sim_config_path)

    drone = Drone(client, world, "Drone1")

    stream = AirsimCameraStream(client, world, drone, "DownCamera", showStream=True)

    for frame in stream:
        pass


if __name__ == "__main__":
    test_airsim_camera_stream()
