from src.utils import *


def multi_radar_to_camera(radar_pts, radar_extrinsics, camera_extrinsics, camera_intrinsics):
    all_camera = {}
    for id, radar_pt in radar_pts.items():
        camera_coor = single_radar_to_camera(radar_pt, radar_extrinsics[id], camera_extrinsics)
        all_camera[id] = camera_coor
    return all_camera


def single_radar_to_camera(radar_pt, radar_extrinsics, camera_extrinsics):
    radar_vehicle = radar_to_vehicle(radar_pt, radar_extrinsics)
    camera_coor = vehicle_to_camera(radar_vehicle, camera_extrinsics)
    return camera_coor

