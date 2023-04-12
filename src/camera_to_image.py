from src.utils import *


def multi_camera_to_img(all_camera, camera_intrinsics):
    all_radar = {}
    for id, radar_pt in all_camera.items():
        image_coor = single_camera_to_img(radar_pt, camera_intrinsics)
        all_radar[id] = image_coor

    return all_radar


def single_camera_to_img(camera_coor, camera_intrinsics):
    image_coor = camera_to_image(camera_coor, camera_intrinsics)
    return image_coor
