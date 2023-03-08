import numpy as np


def load_data(data_load_path):
    '''
    Params:
    - data path

    Output:
    - radar_pt: a dictionary that has keys as radar sensor identifiers, values as point clouds: N x 3.
    - camera: a dictionary that has keys as camera identifiers, values as RGB images: N x 3.
    - lidar_pt: dense lidar point clouds: N x 3.
    '''

    print(f'Loading data for: {data_load_path}')
    data = np.load(data_load_path, allow_pickle=True)

    print(f'Loading camera frames.')
    camera = data['camera_frames'].item()

    # camera_left = camera['21248038']
    # camera_right = camera['21248039']
    # camera_mid = camera['20438665']

    lidar_pt = data['lidar_points']

    radar_pt = data['radar_points'].item()


    return radar_pt, camera, lidar_pt

def load_props(props_path):
    '''
    Params:
    - props path

    Output:
    - camera_intrinsics: a dictionary that has keys as camera sensor identifiers,
        values as tuple: (intrinsics matrix: 3 x 3, distortion array: 1 x 5)
    - camera_extrinsics: a dictionary that has keys as camera sensor identifiers,
        values as 3 x 4 extrinsics matrices.
    - lidar_extrinsics: a 3 x 4 lidar extrinsics matrix.
    - radar_extrinsics: a dictionary that has keys as radar sensor identifiers,
        values as tuple 3 x 4 extrinsics matrices.
    '''

    matrices = np.load(props_path, allow_pickle=True)
    radar_extrinsic = matrices['radar_extrinsics'].item()

    lidar_extrinsic = matrices['lidar_extrinsic']

    camera_extrinsics = matrices['camera_extrinsics'].item()
    camera_intrinsics = matrices['camera_intrinsics'].item()

    return camera_intrinsics, camera_extrinsics, lidar_extrinsic, radar_extrinsic