import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def radar_to_vehicle(radar_pc, radar_extrinsics):
    '''
    Function that takes radar point cloud in radar coordinates and transform to vehicle/global coordinates.

    Params:
    - radar_pc: N x 3 numpy array
    - radar_extrinsics: 3 x 4 numpy array

    Output:
    - radar vehicle coordinates: N x 3 numpy array
    '''
    # radar 3 dim -> 4 dim (with homogeneous coordinate) : N x 4
    homogenous_radar_pc = np.append(radar_pc, [[1]]*len(radar_pc), 1)

    # transform to vehicle coordinates 3 x 4 @ 4 x dims = 3 x dims
    vehicle_coordinates = radar_extrinsics @ homogenous_radar_pc.T

    # Need to reverse the x, y axis for rader.
    # radar_vehicle = np.vstack((-vehicle_coordinates[1, :], vehicle_coordinates[0, :], vehicle_coordinates[2, :]))
    radar_vehicle = vehicle_coordinates
    return radar_vehicle.T


def vehicle_to_camera(vehicle_pc, camera_extrinsics):
    '''
    Function that takes point cloud in vehicle/global coordinates and transform to camera coordinates.

    Params:
    - vehicle_pc: N x 3
    - camera_extrinsics: 3 x 4

    Output:
    - camera coordinates: N x 4
    '''
    # decompose into rotational and translational
    camera_extrinsics_hom = np.vstack((camera_extrinsics,[[0,0,0,1]]))
    
    camera_extrinsics_hom_inv = np.linalg.inv(camera_extrinsics_hom)

    homogenous_vehicle_pc = np.append(vehicle_pc, [[1]]*len(vehicle_pc), 1)

    # 4x4 @ 4xN = 4xN
    camera_coordinates = camera_extrinsics_hom_inv @ homogenous_vehicle_pc.T
    
    # N x 4
    return camera_coordinates.T

def camera_to_image(camera_pt, camera_intrinsics):
    '''
    Function that takes points in camera coordinates and transform to image coordinates. 3D -> 2D

    Params:
    - camera_pt: N x 4 (with homogenous coordinate)
    - camera_intrinsics: tuple as (intrinsics matrix: 3 x 3, distortion array: 1 x 5)

    Output:
    - image coordinates: N x 2
    '''
    # decompose intrinsics
    in_matrix, distortion = camera_intrinsics
    
    in_matrix_hom = np.hstack((in_matrix, [[1]]*3))

    # 3x4 @ 4xN = 3xN
    image_coordinates = in_matrix_hom@camera_pt.T
    
    
    # Divide the x, y components by z to obtain the normalized image coordinates
    image_coordinates = image_coordinates / image_coordinates[2, :]
    
    # Transpose the resulting array to get shape (2, N)
    image_coordinates = image_coordinates[:2, :].T
    
    return image_coordinates

def filter_radar(points, labels):

    filtered_indices = np.where((0 < points[:, 0]) & (points[:, 0] < 1616) & (0 < points[:, 1]) & (points[:, 1] < 1240))
    filtered_points = points[filtered_indices]
    filtered_labels = np.array(labels)[filtered_indices]

    return filtered_indices, filtered_points, filtered_labels

def filter_radar_baseline(points):

    filtered_indices = np.where((0 < points[:, 0]) & (points[:, 0] < 1616) & (0 < points[:, 1]) & (points[:, 1] < 1240))
    filtered_points = points[filtered_indices]
    #filtered_labels = np.array(labels)[filtered_indices]

    return filtered_indices, filtered_points