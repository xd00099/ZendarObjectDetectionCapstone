from src.utils import *

def single_radar_to_img(radar_pt, radar_extrinsics, camera_extrinsics, camera_intrinsics):
    '''
    Function that takes in radar data and project it onto image coordinates space.
    It's a 3D -> 3D -> 2D projection.
    '''
    radar_vehicle = radar_to_vehicle(radar_pt, radar_extrinsics)
    camera_coor = vehicle_to_camera(radar_vehicle, camera_extrinsics)
    image_coor = camera_to_image(camera_coor, camera_intrinsics)
    # x, y = image_coor[:,0], image_coor[:,1]

    return image_coor


def keep_inbound(image_coor, x_lim, y_lim):
    return np.array([[i,j] for i,j in image_coor if 0<i<x_lim and 0<j<y_lim])

def multi_radar_to_img(radar_pts, radar_extrinsics, camera_extrinsics, camera_intrinsics, x_lim=1616, y_lim=1240, image=None):

    all_radar = {}
    for id, radar_pt in radar_pts.items():
        image_coor = single_radar_to_img(radar_pt, radar_extrinsics[id], camera_extrinsics, camera_intrinsics)
        filtered_coor = keep_inbound(image_coor, x_lim, y_lim)
        all_radar[id] = filtered_coor
    
        if image is not None:
            plt.scatter(filtered_coor[:, 0], filtered_coor[:,1], label=id, s=1.5)
            
    if image is not None:
        plt.legend()

    return all_radar

