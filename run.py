from src.data.dataset import Scene, CameraRadarProps
from src.labeling_via_dbscan import get_label_from_image_via_DBclustering, perform_DBScan
from src.utils import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import argparse

LEFT_CAM = '21248038'
MID_CAM = '20438665'
RIGHT_CAM = '21248039'

LEFT_RADAR = 'ZRVE1002'
MID_RADAR = 'ZRVC2001'
RIGHT_RADAR = 'ZRVE1001'

filepath = 'data/00000.npz'
sensor_prop_path = 'data/extrinsics_intrinsics.npz'
label_color = {'car': 'yellow', 'truck':'red', 'person': 'green', 'bus':'orange'}
color_label = {'yellow':'car', 'red':'truck', 'green':'person', 'orange':'bus', 'black':'others'}
save_pics = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script with command-line arguments")
    parser.add_argument("--data", type=str, help="location of data file")
    args = parser.parse_args()

    if args.data:
        filepath = args.data
    data = Scene(filepath)
    sensor_props = CameraRadarProps(sensor_prop_path)

    radar_data = data.get_radar_data()
    camera_data = data.get_camera_data()
    cam = RIGHT_CAM
    view_num = filepath.split('.')[0][7:]

    # camera_data[cam] -- middle image  --> bounding boxes
    # plot boxes [(x1,y1,x2,y2)] x,y  x1 <  x < x2 , y1 < y < y2 

    # camera
    img = camera_data[cam]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    all3_radar_data = list(radar_data.values())
    # (x,y,z)
    radar_data_global_coord = np.vstack([radar_to_vehicle(single_radar, sensor_props.radar_extrinsic[key]) 
                                         for key, single_radar in radar_data.items()])
    all_3D_to_label = {tuple(k):None for k in radar_data_global_coord}
    
    clusters = perform_DBScan(radar_data_global_coord)


    get_label_from_image_via_DBclustering(radar_data_global_coord, camera_data, LEFT_CAM,clusters, sensor_props, all_3D_to_label, view_num, save_pics)
    get_label_from_image_via_DBclustering(radar_data_global_coord, camera_data, MID_CAM,clusters, sensor_props, all_3D_to_label, view_num, save_pics)
    get_label_from_image_via_DBclustering(radar_data_global_coord, camera_data, RIGHT_CAM,clusters, sensor_props, all_3D_to_label, view_num, save_pics)


    # graphing the results: top-down view
    # Group points by label
    points_by_label = {}
    for pt, label in all_3D_to_label.items():
        if label in points_by_label:
            points_by_label[label].append(pt)
        else:
            points_by_label[label] = [pt]

    # Plot points for each label in a single call
    fig, ax = plt.subplots()
    for label, points in points_by_label.items():
        color = label_color.get(label, 'black')
        points_np = np.array(points)
        ax.scatter(-points_np[:, 1], points_np[:, 0], color=color, s=3, label=color_label[color])
    legend = ax.legend(loc='upper right', title='Dot Colors')

    car_logo = mpimg.imread('assets/car.png')
    imagebox = AnnotationBbox(OffsetImage(car_logo, zoom=0.07), (0, 0), boxcoords="offset points", frameon=False)
    ax.add_artist(imagebox)
    ax.set_xlim(-100, 100)
    ax.set_axis_off()

    if not save_pics:
        plt.show()