from src.data.dataset import Scene, CameraRadarProps
from src.radar_to_image import single_radar_to_img, multi_radar_to_img
from src.radar_to_camera import single_radar_to_camera, multi_radar_to_camera
from src.camera_to_image import single_camera_to_img, multi_camera_to_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from src.model.yolo import run_yolo
from DBscan.run_dbscan import run_cluster, run_cluster_2, run_cluster_3
from src.utils import filter_radar
from src.model.merge import merge_labels
from src.model.yolo import draw_prediction
import numpy as np


LEFT_CAM = '21248038'
MID_CAM = '20438665'
RIGHT_CAM = '21248039'

LEFT_RADAR = 'ZRVE1002'
MID_RADAR = 'ZRVC2001'
RIGHT_RADAR = 'ZRVE1001'

filepath = 'data/00000.npz'
sensor_prop_path = 'data/extrinsics_intrinsics.npz'

yolo_cfg_path = 'yolo_setup/yolov3.cfg'
yolo_class_path = 'yolo_setup/yolov3.txt'
yolo_weights_path = 'yolo_setup/yolov3.weights'
labeled_image = 'data/00000.npz_object-detection.jpg'

if __name__ == '__main__':

    data = Scene(filepath)
    sensor_props = CameraRadarProps(sensor_prop_path)

    radar_data = data.get_radar_data()
    camera_data = data.get_camera_data()
    cam = RIGHT_CAM
    rad = RIGHT_RADAR

    # camera_data[cam] -- middle image  --> bounding boxes
    # plot boxes [(x1,y1,x2,y2)] x,y  x1 <  x < x2 , y1 < y < y2 
    
    #labeled_image_toplot = mpimg.imread(labeled_image)
    #plt.figure(figsize=(20,10))

    img = camera_data[cam]
    # img = labeled_image_toplot


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img)

    # Run Image Labeling
    lab_img, bb ,COLORS,class_ids, classes = run_yolo(camera_data[RIGHT_CAM], yolo_cfg_path, yolo_class_path, yolo_weights_path)

    # Map Radar PC in Camera Space
    radar_camera_space = single_radar_to_camera(radar_data[rad], sensor_props.radar_extrinsic[rad],sensor_props.camera_extrinsics[cam])

    # Run DbScan
    points,unique_labels,labels,cluster_dict = run_cluster_3(radar_camera_space)

    # Map Results of DbScan in Image Space
    radar_ImgSpace = single_camera_to_img(radar_camera_space, sensor_props.camera_intrinsics[cam])

    # Filter
    filtered_indices, filtered_points, filtered_labels = filter_radar(radar_ImgSpace, labels)

    # Merge
    img_lab, lab_radar_dict = merge_labels(bb, img, filtered_points, class_ids, classes, COLORS, unique_labels)

    # Show Image
    cv2.imshow('Image with highlighted points', img_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #multi_radar_to_img(radar_data, sensor_props.radar_extrinsic,
    #                sensor_props.camera_extrinsics[cam], sensor_props.camera_intrinsics[cam], image=img)

    #plt.savefig(f"view_100_right.png", dpi=200)