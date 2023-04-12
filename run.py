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
    plt.imshow(img)

    # Run Image Labeling
    lab_img, bb ,COLORS,class_ids, classes = run_yolo(camera_data[RIGHT_CAM], yolo_cfg_path, yolo_class_path, yolo_weights_path)

    plt.imshow(lab_img[0])
    plt.show()

    # Map Radar PC in Camera Space
    radar_camera_space = single_radar_to_camera(radar_data[rad], sensor_props.radar_extrinsic[rad],sensor_props.camera_extrinsics[cam])

    # Run DbScan
    points,unique_labels,labels,cluster_dict = run_cluster_3(radar_camera_space)

    #Map Results of DbScan in Image Space
    radar_ImgSpace = single_camera_to_img(radar_camera_space, sensor_props.camera_intrinsics[cam])

    # Filter
    filtered_indices, filtered_points, filtered_labels = filter_radar(radar_ImgSpace, labels)


    ######### MERGE #########
    # Bounding boxes
    boxes = bb
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros_like(img[:, :, 0])
    # Loop through each bounding box and fill the corresponding region in the mask
    for label_types in range(len(bb)):  # Loop through label types
        for i in range(len(bb[label_types])):
            x1, y1, x2, y2 = bb[label_types][i,:]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            mask[y1:y2, x1:x2] = 255
            confidence = 0 # dummy value
            draw_prediction(img, class_ids[i],confidence, x1, y1, x2, y2,classes,COLORS)
            
        # Loop through each 2D point and check if it falls within any of the bounding boxes for point in filtered_points:
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for label, col in zip(unique_labels, colors):
            if label == -1:
                continue
            cluster_points = filtered_points[np.array(labels) == label, :]
            x, y, z = cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]

            #x, y = point
            #x = int(x)
            #y = int(y)
            for x, y in x,y:
                if mask[y, x] == 255:
                    # If the point is within a bounding box, draw a red circle around it
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                else:
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    # Display the image with highlighted points

    #plt.imshow('Image with highlighted points', img)
    #plt.show()

    cv2.imshow('Image with highlighted points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ######### MERGE #########



    #multi_radar_to_img(radar_data, sensor_props.radar_extrinsic,
    #                sensor_props.camera_extrinsics[cam], sensor_props.camera_intrinsics[cam], image=img)

    #plt.savefig(f"view_100_right.png", dpi=200)