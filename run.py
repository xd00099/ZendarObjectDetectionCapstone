from src.data.dataset import Scene, CameraRadarProps
from src.radar_to_image import single_radar_to_img, multi_radar_to_img

LEFT_CAM = '21248038'
MID_CAM = '20438665'
RIGHT_CAM = '21248039'

LEFT_RADAR = 'ZRVE1002'
MID_RADAR = 'ZRVC2001'
RIGHT_RADAR = 'ZRVE1001'

filepath = 'data/00000.npz'
sensor_prop_path = 'data/extrinsics_intrinsics.npz'

if __name__ == '__main__':

    data = Scene(filepath)
    sensor_props = CameraRadarProps(sensor_prop_path)

    radar_data = data.get_radar_data()
    camera_data = data.get_camera_data()
    cam = MID_CAM

    multi_radar_to_img(radar_data, sensor_props.radar_extrinsic, 
                       sensor_props.camera_extrinsics[cam], sensor_props.camera_intrinsics[cam], image=camera_data[cam])
