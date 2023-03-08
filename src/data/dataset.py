import numpy as np
from src.data.load import load_data, load_props

class Scene:
    """
    This class stores all the data for a particular scene taken on a moving vehicle.
    """
    def __init__(self, file_path):
        """
        Constructor method to initialize the object with radar, images, and lidar data loaded from the file path.

        :param file_path: (str) File path to load the data.
        """
        self.radar, self.images, self.lidar = load_data(file_path)
    
    def get_camera_data(self):
        """
        Method to return the images data.

        :return: (numpy.ndarray) Images data.
        """
        return self.images
    
    def get_lidar_data(self):
        """
        Method to return the lidar data.

        :return: (numpy.ndarray) Lidar data.
        """
        return self.lidar
    
    def get_radar_data(self):
        """
        Method to return the radar data.

        :return: (numpy.ndarray) Radar data.
        """
        return self.radar
    

class CameraRadarProps:

    def __init__(self, props_path):
        
        self.camera_intrinsics, self.camera_extrinsics,  \
        self.lidar_extrinsic, self.radar_extrinsic = load_props(props_path=props_path)
