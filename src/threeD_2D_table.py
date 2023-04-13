from src.radar_to_image import single_radar_to_img

def threeD_2D_table(radar_pt, radar_extrinsics, camera_extrinsics, camera_intrinsics):
    '''
    Function that takes 3D data point and tranform it into 2D, then store them into a dict
    '''
    # link 3D data to 2D data and store into a dict
    
    radar_img_space = single_radar_to_img(radar_pt, radar_extrinsics, camera_extrinsics, camera_intrinsics)
    
    twoD_to_3D_dict = {}
    radar_tpl = tuple(radar_pt) # 3D data point
    radar_img_tpl = tuple(radar_img_space) # 2D data point
    
    for i in range(len(radar_tpl)):
            new_item = {tuple(radar_tpl[i]): radar_img_tpl[i]}
            twoD_to_3D_dict.update(new_item)
    return twoD_to_3D_dict