import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
def local_to_global(p, q, local_point):

    pass

def global_to_local(p, q, global_point):

    translated_point = global_point - p

    q_inverse = np.quaternion.conj(q)

    local_point = quaternion.rotate_vectors(q_inverse, translated_point)

    return local_point

def calculate_focal_length(fov_degrees, image_width):

    fov_radians = np.deg2rad(fov_degrees)
    focal_length = (image_width / 2) / np.tan(fov_radians / 2)
    return focal_length

def annotate_image_offline(annotation, image, fov):
            
    local_point = annotation['curr_local_coords']
    point_3d = [local_point[0], -local_point[1], -local_point[2]] #inconsistency between habitat camera frame and classical convention
    focal_length = calculate_focal_length(fov, image.shape[1])
    x = focal_length * point_3d[0] / point_3d[2]
    x_pixel = int(image.shape[1] / 2 + x)

    y =  focal_length * point_3d[1] / point_3d[2]
    y_pixel = int(image.shape[0] / 2 + y)
    label = annotation['obj']
    # Assuming you have an image captured from the sensor
    cv2.circle(image, (x_pixel, y_pixel), 5, (255, 0, 0), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.85
    font_color = (0, 0, 0)
    font_thickness = 1
    text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_x = int(x_pixel - text_size[0] // 2)
    text_y = int(y_pixel + text_size[1] + 15)
    rect_top_left = (text_x-3, text_y - text_size[1])  # Top-left corner
    rect_bottom_right = (text_x + text_size[0], text_y + 3)  # Bottom-right corner

    # Draw the rectangle to highlight the background
    cv2.rectangle(image, rect_top_left, rect_bottom_right, (255, 255, 255), -1)
    cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return image