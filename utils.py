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
