# Adding modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from time import time
from tf.transformations import quaternion_matrix


## pose calculate
def calculate_grasping_pose(point1, point2):
    # Calculate the direction vector of the line passing through the points
    direction_vector = point2 - point1
    print('x_vector',direction_vector)
    x_vector = np.array([1, 0, 0])
    # Normalize the direction vector to obtain a unit vector
    #unit_vector = direction_vector / np.linalg.norm(direction_vector)
    # Calculate the dot product of A and B
    dot_product = np.dot(direction_vector, x_vector)

    # Calculate the magnitudes of A and B
    magnitude_A = np.linalg.norm(direction_vector)
    magnitude_B = np.linalg.norm(x_vector)

    # Calculate the cosine of the angle between A and B
    cosine_theta = dot_product / (magnitude_A * magnitude_B)

    # Calculate the angle in radians
    theta = np.arccos(cosine_theta)

    print('theta', theta)
    # Create the rotation matrix based on the unit vector
    #rotation_matrix = np.eye(3)
    #rotation_matrix[:, 2] = unit_vector  # Third column of rotation matrix is the unit vector
    #print('rotation_matrix',rotation_matrix)
    # Create the homogeneous matrix
    T_0B = np.eye(4)
    T_0B[:3, 3] = point1  # Set the translation vector

    # hand_tcp 绕x-axis 180
    T_BD = np.asarray([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    
    #z rotate around-axixs -theta
    T_DE = np.array([[np.cos(-theta), -np.sin(-theta), 0,0],
                                [np.sin(-theta), np.cos(-theta), 0,0],
                                [0, 0, 1,0],
                                [0,0,0,1]])


    #hand_tcp-end effector
    t4 = np.asarray([[0],[0],[0.103]])
    q4 = np.asarray([0.000,0.000,-0.383,0.924])
    T_1tcp = quaternion_matrix(q4)
    for i in range(3):
        T_1tcp[i][3] = T_1tcp[i][3] + t4[i]

    #calculate link8
    inv_T_1tcp = np.linalg.inv(T_1tcp)
    T_01 = T_0B@T_BD@T_DE@inv_T_1tcp
    print (T_0B)
    print (T_01)
  
    
    #calculate quaternion
    rot_matrix = T_01[:3, :3]
    position = T_01[:3, 3]
    R01 = R.from_matrix(rot_matrix)
    orientation  = R01.as_quat()
    # orientation = quaternion_from_matrix(rotation_matrix)
    return position, orientation

def pixel_to_3D_pose(contact,Tcp_pos):
    t = np.asarray([Tcp_pos[0],
                    Tcp_pos[1],
                    Tcp_pos[2]])
    q = np.asarray([Tcp_pos[3],Tcp_pos[4],
                    Tcp_pos[5],Tcp_pos[6]])
    T_we = quaternion_matrix(q)
    for i in range(3):
        T_we[i][3] = T_we[i][3] + t[i]
    print(T_we)
    cam_params = [546.2562580065428, 549.4296433161169, 313.0437650794175, 231.8829725503235]
    # transfrom grasping pixel to world frame:
    f_length_x = cam_params[0]
    f_length_y = cam_params[1]
    c_x = cam_params[2]
    c_y = cam_params[3]
    cam_intr = np.array([[f_length_x, 0.0, c_x], [0.0, f_length_y, c_y], [0.0, 0.0, 1.0]])
    T_ec = np.array([[-0.66436242, -0.00796844, -0.7473681,  -0.0326596 ],
                    [-0.74736862, -0.00351288,  0.66440033,  0.05012495],
                    [-0.00791965,  0.99996208, -0.00362154,  0.02556528],
                    [ 0.,          0.,          0.,          1.        ]])
    T_wc = T_we @ T_ec
    R_wc = T_wc[0:3, 0:3]
    t_wc = T_wc[0:3, 3:4]
    pix_contact_pos = np.array([[contact[0]], [contact[1]], [1]])
    P_v = (R_wc @ np.linalg.inv(cam_intr)) @ pix_contact_pos
    zw = 0.19
    zc = (zw - t_wc[2])/P_v[2]
    xw = (t_wc[0] + zc*P_v[0])[0]
    yw = (t_wc[1] + zc*P_v[1])[0]
    print('contact position is: ', (xw, yw))
    positin_w = np.array([xw, yw, zw])
    return positin_w

def image_rotation(image, degree):
    #rotation image
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Define the rotation angle (180 degrees)
    rotation_angle = degree

    # Get the center of the image to rotate around
    center = (width // 2, height // 2)
    print('center',center)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def pixel_to_3D_pose_panda(pixels_1, pixels_2, initial_pose):
    position_1 = pixel_to_3D_pose(pixels_1,initial_pose)
    # print('position1',position_1)
    position_2 = pixel_to_3D_pose(pixels_2,initial_pose)
    # print('position2',position_2)
    # Calculate the orientation
    position, orientation = calculate_grasping_pose(position_1, position_2)
    # print('position:',position,'orientation:',orientation)
    pose = np.concatenate((position, orientation), axis=0)
    print('pose', pose)
    return pose