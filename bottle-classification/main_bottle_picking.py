# Adding modules
# pip install rembg
from robot_control.panda_robot_client import panda_robot_client

import cv2
import numpy as np
from functions_bottle_recog import bottle_recog_rembg, calcuate_pixels_pose
from functions_bottle_picking import pixel_to_3D_pose_panda
from functions_bottle_classification import classify_bottles
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    # load image
    # image = cv2.imread('/home/wanze/Grasp_test_WuLei/bottle-classification/bottles.png')
    image = cv2.imread('/home/wanze/Grasp_test_WuLei/rgb_grasp_0_rgb.png')
    # initial pose
    T1 = [0.0, -np.pi / 4, 0.0, -2 * np.pi / 3, 0.0, np.pi / 3, np.pi / 4]
    camera_pose = np.asarray([0.43504,0.02237,0.71209,-0.692943215,0.222198322,-0.640276134,0.245975807])
    initial_pose = [0.42345189332962036, -0.02387768216431141, 0.6, 0.917822003364563, -0.395513117313385, -0.03175073117017746, 0.012807437218725681]
    place_cans_pose = np.asarray([0.5340031981468201, -0.5361143350601196, 0.41825756430625916, 0.6544522643089294, -0.7557626366615295, 0.017745813354849815, 0.014147541485726833])
    place_cans_joint_pose = np.asarray([-0.9052586555480957, 0.7408181428909302, 0.20681923627853394, -1.0247902870178223, -0.1748465746641159, 1.7844789028167725, 0.9949672818183899])
    place_bottles_pose = np.asarray([0.3310513496398926, -0.605552613735199, 0.44180047512054443, -0.6447434425354004, 0.7638356685638428, -0.02863566018640995, 0.006400275509804487])
    place_bottles_joint_pose = np.asarray([-1.306434154510498, 0.4400630295276642, 0.3152512013912201, -1.4720793962478638, -0.11137084662914276, 1.9394553899765015, -2.49106764793396])
    # seperate aim area
    seperated_areas, crop_imgs = bottle_recog_rembg(image)
    # panda_control = panda_robot_client()
    # input('please press enter to continue ..')
    # res_0 = panda_control.moveToJoint(place_bottles_joint_pose)
    # input('please press enter to continue ..')
    # res_0 = panda_control.moveToJoint(place_cans_joint_pose)

    # calculate object pose in image
    grasping_pose = []
    i = 0
    for boject_area, crop_img in zip(seperated_areas, crop_imgs):
        if len(crop_img) > 0: 
            # plt.imshow(crop_img)
            # plt.show()
            pixel_1, pixel_2 = calcuate_pixels_pose(boject_area)
            print('pixels',pixel_1)
            pose = pixel_to_3D_pose_panda(pixel_1, pixel_2, camera_pose)
            grasping_pose.append(pose)
        #pred = classify_bottles(crop_img)
        prediction = np.array([1,0,1])
        # print('pres',pred)
    panda_control = panda_robot_client()
    input('please press enter to continue ..')
    res_0 = panda_control.moveToJoint(T1)
    panda_control.moveGripper(0.08)
    input('please press enter to continue ..')

    camera = panda_control.moveToPose(camera_pose)
    input('please press enter to continue ..')
    res_0 = panda_control.moveToJoint(T1)
    panda_control.moveGripper(0.08)

    # add box: 
    box_name = 'box'
    refer_frame = 'world'
    box_size = (0.35, 0.60, 0.17)
    object_pose = [0.455,-0.01,0.085,0,0,0,1]
    object_list = panda_control.add_box(box_name, refer_frame, object_pose, box_size)

    for pose in grasping_pose: 
        print(pose)
        bottle_class = prediction[i]
        input('please press enter to continue ..')
        pre_grasp_pose = [pose[0], pose[1], pose[2] + 0.15, pose[3], pose[4], pose[5], pose[6]]
        sim_pose = [pose[0], pose[1], pose[2]+0.015, pose[3], pose[4], pose[5], pose[6]]
        res_1 = panda_control.moveToPose(pre_grasp_pose)
        time.sleep(0.2)
        input('please press enter to continue ..')
        res_2 = panda_control.moveToPose(sim_pose)
        time.sleep(0.2)
        if bottle_class == 1:
            panda_control.moveGripper(0.001)
            time.sleep(0.5)
            input('please press enter to continue ..')
            res_3 = panda_control.moveToPose(pre_grasp_pose)
            input('please press enter to continue ..')
            res_4 = panda_control.moveToJoint(place_bottles_joint_pose)
            time.sleep(0.2)
            panda_control.moveStop()
            panda_control.moveGripper(0.08)
            time.sleep(0.5)
        elif bottle_class == 0:
            panda_control.moveGripper(0.05)
            time.sleep(0.5)
            input('please press enter to continue ..')
            res_3 = panda_control.moveToPose(pre_grasp_pose)
            input('please press enter to continue ..')
            res_4 = panda_control.moveToJoint(place_cans_joint_pose)
            time.sleep(0.2)
            panda_control.moveStop()
            panda_control.moveGripper(0.08)
            time.sleep(0.2)
        else:
            print('wrong!') 
        i= i+1
    input('please press enter to continue ..')
    res_0 = panda_control.moveToJoint(T1)