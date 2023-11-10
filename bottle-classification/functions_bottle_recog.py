import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove 

# def get_image(path):
#     #获取图片
#     img=cv2.imread(path)
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     return img, gray

# def get_image(path):
#     #获取图片
#     img=cv2.imread(path)
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     return img, gray

def delet(image):
    width, height = image.shape[:2]
    for i in range(width):
        for j in range(height):
            for k in range(3):
                #img[i][j][k] = 255
                if int(image[i][j][1])-int(image[i][j][0]) >=50:
                    image[i][j] = 0
                    if int(image[i][j][0]) <= 20:
                        image[i][j] = 0             
    return image

def Gaussian_Blur(gray):
    # 高斯去噪
    blurred = cv2.GaussianBlur(gray, (9, 9),0)
    return blurred

def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    return gradX, gradY, gradient

def Thresh_and_blur(gradient):
    
    blurred = cv2.GaussianBlur(gradient, (9, 9),0)
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    
    return thresh
    
def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    ret, thresholded = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
    # plt.imshow(thresholded)
    # plt.show()
    return thresholded
    
def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    (cnts, _) = cv2.findContours(closed.copy(), 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))
    return box, cnts

def findcnts_and_box_point_all(closed):
    # Find all contours in the binary image
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store all bounding box points
    all_boxes = []
    
    # Compute bounding box for each contour and store the points
    for contour in cnts:
        rect = cv2.minAreaRect(contour)
        box = np.intp(cv2.boxPoints(rect))
        all_boxes.append(box)
    return all_boxes

def drawcnts_and_cut(original_img, box):
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    # draw a bounding box arounded the detected barcode and display the image
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1+hight, x1:x1+width]
    return draw_img, crop_img

def drawcnts_all_boxes(original_img, boxes):
    # Make a copy of the original image to avoid modifying the input image
    draw_img = original_img.copy()
    crop_imgs = []
    H,W,channels = original_img.shape
    # draw a bounding box arounded the detected barcode and display the image
    for box in boxes:
        draw_img = cv2.drawContours(draw_img, [box], -1, (0, 0, 255), 3)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        if x1<=0:
            x1 = 0
        x2 = max(Xs)
        if x2>=W:
            x2 = W
        y1 = min(Ys)
        if y1<= 0:
            y1 = 0
        y2 = max(Ys)
        if y2>= H:
            y2 = H
        hight = y2 - y1
        width = x2 - x1
        crop_img = original_img[y1:y1+hight, x1:x1+width]
        crop_imgs.append(crop_img)
    return draw_img, crop_imgs

def least_squares_fit(points):
    # Extract x and y coordinates from the points
    x = np.array([point[0][0] for point in points])
    y = np.array([point[0][1] for point in points])
    
    # Calculate the coefficients (a and b) for the equation y = ax + b
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

def calculate_center(points):
    # Calculate the average of x-coordinates and y-coordinates
    num_points = len(points)
    if num_points == 0:
        return None  # Return None for an empty list of points
    # Calculate the sum of x-coordinates and y-coordinates
    sum_x = sum(point[0][0] for point in points)
    sum_y = sum(point[0][1] for point in points)
    
    # Calculate the average
    center_x = sum_x / num_points
    center_y = sum_y / num_points
    return center_x, center_y

def calcuate_pixels_pose(points):
    a,b = least_squares_fit(points)
    center_x, center_y = calculate_center(points)
    center_x_2 = center_x + 30
    center_y_2 = a*center_x_2 + b
    pixel_1 = np.array([center_x, center_y])
    pixel_2 = np.array([center_x_2, center_y_2])
    print('grasping_pixel',pixel_1)
    #pixels = np.array([pixel_1, pixel_2])
    return pixel_1, pixel_2

def bottle_recog_proceed(img):
    original_img = img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #image = delet(original_img)
    # plt.imshow(original_img)
    # plt.show()
    # plt.imshow(image)
    # plt.show()
    blurred = Gaussian_Blur(gray)
    # plt.imshow(blurred)
    # plt.show()
    gradX, gradY, gradient = Sobel_gradient(blurred)
    # plt.imshow(gradient)
    # plt.show()
    thresh = Thresh_and_blur(gradient)
    # plt.imshow(thresh)
    # plt.show()
    closed = image_morphology(thresh)
    # plt.imshow(closed)
    # plt.show()

    box, seperated_areas = findcnts_and_box_point(closed)
    draw_img, crop_img = drawcnts_and_cut(original_img,box)
    # 暴力一点，把它们都显示出来看看
    print('len(seperated_areas)',len(seperated_areas))
    # p1,p2 = calcuate_pixels_pose(seperated_areas[0])


    boxes = findcnts_and_box_point_all(closed)
    draw_img, crop_imgs = drawcnts_all_boxes(original_img, boxes)
    #pixel1,pixel2 = calculate
    #draw_img_img, crop_img_img = drawcnts_and_cut(original_img,all_boxes)
    # cv2.imshow('original_img', original_img)
    # cv2.imshow('blurred', blurred)
    # cv2.imshow('gradX', gradX)
    # cv2.imshow('gradY', gradY)
    # cv2.imshow('final', gradient)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('closed', closed)
    # cv2.imshow('draw_img', draw_img)
    # cv2.imshow('crop_img', crop_imgs[1])
    # cv2.waitKey(20171219)
    return  seperated_areas, crop_imgs

def bottle_recog_rembg(img):
    # Removing the background from the given Image 
    output_img = remove(img) 
    # cv2.imshow('output_img', output_img)
    gray_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (11, 11),0)
    (_, thresh) = cv2.threshold(blurred, 3, 255, cv2.THRESH_BINARY)
    closed = image_morphology(thresh)
    # plt.imshow(thresh)
    # plt.show()
    #closed = image_morphology(thresh)
    # plt.imshow(closed)
    # plt.show()

    box, seperated_areas = findcnts_and_box_point(closed)
    draw_img, crop_img = drawcnts_and_cut(img,box)
    # 暴力一点，把它们都显示出来看看
    print('len(seperated_areas)',len(seperated_areas))
    # p1,p2 = calcuate_pixels_pose(seperated_areas[0])
    boxes = findcnts_and_box_point_all(closed)
    draw_img, crop_imgs = drawcnts_all_boxes(img, boxes)
    #pixel1,pixel2 = calculate
    #draw_img_img, crop_img_img = drawcnts_and_cut(original_img,all_boxes)
    # cv2.imshow('original_img', original_img)
    # cv2.imshow('blurred', blurred)
    # cv2.imshow('gradX', gradX)
    # cv2.imshow('gradY', gradY)
    # cv2.imshow('final', gradient)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('closed', closed)
    # cv2.imshow('draw_img', draw_img)
    # cv2.imshow('crop_img', crop_imgs[1])
    # cv2.waitKey(20171219)
    plt.imshow(draw_img)
    plt.show()
    return  seperated_areas, crop_imgs