#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-02-29 10:06:11
@LastEditors  : LI Jinjie
@LastEditTime : 2020-02-29 18:22:23
@Units        : meters
@Description  : apriltag map, 左下角为零点，向右递增，满格换行继续从左向右
@Dependencies : None
@NOTICE       : None
'''
import numpy as np
import cv2
from apriltag import apriltag

SIDE_LENGTH = 0.3  # the side length of every tag
CENTER_DISTANCE = 0.9  # the distance between two tags' centers
NUM_X = 5  # the tags' number in x direction
NUM_Y = 5  # the tags' number in y direction


def get_coordinates(tag_id):
    ''' Get the world coordinates of QR code
    :param tag_id: the id of QR code
    :return:the coordinates of the input tag's center and the coordinates of the square's four vertices
    :rtype: dict, float
    '''
    x_center = tag_id % NUM_X * CENTER_DISTANCE
    # return the integer part of the quotient
    y_center = tag_id // NUM_Y * CENTER_DISTANCE

    # 4/5, 因为apriltags边长8个像素，最外一圈是白色，黑色角点间距8像素
    dxy = (SIDE_LENGTH / 2) * (4 / 5)

    # order: lb-rb-rt-lt
    mf_coordinates = np.zeros((4, 2), dtype='float')

    # coordinates of left bottom point
    mf_coordinates[0, :] = (x_center-dxy, y_center-dxy)
    # coordinates of right bottom point
    mf_coordinates[1, :] = (x_center+dxy, y_center-dxy)
    # coordinates of right top point
    mf_coordinates[2, :] = (x_center+dxy, y_center+dxy)
    # coordinates of left top point
    mf_coordinates[3, :] = (x_center-dxy, y_center+dxy)

    dict_coordinators = {'center': np.array(
        (x_center, y_center), dtype='float'), 'lb-rb-rt-lt': mf_coordinates}

    return dict_coordinators


if __name__ == '__main__':
    print("test the functions: \n")
    imagepath = 'Raw_pictures/image5.png'
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("1", image)
    detector = apriltag("tag36h11")
    detections = detector.detect(image)
    world_coor = get_coordinates(detections[0]['id'])
    print("id0=\n", detections[0]['id'])
    print("tags_coordinations= \n",
          world_coor['center'], "\n", world_coor['lb-rb-rt-lt'])
