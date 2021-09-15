#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: sender_server_demo.py
Date: 2021/9/6 10:03
LastEditors: LI Jinjie
LastEditTime: 2021/9/6 10:03
Description: file content
'''

import socket
import json

import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from multiprocessing import Process, Queue
import sys
import numpy as np
import time

# window dimensions
WIDTH = 1920
HEIGHT = 2160
nRange = 1.0

# params
DELTA_L = 4  # an essential value
sign = 1

# position of circle
x = 0.
y = 0.
z = 0.

# background picture
frame_org = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 200
frame_0 = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 200
frame_1 = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 200

# UDP related
UDP_IP = '192.168.50.216'  # The server's hostname or IP address
UDP_PORT = 65432  # The port used by the server


def udp_process(queue):
    """
    process function for udp
    :param queue: Queue()
    :return:
    """
    print(">>> Create the child process successfully!!!")
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # SOCK_DGRAM: UDP
    udp_sock.bind((UDP_IP, UDP_PORT))
    # time_start = time.time()
    while True:
        data, addr = udp_sock.recvfrom(1024)  # buffer size is 1024 bytes
        data_str = data.decode()
        print("received message: %s" % data_str)

        if data_str == "quit":
            break

        queue.put(data_str)
        # msg = json.loads(data_str)
        # x = msg["x"]
        # y = msg["y"]
        # z = msg["z"]

        # time_end = time.time()
        # if (time_end - time_start) > 15:
        #     break

    print("close the socks")
    udp_sock.close()


class OpenGLPainter:

    def __init__(self, w, h, queue, map_mask):
        self.width = w
        self.height = h
        self.queue = queue
        self.map_mask = map_mask

        # glclearcolor (r, g, b, alpha)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        glutDisplayFunc(self.display)
        # glutReshapeFunc(self.reshape)
        # glutKeyboardFunc(self.keyboard)
        glutIdleFunc(self.idle)

    def color_space_conversion(self, x, y, w, h, sign, img_org, img_masked):
        down = int(y - h / 2.)
        up = int(y + h / 2. + 1)
        left = int(x - w / 2.)
        right = int(x + w / 2. + 1)

        img_lc = img_org[down: up, left: right, :]
        img_lc_lab = cv2.cvtColor(img_lc, code=cv2.COLOR_BGR2Lab)
        lightness_c = img_lc_lab[:, :, 0].astype(np.int32)
        lightness_c += sign * DELTA_L * self.map_mask[down: up, left: right]
        lightness_c[lightness_c > 255] = 255
        lightness_c[lightness_c < 0] = 0
        img_lc_lab[:, :, 0] = lightness_c.astype(np.uint8)
        img_masked[down: up, left: right, :] = cv2.cvtColor(img_lc_lab, code=cv2.COLOR_Lab2BGR)
        return img_masked

    def idle(self):
        global sign, x, y, z, frame_org, frame_0, frame_1

        time_0 = time.time()

        # get data from UDP
        data_str = None
        try:
            data_str = self.queue.get(block=False)
        except:
            pass
        if data_str is not None:
            # print(data_str)
            msg = json.loads(data_str)
            x = msg["x"]
            y = msg["y"]
            z = msg["z"]

        # draw circles. 160 pixel, 0.18 m
        center_x = int(x / 0.18 * 160 + self.width / 2.)
        center_y = int(self.height / 2. - y / 0.18 * 160)
        r = int(20 + z / 1. * 20)

        if sign == -1:
            img_masked = frame_0.copy()
        else:
            img_masked = frame_1.copy()

        # draw the circle on the background first!
        frame = cv2.circle(frame_org.copy(), (center_x, center_y), r, (0, 0, 245), -1)
        frame = self.color_space_conversion(center_x, center_y, r * 2, r * 2, sign, frame, img_masked)

        # local_img = frame[center_y - r: center_y + r + 1, center_x - r: center_x + r + 1, :]
        # print(local_img.shape)
        # cv2.imshow("local_img", local_img)
        # cv2.waitKey(1)

        sign = -sign

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        time_1 = time.time()
        print(">>> speed is ", 1 / (time_1 - time_0), " fps")

        # you must convert the image to array for glTexImage2D to work
        # maybe there is a faster way that I don't know about yet...

        # Create Texture
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGB,
                     self.width, self.height,
                     0,
                     GL_RGB,
                     GL_UNSIGNED_BYTE,
                     frame)
        # cv2.imshow('frame', image)
        glutPostRedisplay()

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        # this one is necessary with texture2d for some reason
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # 我推测这里是通过改变纹理来显示图像的。glTexParameterf设置了缩放的性质

        # Set Projection Matrix
        glMatrixMode(GL_PROJECTION)  # Applies subsequent matrix operations to the projection matrix stack.
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)

        # Switch to Model View Matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Draw textured Quads
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(self.width, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(self.width, self.height)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(0.0, self.height)
        glEnd()

        glFlush()
        glutSwapBuffers()

    def reshape(self, w, h):
        if h == 0:
            h = 1

        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)

        glLoadIdentity()
        # allows for reshaping the window without distoring shape

        if w <= h:
            glOrtho(-nRange, nRange, -nRange * h / w, nRange * h / w, -nRange, nRange)
            # glOrtho(-nRange, nRange, -nRange * h / w, nRange * h / w, -nRange, nRange)
        else:
            glOrtho(-nRange * w / h, nRange * w / h, -nRange, nRange, -nRange, nRange)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def keyboard(self, key, x, y):
        global anim
        if key == chr(27):
            sys.exit()


def main():
    global frame_0, frame_1

    # create a child process for udp
    q = Queue()
    p = Process(target=udp_process, args=(q,))
    p.start()

    apriltag_map = cv2.imread('tag_map/apriltagMap_9x9_1920.0000m.png', flags=cv2.IMREAD_GRAYSCALE)
    [height, width] = apriltag_map.shape
    map_mask = np.zeros([height, width], dtype=np.int32)
    # white: 255 > 1 black: 0 > -1 no color: 127 > 0
    map_mask[apriltag_map >= 255] = 1  # white
    map_mask[apriltag_map <= 0] = -1  # black

    frame_Lab = cv2.cvtColor(frame_0, code=cv2.COLOR_BGR2Lab)  # transform from BGR to CIELAB
    lightness_masked = frame_Lab[:, :, 0].astype(np.int32)
    lightness_masked += -1 * DELTA_L * map_mask
    lightness_masked[lightness_masked > 255] = 255  # pay attention to the value that more than 255.
    lightness_masked[lightness_masked < 0] = 0
    frame_Lab[:, :, 0] = lightness_masked.astype(np.uint8)
    frame_0 = cv2.cvtColor(frame_Lab, code=cv2.COLOR_Lab2BGR)  # transform from CIELAB to BGR

    frame_Lab = cv2.cvtColor(frame_1, code=cv2.COLOR_BGR2Lab)  # transform from BGR to CIELAB
    lightness_masked = frame_Lab[:, :, 0].astype(np.int32)
    lightness_masked += +1 * DELTA_L * map_mask
    lightness_masked[lightness_masked > 255] = 255  # pay attention to the value that more than 255.
    lightness_masked[lightness_masked < 0] = 0
    frame_Lab[:, :, 0] = lightness_masked.astype(np.uint8)
    frame_1 = cv2.cvtColor(frame_Lab, code=cv2.COLOR_Lab2BGR)  # transform from CIELAB to BGR

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)  # double buffered window, RGB, with a depth buffer
    glutInitWindowSize(WIDTH, HEIGHT)
    glutInitWindowPosition(0, 0)
    glutCreateWindow("OpenGL + OpenCV")

    painter = OpenGLPainter(WIDTH, HEIGHT, q, map_mask)
    glutMainLoop()


if __name__ == '__main__':
    main()
