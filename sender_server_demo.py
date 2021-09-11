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
import sys
import numpy as np

from PyQt5.QtCore import Qt, QSize, QTimer, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage
import cv2
import random
import time

W = 1920
H = 2160
x = 0.
y = 0.
z = 0.

UDP_IP = '192.168.50.216'  # The server's hostname or IP address
UDP_PORT = 65432  # The port used by the server

DELTA_L = 4  # an essential value
sign = 1


def main():
    global x, y, z
    # init qt
    app = QApplication([])

    window = QWidget()
    window.setLayout(QGridLayout(window))
    window.setMinimumSize(QSize(W, H))

    label = QLabel()
    label.setFixedSize(W, H)
    window.layout().addWidget(label, 0, 0)

    window.show()

    # UDP
    t = UDPThread()
    t.start()

    timer = QTimer()
    timer.timeout.connect(lambda: next_frame_slot(label))
    timer.start(int(1000. / 50))

    return app.exit(app.exec_())


def next_frame_slot(label: QLabel):
    # 绘制图像的主进程
    global x, y, z, map_mask, sign

    time_0 = time.time()

    frame = np.ones((H, W, 3), dtype=np.uint8) * 200

    # draw circles. 160 pixel, 0.18 m
    center_x = x / 0.18 * 160 + W / 2.
    center_y = H / 2. - y / 0.18 * 160
    r = 20 + z / 1. * 20
    frame = cv2.circle(frame, (int(center_x), int(center_y)), int(r), (0, 0, 245), -1)

    gpu_org = cv2.cuda_GpuMat()
    gpu_lab = cv2.cuda_GpuMat()
    gpu_org.upload(frame)

    # embedding
    gpu_lab = cv2.cuda.cvtColor(gpu_org, code=cv2.COLOR_BGR2Lab)  # transform from BGR to CIELAB

    sign = -sign
    frame_lab = gpu_lab.download()
    lightness_masked = frame_lab[:, :, 0].astype(np.int32)
    lightness_masked += sign * DELTA_L * map_mask
    lightness_masked[lightness_masked > 255] = 255  # pay attention to the value that more than 255.
    lightness_masked[lightness_masked < 0] = 0  # pay attention to the value that less than 255.

    frame_lab[:, :, 0] = lightness_masked.astype(np.uint8)
    gpu_lab.upload(frame_lab)
    gpu_org = cv2.cuda.cvtColor(gpu_lab, cv2.COLOR_Lab2RGB)
    frame = gpu_org.download()

    time_1 = time.time()
    print("time:", (time_1 - time_0), " fps:", (1 / (time_1 - time_0)))

    # change to QImage
    image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    label.setPixmap(pixmap)


class UDPThread(QThread):
    """
    负责监听数据的子进程
    """

    def __init__(self):
        # 初始化函数
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # SOCK_DGRAM: UDP
        super(UDPThread, self).__init__()

    def run(self):
        global x, y, z

        self.udp_sock.bind((UDP_IP, UDP_PORT))
        time_start = time.time()
        while True:
            data, addr = self.udp_sock.recvfrom(1024)  # buffer size is 1024 bytes
            data_str = data.decode()
            print("received message: %s" % data_str)

            if data_str == "quit":
                break

            msg = json.loads(data_str)
            x = msg["x"]
            y = msg["y"]
            z = msg["z"]

            time_end = time.time()

            # if (time_end - time_start) > 15:
            #     break

        print("close the socks")
        self.udp_sock.close()


if __name__ == '__main__':
    apriltag_map = cv2.imread('tag_map/apriltagMap_9x9_1920.0000m.png', flags=cv2.IMREAD_GRAYSCALE)
    [height, width] = apriltag_map.shape
    map_mask = np.zeros([height, width], dtype=np.int32)
    # white: 255 > 1 black: 0 > -1 no color: 127 > 0
    map_mask[apriltag_map >= 255] = 1  # white
    map_mask[apriltag_map <= 0] = -1  # black

    exit_code = main()
    sys.exit(exit_code)

# UDP_IP = '192.168.50.216'  # The server's hostname or IP address
# UDP_PORT = 65432  # The port used by the server
#
# udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # SOCK_DGRAM: UDP
# udp_sock.bind((UDP_IP, UDP_PORT))
#
# while True:
#     data, addr = udp_sock.recvfrom(1024)  # buffer size is 1024 bytes
#     data_str = data.decode()
#     print("received message: %s" % data_str)
#     msg = json.loads(data_str)
#
#     if data_str == "quit":
#         break
#
# udp_sock.close()
