# coding: utf-8

import argparse

from darknet_util import *
from darknet import Darknet
from preprocess import prep_image, process_img, inp_to_image

import os
import torch
import torchvision
import cv2
import PIL
from PIL import Image


# import sys
# import re
# import time
# import pickle
# import shutil
# import random
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.widgets import Cursor
# from matplotlib.image import AxesImage
# from scipy.spatial.distance import cityblock
# from tqdm import tqdm

# -------------------------------------
# for matplotlib to displacy chinese characters correctly


from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

use_cuda = True  # True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(
    'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')

if use_cuda:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
print('=> device: ', device)

local_model_path = './checkpoints/epoch_39.pth'
local_car_cfg_path = './car.cfg'
local_car_det_weights_path = './car_detect.weights'


# ------------------------------------- vehicle detection model
class Car_DC(object):
    # def __init__(self,
    #              src_dir,
    #              dst_dir,
    #              car_cfg_path=local_car_cfg_path,
    #              car_det_weights_path=local_car_det_weights_path,
    #              inp_dim=768,
    #              prob_th=0.2,
    #              nms_th=0.4,
    #              num_classes=1):
    def __init__(self,
                 vdo,
                 car_cfg_path=local_car_cfg_path,
                 car_det_weights_path=local_car_det_weights_path,
                 inp_dim=768,
                 prob_th=0.2,
                 nms_th=0.4,
                 num_classes=1):
        """
        model initialization
        """
        # super parameters
        self.inp_dim = inp_dim
        self.prob_th = prob_th
        self.nms_th = nms_th
        self.num_classes = num_classes
#        self.dst_dir = dst_dir
        self.vdo = vdo

        # initialize vehicle detection model

        self.detector = Darknet(car_cfg_path)
        self.detector.load_weights(car_det_weights_path)
        # set input dimension of image
        self.detector.net_info['height'] = self.inp_dim
        self.detector.to(device)
        self.detector.eval()  # evaluation mode
        print('=> car detection model initiated.')

    def process_predict(self, prediction, prob_th, num_cls, nms_th, inp_dim, orig_img_size):

        # processing detection

        scaling_factor = min([inp_dim / float(x)
                              for x in orig_img_size])  # W, H scaling factor
        output = post_process(prediction,
                              prob_th,
                              num_cls,
                              nms=True,
                              nms_conf=nms_th,
                              CUDA=True)  # post-process such as nms

        if type(output) != int:
            output[:, [1, 3]] -= (inp_dim - scaling_factor *
                                  orig_img_size[0]) / 2.0  # x, w
            output[:, [2, 4]] -= (inp_dim - scaling_factor *
                                  orig_img_size[1]) / 2.0  # y, h
            output[:, 1:5] /= scaling_factor
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(
                    output[i, [1, 3]], 0.0, orig_img_size[0])
                output[i, [2, 4]] = torch.clamp(
                    output[i, [2, 4]], 0.0, orig_img_size[1])
        return output

    def cls_draw_bbox(self, output, orig_img, log_file, frame_nr, imagesize, min_box_size=0.):
        """
        draw bbox to orig_img
        """
        nr_det = []
        box_size = min_box_size * imagesize       # 目标占画面最小像素比, imagesize为画面像素数

        for det in output:                        # 检测数据储存在list 'nr_det'
            # nr_det.append(
            #     (tuple(det[1:3].int()),
            #      tuple(det[3:5].int()),
            #      det[5]
            #      ))
            if abs((det[1] - det[3]) * (det[2] - det[4])) > box_size:
                nr_det.append(
                    (tuple(det[1:3].int()),
                     tuple(det[3:5].int()),
                     det[5]
                     ))

        # for j in range(len(nr_det) - 1):          # 根据检测bounding box位置从左到右排序
        #     for k in range(len(nr_det) - j - 1):
        #         if nr_det[k][0][0] > nr_det[k+1][0][0]:
        #             nr_det[k], nr_det[k+1] = nr_det[k+1], nr_det[k]
        nr_det.sort(key=lambda v: v[0][0])          # 冒泡排序可以直接用sort简化

        for i, sorted_det in enumerate(nr_det):
            pt_1 = sorted_det[0]                   # box左上角坐标
            pt_2 = sorted_det[1]                   # box右下角坐标

            # draw bounding box
            cv2.rectangle(orig_img, pt_1, pt_2, (0, 255, 0), thickness=2)
            cv2.putText(orig_img, '%d' % i, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            x1 = int('{:.0f}'.format(pt_1[0]))     # extract x1y1 x2y2 from torch tensor format
            y1 = int('{:.0f}'.format(pt_1[1]))
            x2 = int('{:.0f}'.format(pt_2[0]))
            y2 = int('{:.0f}'.format(pt_2[1]))
            w = x2 - x1
            h = y2 - y1
            conf = '{:.3f}'.format(sorted_det[2])

            log_file.write(  # 记录结果
                ','.join(
                     map(str, [int(frame_nr), '-1', x1, y1, w, h, conf, 1, 1, 1]))
            )
            log_file.write('\n')


        # for i, det in enumerate(output):
        #     pt_1 = pt_1s[i]
        #     pt_2 = pt_2s[i]
        #     conf = confs[i]
        #
        #     # draw bounding box
        #     cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=2)
        #     cv2.putText(orig_img, '%d' % i, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        #
        #     x1 = int('{:.0f}'.format(pt_1[0]))          # extract x1y1 x2y2 from torch tensor format
        #     y1 = int('{:.0f}'.format(pt_1[1]))
        #     x2 = int('{:.0f}'.format(pt_2[0]))
        #     y2 = int('{:.0f}'.format(pt_2[1]))
        #     w = x2 - x1
        #     h = y2 - y1
        #     conf = '{:.3f}'.format(conf)
        #
        #     log_file.write(  # 记录结果
        #         ','.join(
        #              map(str, [int(frame_nr), '-1', x1, y1, w, h, conf, 1, 1, 1]))
        #     )
        #     log_file.write('\n')

    def detect_classify(self):
        """
        detect ONLY # and NO classify
        """

        clip1 = cv2.VideoCapture(self.vdo)
        width = int(clip1.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频的宽度
        height = int(clip1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频的高度
        fps = int(clip1.get(cv2.CAP_PROP_FPS))  # 视频的帧率
        fourcc = int(clip1.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
        framenum = int(clip1.get(7))  # 视频总帧数

        writer = cv2.VideoWriter('example/project_output.mp4', fourcc, fps, (width, height))
        frame_id = 0

        f_file = open('Detection_log/detection.txt', 'w+')  # 打开log文件准备写入 frame 和 box

        while clip1.isOpened():
            if frame_id < framenum:
                ret, frame = clip1.read()  # ret为是否读到帧画面T/F. clip为帧画面
                print('\r', 'processing frame: %d / %d' % (frame_id, framenum), end='', flush=True)  # 打印当前处理帧ID

                frame2det = process_img(frame, self.inp_dim)
                frame2det = frame2det.to(device)    # put image data to device

                # vehicle detection
                prediction = self.detector.forward(frame2det, CUDA=True)

                # calculating scaling factor
                # orig_img_size = list(frame.size)
                orig_img_size = [width, height]     # 输出视频格式大小和原视频一致
                output = self.process_predict(prediction,
                                              self.prob_th,
                                              self.num_classes,
                                              self.nms_th,
                                              self.inp_dim,
                                              orig_img_size)

                # orig_img = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                # RGB => BGR

                self.cls_draw_bbox(output, frame, f_file, frame_id, (width * height), 0.004)   # 绘制bounding box， 设置最小检测阈值


                # cv2.imshow('Vid_out', frame)
                # cv2.waitKey(int(1000/fps))
                # cv2.waitKey()
                writer.write(frame)
                frame_id += 1
            else:
                break

        f_file.close()

# -----------------------------------------------------------


parser = argparse.ArgumentParser(description='Detect and classify cars.')
parser.add_argument('-src-dir',
                    type=str,
                    default='./test_imgs',
                    help='source directory of images')
parser.add_argument('-dst-dir',
                    type=str,
                    default='./test_result',
                    help='destination directory of images to store results.')

if __name__ == '__main__':
    # ---------------------------- Car detect and classify
    args = parser.parse_args()
    # DR_model = Car_DC(src_dir=args.src_dir, dst_dir=args.dst_dir)
    DR_model = Car_DC('example/vdo.avi')    # vdo = string of Video address
    DR_model.detect_classify()
