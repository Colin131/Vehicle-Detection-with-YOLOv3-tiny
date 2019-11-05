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

    def cls_draw_bbox(self, output, orig_img):
        """
        draw bbox to orig_img
        """
        labels = []
        pt_1s = []
        pt_2s = []

        for det in output:
            # rectangle points
            pt_1 = tuple(det[1:3].int())  # the left-up point
            pt_2 = tuple(det[3:5].int())  # the right down point
            pt_1s.append(pt_1)
            pt_2s.append(pt_2)

        color = (0, 255, 0)
        for i, det in enumerate(output):
            pt_1 = pt_1s[i]
            pt_2 = pt_2s[i]

            # draw bounding box
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=2)

    def detect_classify(self):
        """
        detect ONLY # and NO classify
        """

        clip1 = cv2.VideoCapture(self.vdo)
        width = int(clip1.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频的宽度
        height = int(clip1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频的高度
        fps = clip1.get(cv2.CAP_PROP_FPS)  # 视频的帧率
        fourcc = int(clip1.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
        framenum = int(clip1.get(7))  # 视频总帧数

        writer = cv2.VideoWriter('example/project_output.mp4', fourcc, fps, (width, height))
        frame_id = 0

        while clip1.isOpened():
            if frame_id < framenum - 2:
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

                self.cls_draw_bbox(output, frame)   # 绘制bounding box
                writer.write(frame)

                # cv2.imshow('det', orig_img)
                # cv2.waitKey()

                frame_id += 1
            else:
                break

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
    DR_model = Car_DC('example/vdo.mp4')    # vdo = string of Video address
    DR_model.detect_classify()
