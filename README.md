# Vehicle_Detection_with_YOLOv3_tiny
vehicle detection with yolo v3 tiny
基于yolo v3 tiny的车辆识别

This is a simple vehicle detection algorithm with yolo v3 tiny.

How to run:

1.download weight from 
https://drive.google.com/open?id=1lD20AXtK_Sti68eXctQPdFo6O2OA9I0d

2.save your test video to directory "/example/vdo.mp4"

3.run VehicleDC.py


on default it takes a piece of video and overlay the detection on the output video and saves it in the same example folder, you can also tweak it a little so that it runs detection on images.


I adopt the code from CaptainEven’s Vehicle-Car-detection-and-multilabel-classification algorithm and removed the function of vehicle classification and did some changes, since I will only use the output from yolo v3 tiny as an input of object tracking using SORT in my later work. You can also check CaptainEven’s great work.
https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification
