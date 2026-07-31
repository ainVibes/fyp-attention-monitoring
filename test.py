import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe
from ultralytics import YOLO

print("OpenCV version:", cv2.__version__)
print("MediaPipe imported successfully")
print("YOLOv8 imported successfully")
print("All libraries ready!")