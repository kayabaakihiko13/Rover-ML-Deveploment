import cv2

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
from src.parserV11 import YOLODetector



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # ===== Example usage =====
    model_path = r'runs\detect\train2\weights\best.onnx'
    class_yaml = 'data\data.yaml'
    detector = YOLODetector(model_path, class_yaml, conf_thresh=0.5, iou_thresh=0.45)

    image_path = r'bb6ababa53f94b9922bd2e68b7e07f40.jpg'
    image = cv2.imread(image_path)

    boxes, scores, class_ids = detector.detect(image)
    
