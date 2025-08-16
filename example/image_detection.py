import time
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
    model_path = r'runs\detect\train\weights\best.onnx'
    class_yaml = 'data\data.yaml'
    image_path = r'bb6ababa53f94b9922bd2e68b7e07f40.jpg'

    # check file image is exits
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # load image
    image_arr = cv2.imread(image_path)

    detector = YOLODetector(model_path,class_yaml,conf_thresh=.5,iou_thresh=.45)

    # detect
    start_time = time.time()
    boxes,scores,class_ids = detector.detect(image_arr)
    finish_time = (time.time() - start_time ) * 1000
    print(f"inference image is:{finish_time}in ms")
        # Draw results
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{detector.CLASSES[class_id]} {score *100:.2f}%"
        cv2.rectangle(image_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_arr, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)

    # Show with Matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

    # Save output
    cv2.imwrite('output_detected.jpg', image_arr)
    print("Detection saved as output_detected.jpg")