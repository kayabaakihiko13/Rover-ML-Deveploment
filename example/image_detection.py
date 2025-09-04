import cv2
import os
import sys
import psutil
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from src.parserV11 import YOLODetector

if __name__ == "__main__":
    # ===== Example usage =====
    model_path = "model/best.onnx"
    class_yaml = "data/data.yaml"
    image_path = "IMG20250712084222.jpeg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image_arr = cv2.imread(image_path)

    # Init detector
    detector = YOLODetector(
        model_path, class_yaml, conf_thresh=0.25, iou_thresh=0.65, optimize=True
    )

    # Generate distinct colors for each class
    num_classes = len(detector.CLASSES)
    np.random.seed(42)  # biar warnanya konsisten setiap run
    COLORS = (np.random.rand(num_classes, 3) * 255).astype(int)

    # Process monitoring
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    process.cpu_percent()  # reset counter

    # Measure inference latency only
    start = time.perf_counter()
    boxes, scores, class_ids = detector.detect(image_arr)
    end = time.perf_counter()

    # After inference
    cpu_usage = process.cpu_percent(interval=0.1)
    mem_after = process.memory_info().rss / (1024 * 1024)

    # Metrics
    latency_ms = (end - start) * 1000
    mem_change = mem_after - mem_before

    print(f"Latency (inference only): {latency_ms:.2f} ms")
    print(f"CPU usage during inference: {cpu_usage:.2f}%")
    print(f"Memory before: {mem_before:.2f} MB, after: {mem_after:.2f} MB, change: {mem_change:.2f} MB")

    # ===== Draw results on image =====
    for (box, score, class_id) in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{detector.CLASSES[class_id]}: {score:.2f}"

        color = (int(COLORS[class_id][0]),
                 int(COLORS[class_id][1]),
                 int(COLORS[class_id][2]))

        # Draw rectangle
        cv2.rectangle(image_arr, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image_arr, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)

        # Put label text
        cv2.putText(image_arr, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Detection Result", image_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optional: save
    cv2.imwrite("output.jpg", image_arr)
