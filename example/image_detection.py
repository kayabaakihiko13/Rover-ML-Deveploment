import cv2
import os
import sys
import psutil
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from src.parserV11 import YOLODetector

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ===== Example usage =====
    model_path = "model/best.onnx"
    class_yaml = "data/data.yaml"
    image_path = "images/30677689379-sawit_2.jpg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image_arr = cv2.imread(image_path)

    # Init detector (model load time tidak dihitung latency)
    detector = YOLODetector(
        model_path, class_yaml, conf_thresh=0.25, iou_thresh=0.7, optimize=True
    )

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
