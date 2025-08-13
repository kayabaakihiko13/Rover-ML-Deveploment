import os
import random
import time
import yaml
import numpy as np
import cv2
from .utils import FileExistsNotFound, FormatFileError
from src.parserV11 import YOLODetector

class TestInference:
    """Test inference untuk mengukur kecepatan model ONNX."""
    def __init__(self, model_path: str, label_path: str):
        if not os.path.exists(model_path):
            raise FileExistsNotFound(f"Model file not found: {model_path}")
        self.model_path = model_path
        self.label_path = label_path
    
    def setUp(self):
        self.yolov11_onnx = YOLODetector(self.model_path, class_yaml=self.label_path)

    def test_one_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileExistsNotFound(f"Gambar tidak ditemukan atau rusak: {image_path}")
        
        start = time.time()
        boxes, scores, class_ids = self.yolov11_onnx.detect(image) # <--- pakai array, bukan path
        duration = time.time() - start
        return class_ids, duration

def main():
    model_path = r'runs\detect\train2\weights\best.onnx'
    label_path = r'data\data.yaml'
    image_dir = r'data\test\images'

    tester = TestInference(model_path, label_path)
    tester.setUp()

    all_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not all_images:
        print("Tidak ada gambar di folder test.")
        return

    for num in [1, 10, 100, 1000]:
        if len(all_images) < num:
            print(f"[WARNING] Jumlah gambar kurang dari {num}, pakai {len(all_images)} saja.")
        sample_images = random.sample(all_images, min(num, len(all_images)))
        
        total_time = 0.0
        for img_path in sample_images:
            _, duration = tester.test_one_image(img_path)
            total_time += duration

        avg_time = total_time / len(sample_images)
        print(f"\n=== Test {len(sample_images)} gambar ===")
        print(f"Total waktu   : {total_time:.4f} detik")
        print(f"Rata-rata/gambar: {avg_time:.4f} detik")

if __name__ == "__main__":
    main()
