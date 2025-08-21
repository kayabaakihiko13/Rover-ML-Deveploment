import os
import random
import cv2
import time
import csv
import psutil
import multiprocessing
import onnxruntime as ort
from src.parserV11 import YOLODetector


class TestInference:
    """Test inference untuk mengukur performa model ONNX (latency, CPU, memory)."""
    def __init__(self, model_path: str, label_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        self.model_path = model_path
        self.label_path = label_path
        self.record_folder_path = "records"

        os.makedirs(self.record_folder_path, exist_ok=True)
        self.csv_path = os.path.join(self.record_folder_path, "inference_results.csv")

        self.setUp()
        self._init_csv()

    def setUp(self):
        """Set up 2 versi detector: optimized & unoptimized"""
        self.yolov11_onnx_optimize = YOLODetector(
            self.model_path,
            label_yaml=self.label_path,
            optimize=True,
        )
        self.yolov11_onnx_unoptimize = YOLODetector(
            self.model_path,
            label_yaml=self.label_path,
            optimize=False
        )

    def _init_csv(self):
        """Buat file CSV dengan header kalau belum ada"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image", "mode", "batch_size",
                    "latency_ms", "cpu_percent", "memory_mb"
                ])

    def _record_result(self, image_path, mode, batch_size, latency, cpu, memory):
        """Catat hasil ke CSV"""
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                os.path.basename(image_path),
                mode, batch_size,
                f"{latency:.2f}", f"{cpu:.2f}", f"{memory:.2f}"
            ])

    def _run_inference(self, detector, image_path: str, mode: str, warmup: int = 3):
        """Jalankan inference + monitor resource"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

        # warmup run (biar inference lebih stabil)
        for _ in range(warmup):
            detector.detect(image)

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        process.cpu_percent()  # reset counter
        start = time.perf_counter()
        _, _, _ = detector.detect(image)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        cpu_usage = process.cpu_percent()
        mem_after = process.memory_info().rss / (1024 * 1024)

        mem_usage = mem_after - mem_before

        print(f"[{mode}] Latency: {latency_ms:.2f} ms | CPU: {cpu_usage:.2f}% | Mem Change: {mem_usage:.2f} MB")
        self._record_result(image_path, mode, 1, latency_ms, cpu_usage, mem_usage)

        return latency_ms, cpu_usage, mem_usage

    def test_with_optimize(self, image_path: str):
        return self._run_inference(self.yolov11_onnx_optimize, image_path, "optimized")

    def test_with_unoptimize(self, image_path: str):
        return self._run_inference(self.yolov11_onnx_unoptimize, image_path, "unoptimized")

    def run_batch(self, sample_images: list, mode: str, detector):
        """Jalankan inference batch untuk optimize atau unoptimize."""
        total_time, total_cpu, total_mem = 0.0, 0.0, 0.0

        for img in sample_images:
            dur, cpu, mem = self._run_inference(detector, img, mode)
            total_time += dur
            total_cpu += cpu
            total_mem += mem

        avg_time = total_time / len(sample_images)
        avg_cpu = total_cpu / len(sample_images)
        avg_mem = total_mem / len(sample_images)
        fps = 1000 / avg_time if avg_time > 0 else 0

        print(f"[{mode}] {len(sample_images)} images → "
              f"Latency: {avg_time:.2f} ms | CPU: {avg_cpu:.2f}% | Mem: {avg_mem:.2f} MB | {fps:.2f} FPS")

        self._record_result(f"batch_{len(sample_images)}", mode, len(sample_images), avg_time, avg_cpu, avg_mem)


if __name__ == "__main__":
    tester = TestInference("model/best.onnx", "data/data.yaml")
    image_dir = r"images"

    print("Available CPU cores:", multiprocessing.cpu_count())

    all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not all_images:
        raise FileNotFoundError(f"Tidak menemukan gambar di folder {image_dir}")

    random.seed(42)  # biar reproducible

    for num in [1, 10, 25]:
        if len(all_images) < num:
            print(f"[WARNING] Jumlah gambar kurang dari {num}, pakai {len(all_images)} saja.")
        sample_images = random.sample(all_images, min(num, len(all_images)))

        # optimized
        tester.run_batch(sample_images, "optimized", tester.yolov11_onnx_optimize)
        # unoptimized
        tester.run_batch(sample_images, "unoptimized", tester.yolov11_onnx_unoptimize)
