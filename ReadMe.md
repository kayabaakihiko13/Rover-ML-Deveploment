# Rover ML Development


Rover ML Development merupakan bagian dari proyek __Rover__, yang berfokus pada deteksi tingkat kematangan buah kelapa sawit.

Proyek ini mengembangkan aplikasi mobile berbasis kecerdasaan bahan untuk membantu pengguna mengidentifikasi kematangan bah kelapa sawit secara cepat dan akurasi.

Proyek ini mengembangkan aplikasi mobile berbasis kecerdasan buatan untuk membantu penggunaan mengidenfikasi kematangan buah kelapa sawit secara cepat dan akurat.

## Get Started

Project ini menggunakan teknologi dan framework berikut:

* [Python 3.11](https://www.python.org/downloads/release/python-3116/)  
* [PyTorch 2.7.0 + CUDA](https://pytorch.org/get-started/previous-versions/)  
* [Ultralytics YOLO](https://docs.ultralytics.com/quickstart/)  

Training Model dan saving model
---
untuk mengedit atau menjalankan model yolo bisa akses ke folder `development/` lalu jalankan seperti berikut

```sh
# windows plaforms
python train.py
# linux platform
python3 train.py
```
atau juga bisa dijalankan di main root pada repository ini

```sh
# windows plaforms
python development/train.py
# linux platform
python3 development/train.py
```

untuk saving model ke dalam format file onnx dilakukan pada file `test.py`. bisa dijalankan pada berikut

```sh
# windows plaforms
python development/test.py
# linux platform
python3 development/test.py
```

## testing inference

Testing inference bertujuan untuk mengecek apakah model berjalan dengan ringan dan sesuai ekspektasi pada perangkat target.

Proses ini akan menampilkan metrik performa seperti kecepatan inferensi (ms per image), akurasi deteksi, serta validasi apakah model bisa digunakan untuk deployment.

Contoh menjalankan inference:
```sh
# Windows Platfrom
python -m test.inference
# Linux Platform
python3 -m test.inference
```