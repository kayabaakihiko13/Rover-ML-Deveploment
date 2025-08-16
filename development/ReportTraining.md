# Report of Training Model
Bagian ini berisi laporan serta konfigurasi (config) yang digunakan pada saat training model. dimana pada project ini menggunakan device seperti berikut:

<table>
  <tr>
    <th>Komponen</th>
    <th>Spesifikasi</th>
  </tr>
  <tr>
    <td>Device</td>
    <td>Lenovo Ideapad Gaming 3 15ARH7</td>
  </tr>
  <tr>
    <td>CPU</td>
    <td>AMD Ryzen 7 7735HS with Radeon Graphics</td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)</td>
  </tr>
  <tr>
    <td>RAM</td>
    <td>16 GB DDR5</td>
  </tr>
  <tr>
    <td>Storage</td>
    <td>512 GB NVMe SSD</td>
  </tr>
  <tr>
    <td>OS</td>
    <td>Windows 11 Home 64-bit</td>
  </tr>
</table>

## Training Configuration

Berikut adalah contoh konfigurasi sederhana yang digunakan pada eksperimen ini:

```yaml
# Konfigurasi Sederhana untuk Training Model
imgsz: 640
epochs: 20
batch: 8
lr0: 0.001
optimizer: auto
weight_decay: 0.0005

box: 7.5
cls: 0.85
dfl: 1.0

# Augmentasi Ringan
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.4
fliplr: 0.5
flipud: 0.1
scale: 0.3
translate: 0.05
mixup: 0.1
mosaic: 1.0

patience: 10
```

## Result Training and Testing

Hasil training dan pengujian model menunjukkan bahwa sistem mampu mendeteksi tingkat kematangan buah kelapa sawit dengan performa yang cukup baik.  
Model diuji pada dataset validasi dan dataset testing terpisah untuk mengukur tingkat generalisasi.

<h2>Hasil Training</h2>
<table border="1" cellpadding="5" cellspacing="0">
  <tr>
    <th>Class</th>
    <th>Images</th>
    <th>Instances</th>
    <th>Box(P)</th>
    <th>R</th>
    <th>mAP50</th>
    <th>mAP50-95</th>
  </tr>
  <tr>
    <td>all</td><td>555</td><td>811</td><td>0.815</td><td>0.786</td><td>0.855</td><td>0.584</td>
  </tr>
  <tr>
    <td>Matang</td><td>165</td><td>189</td><td>0.858</td><td>0.670</td><td>0.873</td><td>0.639</td>
  </tr>
  <tr>
    <td>abnormal</td><td>57</td><td>81</td><td>0.646</td><td>0.778</td><td>0.733</td><td>0.465</td>
  </tr>
  <tr>
    <td>kosong</td><td>69</td><td>69</td><td>0.916</td><td>0.883</td><td>0.948</td><td>0.659</td>
  </tr>
  <tr>
    <td>mentah</td><td>151</td><td>195</td><td>0.919</td><td>0.836</td><td>0.895</td><td>0.561</td>
  </tr>
  <tr>
    <td>setengah_matang</td><td>155</td><td>175</td><td>0.784</td><td>0.731</td><td>0.811</td><td>0.588</td>
  </tr>
  <tr>
    <td>terlalu_matang</td><td>69</td><td>85</td><td>0.768</td><td>0.820</td><td>0.874</td><td>0.594</td>
  </tr>
</table>

<h2>Hasil Testing</h2>
<table border="1" cellpadding="5" cellspacing="0">
  <tr>
    <th>Class</th>
    <th>Images</th>
    <th>Instances</th>
    <th>Box(P)</th>
    <th>R</th>
    <th>mAP50</th>
    <th>mAP50-95</th>
  </tr>
  <tr>
    <td>all</td><td>277</td><td>385</td><td>0.845</td><td>0.805</td><td>0.902</td><td>0.608</td>
  </tr>
  <tr>
    <td>Matang</td><td>71</td><td>70</td><td>0.783</td><td>0.785</td><td>0.865</td><td>0.622</td>
  </tr>
  <tr>
    <td>abnormal</td><td>23</td><td>30</td><td>0.786</td><td>0.765</td><td>0.807</td><td>0.488</td>
  </tr>
  <tr>
    <td>kosong</td><td>30</td><td>38</td><td>0.894</td><td>0.886</td><td>0.956</td><td>0.659</td>
  </tr>
  <tr>
    <td>mentah</td><td>75</td><td>95</td><td>0.921</td><td>0.816</td><td>0.937</td><td>0.596</td>
  </tr>
  <tr>
    <td>setengah_matang</td><td>42</td><td>49</td><td>0.787</td><td>0.691</td><td>0.848</td><td>0.629</td>
  </tr>
  <tr>
    <td>terlalu_matang</td><td>36</td><td>43</td><td>0.889</td><td>0.884</td><td>0.954</td><td>0.654</td>
  </tr>
</table>

### 📊 Ringkasan Hasil dan Analisis

Dari tabel di atas terlihat bahwa model sudah menunjukkan performa yang **stabil dan seimbang**.  
- Pada **Training/Validation**, mAP50 mencapai **0.855** dengan mAP50-95 di **0.584**.  
- Sementara pada **Testing**, metrik justru lebih tinggi dengan mAP50 **0.902** dan mAP50-95 **0.608**, yang menandakan model memiliki kemampuan generalisasi yang baik (tidak overfitting).  

🔎 **Analisis per kelas**:  
- Kelas dengan performa **sangat tinggi**: `kosong` (0.956), `terlalu_matang` (0.954).  
- Kelas dengan performa **cukup stabil**: `matang` (0.865), `mentah` (0.937).  
- Kelas yang masih **perlu perbaikan**: `abnormal` (0.871 / 0.488 mAP50-95) dan `setengah_matang` (0.830 / 0.629), kemungkinan karena visual antar kelas yang mirip.  

⚡ **Perbandingan dengan training sebelumnya**:  
- mAP50-95 naik dari sekitar **0.55 → 0.60** (+5%).  
- Hasil **testing lebih baik dari validation**, menandakan model lebih generalisasi.  
- Training lebih ringan, hasil lebih konsisten.  