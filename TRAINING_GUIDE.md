# Hướng dẫn Training — Từng bước một

Tài liệu này hướng dẫn bạn từ máy trắng (chưa có gì) đến có 2 model đã train xong, đánh giá xong, và demo chạy được. Đọc theo thứ tự, đừng nhảy bước.

---

## Bước 0 — Chọn nơi chạy: máy cá nhân hay Colab/Kaggle Notebook?

Training CNN trên ảnh **cần khá nhiều compute**. Máy cá nhân không có GPU vẫn train được nhưng chậm hơn nhiều (baseline CNN có thể mất 10-20 phút/epoch trên CPU thay vì vài chục giây trên GPU).

| Tiêu chí | Máy cá nhân (CPU) | Google Colab (miễn phí) | Kaggle Notebook (miễn phí) |
|---|---|---|---|
| GPU | Không (trừ khi bạn có GPU NVIDIA) | Có (T4), giới hạn theo phiên | Có (P100/T4), 30h/tuần |
| Tốc độ train | Chậm | Nhanh | Nhanh |
| Rủi ro | Không có | Session có thể bị ngắt sau vài giờ | Ổn định hơn Colab |
| Dataset | Tải thủ công | Tải qua Kaggle API hoặc mount Drive | Add dataset trực tiếp, không cần tải |
| Phù hợp khi | Dataset nhỏ, chỉ cần chạy vài epoch để test pipeline | Train thật, cần nhiều epoch | Train thật, tiện nhất vì dataset đã có sẵn trên Kaggle |

**Khuyến nghị:** Dùng **Kaggle Notebook** để train (dataset đã nằm sẵn trên Kaggle, chỉ cần "Add Data" chứ không cần tải), sau đó tải checkpoint `.keras` về máy để chạy Streamlit demo local.

Nếu bạn train trên Colab/Kaggle, đơn giản nhất là:
1. Upload toàn bộ project (trừ `data/raw/` và `checkpoints/`) lên notebook (hoặc `git clone` nếu đã push lên GitHub).
2. Cài `pip install -r requirements.txt` trong cell đầu.
3. Chạy các lệnh y hệt như hướng dẫn dưới đây (chúng đều là lệnh Python module, chạy được trong cell bằng `!python -m ...`).
4. Sau khi train xong, tải thư mục `checkpoints/` và `reports/` về máy cá nhân.

---

## Bước 1 — Cài đặt môi trường

```bash
cd vegetable-classifier

# Tạo virtual environment (khuyến nghị — tránh xung đột package với project khác)
python -m venv venv
source venv/bin/activate          # Windows (cmd): venv\Scripts\activate.bat

# Cài dependencies
pip install -r requirements.txt
```

**Kiểm tra cài đặt thành công:**

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit; print('Streamlit OK')"
```

Nếu bạn có GPU NVIDIA và muốn dùng, cài `tensorflow[and-cuda]` thay vì `tensorflow` thường (xem [hướng dẫn chính thức](https://www.tensorflow.org/install/pip)) — không bắt buộc, CPU vẫn chạy được, chỉ chậm hơn.

---

## Bước 2 — Tải dataset

### Cách A — Dùng Kaggle API (khuyến nghị cho máy cá nhân)

1. Đăng nhập [kaggle.com](https://www.kaggle.com) → góc phải trên → **Account** → **Create New API Token**. File `kaggle.json` sẽ được tải về.
2. Đặt file vào đúng vị trí:
   - Linux/Mac: `~/.kaggle/kaggle.json` (rồi chạy `chmod 600 ~/.kaggle/kaggle.json`)
   - Windows: `C:\Users\<tên_user>\.kaggle\kaggle.json`
3. Chạy script tải sẵn trong project:

```bash
python scripts/download_dataset.py
```

Script sẽ tải và giải nén vào `data/raw/`. Nếu cấu trúc thư mục sau khi giải nén không khớp với `train/validation/test`, script sẽ in cảnh báo — xem Bước 3 để kiểm tra và sửa.

### Cách B — Tải thủ công

1. Vào [trang dataset trên Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition).
2. Bấm **Download**.
3. Giải nén file zip vào thư mục `data/raw/` của project, sao cho có cấu trúc:

```
data/raw/
├── train/
│   ├── apple/
│   │   ├── Image_1.jpg
│   │   └── ...
│   ├── banana/
│   └── ... (36 class)
├── validation/
│   └── ... (cùng cấu trúc)
└── test/
    └── ... (cùng cấu trúc)
```

### Cách C — Kaggle Notebook (không cần tải)

Nếu bạn chạy trực tiếp trên Kaggle Notebook, bấm **Add Input** → tìm `fruit-and-vegetable-image-recognition` → Add. Dataset sẽ nằm ở `/kaggle/input/fruit-and-vegetable-image-recognition/`. Sửa lại 3 dòng `train_dir`, `val_dir`, `test_dir` trong `configs/config.py` để trỏ đúng đường dẫn này (thường có thêm 1 cấp thư mục lồng — kiểm tra bằng `!ls /kaggle/input/.../`).

---

## Bước 3 — Kiểm tra dataset đã đúng cấu trúc chưa

**Đừng bỏ qua bước này.** Chạy lệnh sau để kiểm tra:

```bash
python -m src.data.dataset
```

Kết quả mong đợi (số liệu thực tế có thể khác 1 chút tuỳ phiên bản dataset):

```
Số class: 36
Tổng số ảnh: ~5000
  train: ~3600 ảnh
  validation: ~700 ảnh
  test: ~700 ảnh
Batch shape: images=(32, 224, 224, 3), labels=(32, 36)
Giá trị pixel: min=0.000, max=1.000
```

Nếu bạn thấy lỗi `FileNotFoundError`, nghĩa là đường dẫn trong `configs/config.py` (mục `paths.train_dir`, `val_dir`, `test_dir`) không khớp với nơi bạn vừa giải nén dataset — mở file đó và sửa lại đường dẫn cho đúng.

---

## Bước 4 — Chạy EDA (Exploratory Data Analysis)

Mở notebook và chạy toàn bộ (Cell → Run All):

```bash
jupyter notebook notebooks/EDA.ipynb
```

Notebook sẽ cho bạn thấy: phân bố số ảnh mỗi class, có class imbalance không, ảnh mẫu từng class, phân bố kích thước ảnh, và phát hiện ảnh lỗi (nếu có). Các biểu đồ cũng được lưu tự động vào `reports/figures/` để dùng trong báo cáo/thuyết trình.

Ngoài notebook, bạn cũng có thể chạy nhanh bằng script (không cần Jupyter):

```bash
python -m src.data.eda
```

**Việc cần làm sau bước này:** đọc phần "Kết luận EDA" ở cuối notebook, điền số liệu thực tế bạn quan sát được — đây là nội dung bạn sẽ dùng để giải thích dataset khi thuyết trình.

---

## Bước 5 — Hiểu về Configuration (trước khi train)

Mở `configs/config.py`. Đây là nơi DUY NHẤT bạn cần sửa hyperparameter — không sửa hard-code ở file khác.

Các giá trị quan trọng nhất nếu bạn muốn thử nghiệm:

```python
"data": {
    "img_size": 224,       # kích thước ảnh input — để nguyên, khớp với MobileNetV2
    "batch_size": 32,       # giảm xuống 16 nếu máy yếu/hết RAM khi train
},
"baseline_cnn": {
    "epochs": 25,           # số epoch tối đa cho baseline (có EarlyStopping nên có thể dừng sớm hơn)
    "learning_rate": 1e-3,
},
"transfer_model": {
    "epochs_feature_extraction": 15,  # giai đoạn 1: train head, base bị freeze
    "epochs_fine_tune": 10,           # giai đoạn 2: mở khóa 1 phần base, train tiếp với LR rất nhỏ
},
```

Nếu máy bạn yếu hoặc chỉ muốn chạy thử nhanh để kiểm tra pipeline hoạt động, giảm `epochs` xuống 2-3 trước, chạy thử, rồi tăng lên khi train thật.

---

## Bước 6 — Training

### Train cả 2 model (khuyến nghị — đúng yêu cầu môn học là phải có ≥2 model để so sánh)

```bash
python -m src.training.train --model both
```

### Hoặc train riêng từng model

```bash
python -m src.training.train --model baseline    # chỉ CNN baseline
python -m src.training.train --model transfer     # chỉ MobileNetV2 transfer learning
```

**Những gì sẽ xảy ra khi train:**

- Log tiến trình từng epoch hiện trực tiếp trên terminal (loss, accuracy, val_loss, val_accuracy).
- `ModelCheckpoint` tự lưu model tốt nhất (theo `val_accuracy`) vào `checkpoints/baseline_cnn.keras` hoặc `checkpoints/mobilenetv2_transfer.keras`.
- `EarlyStopping` tự dừng nếu `val_loss` không cải thiện sau 5 epoch liên tiếp — tránh train thừa/overfitting thêm.
- `ReduceLROnPlateau` tự giảm learning rate nếu quá trình train chững lại.
- Lịch sử training (loss/accuracy mỗi epoch) được lưu vào `reports/logs/*_history.csv`.
- **Biểu đồ Training vs Validation (Accuracy & Loss)** được vẽ tự động và lưu vào `reports/figures/*_training_curves.png` — dùng trực tiếp cho báo cáo.
- **Class imbalance được xử lý tự động:** nếu EDA cho thấy dataset lệch (chênh lệch vượt 30% so với trung bình), hệ thống tự tính `class_weight` để class hiếm không bị model "bỏ rơi". Terminal sẽ in rõ có áp dụng hay không. Tắt tính năng này bằng `use_class_weight: False` trong config.
- Metadata đầy đủ (architecture, số parameters, training time, kết quả cuối...) được lưu vào `reports/*_metadata.json` — dùng để điền vào báo cáo.

Nếu muốn vẽ lại biểu đồ training curves mà không cần train lại (ví dụ đã có sẵn file CSV log):

```bash
python -m src.evaluation.plot_history
```

**Thời gian ước tính** (dataset ~3600 ảnh train, batch_size=32):

| Model | CPU (máy cá nhân) | GPU (Colab/Kaggle T4) |
|---|---|---|
| Baseline CNN | ~15-30 phút (10-15 epoch) | ~2-4 phút |
| MobileNetV2 transfer | ~40-70 phút (25 epoch, cả 2 giai đoạn) | ~5-10 phút |

**Nếu train trên Colab và bị ngắt session giữa chừng:** checkpoint tốt nhất tính đến thời điểm ngắt vẫn đã được lưu (nhờ `ModelCheckpoint`), nhưng training history của epoch cuối có thể mất. Nên mount Google Drive và trỏ `checkpoints_dir`, `logs_dir` trong config sang thư mục trên Drive để không mất dữ liệu khi session bị ngắt.

---

## Bước 7 — Đánh giá & so sánh model

```bash
python -m src.evaluation.evaluate --compare
```

Lệnh này sẽ:
1. Load cả 2 checkpoint (`baseline_cnn.keras`, `mobilenetv2_transfer.keras`).
2. Chạy inference trên **test set độc lập** (chưa từng dùng để train hay tune).
3. Tính Accuracy, Precision/Recall/F1 (macro & weighted), Confusion Matrix, Classification Report.
4. Phát hiện các cặp class hay bị nhầm lẫn nhất.
5. Vẽ + lưu: confusion matrix heatmap, ảnh dự đoán đúng, ảnh dự đoán sai (kèm confidence) → `reports/figures/`.
6. Xuất bảng so sánh 2 model → `reports/model_comparison.csv`.
7. Phân tích overfitting/underfitting dựa trên gap train vs val accuracy.
8. **Sinh báo cáo tổng hợp dạng Markdown** → `reports/evaluation_report.md`, gồm đầy đủ bảng số liệu, biểu đồ nhúng sẵn, và các mục _Nhận xét_ để bạn điền phần phân tích của mình.

Nếu muốn sinh lại riêng báo cáo Markdown (khi đã có sẵn kết quả evaluate):

```bash
python -m src.evaluation.report
```

Nếu chỉ muốn đánh giá 1 model:

```bash
python -m src.evaluation.evaluate --model checkpoints/mobilenetv2_transfer.keras --name mobilenetv2_transfer
```

**Việc cần làm sau bước này:** mở `reports/model_comparison.csv` và các file `reports/*_summary.json`, đọc phần `top_confused_pairs` (class nào hay bị nhầm) và `overfitting_analysis` — đây là nội dung cho phần "Model Comparison" và "Error Analysis" trong báo cáo. Dựa vào bảng so sánh, **chọn model tốt nhất làm final model** (không chỉ dựa vào Accuracy — cân nhắc cả F1, training time, khả năng tổng quát hoá).

---

## Bước 8 — Chạy inference thử với 1 ảnh mới

Dùng 1 ảnh **chưa từng xuất hiện trong dataset** (chụp bằng điện thoại, tải từ Google...) để kiểm tra model tổng quát hoá tốt đến đâu:

```bash
python -m src.inference.predict path/to/anh_moi.jpg --checkpoint checkpoints/mobilenetv2_transfer.keras
```

Output mẫu:

```
Prediction: Carrot
Confidence: 96.42%
Top-K predictions:
  Carrot: 96.42%
  Sweet Potato: 2.10%
  ...
```

---

## Bước 9 — Chạy demo application

```bash
streamlit run app/streamlit_app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`. App tự động ưu tiên dùng `checkpoints/mobilenetv2_transfer.keras` nếu có, nếu không sẽ fallback sang `checkpoints/baseline_cnn.keras`.

**Trước khi thuyết trình/nộp bài:** mở `app/streamlit_app.py`, kiểm tra biến `DEFAULT_CHECKPOINT_CANDIDATES` — đảm bảo model đầu tiên trong danh sách chính là **final model bạn đã chọn** sau bước so sánh (Bước 7), không phải model khác.

---

## Xử lý sự cố thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError: Không tìm thấy ảnh nào...` | Chưa tải dataset hoặc sai đường dẫn | Xem lại Bước 2-3 |
| `ResourceExhaustedError` / hết RAM khi train | `batch_size` quá lớn so với RAM/GPU memory | Giảm `batch_size` trong `configs/config.py` xuống 16 hoặc 8 |
| Training rất chậm | Đang chạy trên CPU | Dùng Colab/Kaggle Notebook (Bước 0), hoặc giảm `epochs` để test nhanh trước |
| `URL fetch failure` khi build MobileNetV2 | Không có kết nối internet để tải pretrained weights lần đầu | Cần internet ổn định lần đầu chạy (weights được cache lại sau đó, không tải lại lần 2) |
| Streamlit báo "Chưa tìm thấy model" | Chưa train xong hoặc checkpoint bị xoá | Chạy lại Bước 6 |
| `accuracy` = 0 hoặc `NaN` trong lúc train | Learning rate quá cao, hoặc data bị lỗi | Giảm `learning_rate` trong config; kiểm tra lại EDA xem có ảnh hỏng không |

---

## Tóm tắt lệnh (chép nhanh)

```bash
pip install -r requirements.txt
python scripts/download_dataset.py
python -m src.data.dataset                 # sanity check dataset
python -m src.data.eda                     # EDA (hoặc mở notebooks/EDA.ipynb)
python -m src.training.train --model both  # train cả 2 model (tự vẽ training curves)
python -m src.evaluation.evaluate --compare  # so sánh + sinh evaluation_report.md
streamlit run app/streamlit_app.py         # chạy demo
pytest tests/ -v                            # chạy test suite (27 test)
```

---

## Danh sách file kết quả sinh ra

Sau khi chạy đủ các bước trên, bạn sẽ có:

```
reports/
├── evaluation_report.md              # ⭐ Báo cáo tổng hợp — dùng trực tiếp cho bài nộp
├── model_comparison.csv              # Bảng so sánh 2 model
├── baseline_cnn_metadata.json        # Hyperparameters + kết quả training
├── baseline_cnn_summary.json         # Metrics test + class nhầm lẫn + phân tích overfitting
├── baseline_cnn_per_class_metrics.csv
├── mobilenetv2_transfer_*.json/csv   # (tương tự cho model 2)
├── logs/*_history.csv                # Loss/accuracy từng epoch
└── figures/
    ├── class_distribution_train.png
    ├── sample_images.png
    ├── image_size_distribution.png
    ├── *_training_curves.png         # Training vs Validation (Accuracy & Loss)
    ├── *_confusion_matrix.png
    ├── *_misclassified.png
    └── *_correct_predictions.png
```
