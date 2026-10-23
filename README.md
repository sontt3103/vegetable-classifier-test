# 🥕 Vegetable & Fruit Image Classification

Đồ án môn **Machine Learning** — xây dựng pipeline phân loại rau củ quả từ ảnh với 3 phương pháp khác nhau để so sánh.

**Dataset**: [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) — 36 class, ~3.600 ảnh.

```
Ảnh đầu vào → Tiền xử lý → Model → Dự đoán → Confidence Score → Kết quả
```

---

## Mục lục

1. [3 Model được so sánh](#3-model-được-so-sánh)
2. [Cấu trúc project](#cấu-trúc-project)
3. [Cài đặt](#cài-đặt)
4. [Dataset](#dataset)
5. [Chạy từ đầu (Quickstart)](#chạy-từ-đầu-quickstart)
6. [EDA — Phân tích dữ liệu](#eda--phân-tích-dữ-liệu)
7. [Huấn luyện model](#huấn-luyện-model)
8. [Đánh giá & So sánh](#đánh-giá--so-sánh)
9. [Inference](#inference)
10. [Demo Streamlit](#demo-streamlit)
11. [Kết quả](#kết-quả)
12. [Hạn chế & Hướng phát triển](#hạn-chế--hướng-phát-triển)

---

## 3 Model được so sánh

| Model | Phương pháp | File |
|-------|------------|------|
| **Classical SVM** | HOG + Color Histogram → PCA → SVM (RBF kernel + GridSearchCV) | `src/models/classical_svm.py` |
| **Baseline CNN** | CNN từ đầu: 3×(Conv+ReLU+MaxPool) → Dense → Softmax | `src/models/baseline_cnn.py` |
| **MobileNetV2** | Transfer learning từ ImageNet, fine-tune lớp cuối | `src/models/transfer_model.py` |

**Mục tiêu**: So sánh phương pháp ML cổ điển (SVM) với Deep Learning (CNN) và Transfer Learning (MobileNetV2) để rút ra kết luận có cơ sở thực nghiệm.

---

## Cấu trúc project

```
vegetable-classifier/
├── configs/
│   └── config.py               # Tất cả hyperparameter & đường dẫn (nguồn duy nhất)
│
├── src/
│   ├── features/               # Rút trích đặc trưng (cho SVM)
│   │   ├── __init__.py
│   │   ├── hog_features.py     # HOG + Color Histogram
│   │   └── pca_wrapper.py      # Giảm chiều với PCA
│   ├── models/
│   │   ├── baseline_cnn.py     # CNN đơn giản
│   │   ├── transfer_model.py   # MobileNetV2 transfer learning
│   │   └── classical_svm.py    # SVM với GridSearchCV
│   ├── data/
│   │   ├── dataset.py          # Load train/val/test splits
│   │   ├── augmentation.py     # Data augmentation (cho CNN/MobileNetV2)
│   │   └── eda.py              # Phân tích & trực quan hóa dataset
│   ├── training/
│   │   ├── train.py            # Training CNN / MobileNetV2
│   │   └── tune.py             # Hyperparameter grid search
│   ├── evaluation/
│   │   ├── evaluate.py         # Metrics: Accuracy, F1, Confusion Matrix
│   │   ├── compare_models.py   # So sánh 3 model, sinh bảng kết quả
│   │   ├── plot_history.py     # Vẽ training/validation curves
│   │   ├── confusion_matrix.py
│   │   ├── error_analysis.py
│   │   └── report.py
│   └── inference/
│       └── predict.py          # Inference dùng chung cho cả 3 model
│
├── app/
│   └── streamlit_app.py        # Demo UI — upload ảnh, chọn model, xem kết quả
│
├── notebooks/
│   └── EDA.ipynb               # Phân tích dữ liệu tương tác
│
├── scripts/
│   ├── download_dataset.py     # Tải dataset từ Kaggle API
│   └── make_synthetic_dataset.py  # Tạo dataset giả để test
│
├── checkpoints/                # Model đã train (*.keras, *.pkl)
├── reports/                    # Kết quả đánh giá, biểu đồ, báo cáo
├── tests/                      # Pytest test suite
│
├── requirements.txt
├── .gitignore
├── TRAINING_GUIDE.md
├── ACADEMIC_PRESENTATION.md
└── README.md
```

---

## Cài đặt

Yêu cầu **Python 3.10+**.

```bash
git clone https://github.com/YOUR_USERNAME/vegetable-classifier.git
cd vegetable-classifier
pip install -r requirements.txt
```

> Nếu môi trường báo lỗi "externally managed": dùng virtualenv:
> ```bash
> python -m venv .venv
> .venv\Scripts\activate      # Windows
> source .venv/bin/activate   # Mac/Linux
> pip install -r requirements.txt
> ```

---

## Dataset

**Nguồn**: [Kaggle — Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

- 36 class (rau củ quả): apple, banana, carrot, tomato, ...
- ~100 ảnh/class cho tập train; có sẵn validation và test riêng
- Cấu trúc: `train/<class>/`, `validation/<class>/`, `test/<class>/`

**Tải dataset**:

```bash
# 1. Lấy API token tại https://www.kaggle.com/settings → "Create New API Token"
#    Đặt file kaggle.json vào ~/.kaggle/
# 2. Chạy:
python scripts/download_dataset.py --output-dir data/raw
```

Dataset sẽ được giải nén vào `data/raw/`.

---

## Chạy từ đầu (Quickstart)

### Model SVM (HOG + PCA + SVM)

```bash
# Rút trích đặc trưng → train SVM (chạy trên CPU, không cần GPU)
python -m src.models.classical_svm --data-dir data/raw
```

Kết quả lưu tại:
- `checkpoints/svm_model.pkl` — SVM model
- `checkpoints/pca_model.pkl` — PCA model
- `reports/svm_evaluation.json` — metrics

### Model CNN Baseline

```bash
python -m src.training.train --model baseline --data-dir data/raw --epochs 30
python -m src.evaluation.evaluate --checkpoint checkpoints/baseline_cnn_best.keras --data-dir data/raw
```

### Model MobileNetV2

```bash
python -m src.training.train --model transfer --data-dir data/raw --epochs 30
python -m src.evaluation.evaluate --checkpoint checkpoints/mobilenetv2_transfer_best.keras --data-dir data/raw
```

### So sánh 3 model

```bash
python -m src.evaluation.compare_models \
    --checkpoints checkpoints/svm_model.pkl checkpoints/baseline_cnn_best.keras checkpoints/mobilenetv2_transfer_best.keras \
    --data-dir data/raw
```

### Demo Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## EDA — Phân tích dữ liệu

```bash
python -m src.data.eda --data-dir data/raw --split train
```

Hoặc mở notebook tương tác:

```bash
jupyter notebook notebooks/EDA.ipynb
```

Kết quả lưu vào `reports/figures/`: biểu đồ phân bố class, ảnh mẫu, phân bố kích thước ảnh, HOG visualization.

---

## Huấn luyện model

Tất cả hyperparameter quan trọng được quản lý tập trung trong [`configs/config.py`](configs/config.py):

| Tham số | CNN/MobileNetV2 | SVM |
|---------|----------------|-----|
| Input size | `IMG_SIZE = (128, 128)` | `IMG_SIZE = (128, 128)` |
| Batch size | `BATCH_SIZE = 32` | N/A |
| Epochs | `EPOCHS = 30` | N/A |
| Learning rate | `LEARNING_RATE = 1e-3` | N/A |
| C values | N/A | `SVM_C_LIST = [0.1, 1, 10, 100]` |
| Gamma values | N/A | `SVM_GAMMA_LIST = ['scale', 'auto']` |
| PCA components | N/A | `PCA_N_COMPONENTS = 200` |

**Lưu ý**: CNN và MobileNetV2 cần GPU để train nhanh. SVM chạy được trên CPU (mất ~1–3 giờ với GridSearchCV 5-fold).

---

## Đánh giá & So sánh

Metric đánh giá mỗi model:

- **Accuracy** — % dự đoán đúng
- **Precision / Recall / F1-score** — theo từng class và macro/weighted
- **Confusion Matrix** — phát hiện class hay bị nhầm lẫn
- **Classification Report** — chi tiết từng trong 36 class

```bash
# Đánh giá 1 model
python -m src.evaluation.evaluate --checkpoint <path_to_checkpoint> --data-dir data/raw

# So sánh tất cả model
python -m src.evaluation.compare_models --checkpoints <ck1> <ck2> <ck3> --data-dir data/raw
```

Kết quả lưu tại `reports/model_comparison.md`.

---

## Inference

```bash
python -m src.inference.predict \
    --image path/to/image.jpg \
    --checkpoint checkpoints/baseline_cnn_best.keras
```

Output mẫu:

```
Prediction  : Carrot
Confidence  : 96.42%

Top-3 predictions:
  1. Carrot        96.42%
  2. Sweet potato   2.11%
  3. Beetroot       0.83%
```

---

## Demo Streamlit

```bash
streamlit run app/streamlit_app.py
```

Tính năng:
- Upload ảnh từ máy tính
- Chọn model: SVM / CNN / MobileNetV2
- Xem predicted class + confidence score
- Xem Top-K predictions
- Xem thông tin model
- Xử lý lỗi khi ảnh không hợp lệ

---

## Kết quả

> ⏳ **Đang cập nhật** — SVM đang trong quá trình training. CNN và MobileNetV2 đã train xong.

Bảng so sánh sẽ được cập nhật sau khi SVM training hoàn tất:

| Model | Accuracy | Precision | Recall | F1-score | Thời gian train |
|-------|:--------:|:---------:|:------:|:--------:|:--------------:|
| HOG + PCA + SVM | — | — | — | — | — |
| CNN Baseline | — | — | — | — | ~2h (GPU) |
| MobileNetV2 | — | — | — | — | ~1h (GPU) |

Các file kết quả hiện có trong `reports/`:
- `reports/baseline_cnn_evaluation.json`
- `reports/mobilenetv2_transfer_evaluation.json`
- `reports/figures/` — confusion matrix, training curves

---

## Hạn chế & Hướng phát triển

**Hạn chế hiện tại:**
- SVM không scale tốt khi dataset lớn (HOG extraction tốn RAM)
- CNN baseline có thể overfit do dataset nhỏ (~100 ảnh/class)
- Chưa thử ResNet, EfficientNet để có thêm điểm so sánh

**Hướng phát triển:**
- Thêm architecture (EfficientNet, ResNet50) vào bảng so sánh
- Data augmentation mạnh hơn để giảm overfitting
- Triển khai REST API thay vì chỉ Streamlit
- Fine-tune MobileNetV2 sâu hơn (unfreeze nhiều layer hơn)

---

## Phân công nhóm

| Thành viên | Nhiệm vụ chính |
|-----------|---------------|
| TN 1 | `classical_svm.py` · `configs/config.py` · Báo cáo Word |
| TN 2 | `hog_features.py` · `pca_wrapper.py` · EDA notebook |
| TN 3 | Chạy SVM training · Bảng so sánh 3 model · PPT thuyết trình |
| TN 4 | `streamlit_app.py` · Demo · Q&A kỹ thuật |

---

## Tài liệu thêm

- [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) — hướng dẫn train trên máy local và Colab
- [`ACADEMIC_PRESENTATION.md`](ACADEMIC_PRESENTATION.md) — câu hỏi & câu trả lời chuẩn bị cho buổi bảo vệ
