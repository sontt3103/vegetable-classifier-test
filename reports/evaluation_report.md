# Báo cáo Đánh giá Model — Phân loại Rau Củ Quả

> File này được sinh tự động bởi `python -m src.evaluation.report`.
> Các phần _Nhận xét_ ở cuối cần bạn tự viết dựa trên số liệu bên trên.

---

## 1. Dữ liệu đánh giá

- Số class: **36**
- Danh sách class: apple, banana, beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalepeno, kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, pomegranate, potato, raddish, soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip, watermelon

![Phân bố class](figures/class_distribution_train.png)

---

## 2. Kết quả từng model

### Model 1 — CNN Baseline (tự xây)

**Cấu hình huấn luyện**

| Thông số | Giá trị |
|---|---|
| Architecture | Custom CNN (3x Conv2D-BN-MaxPool + GAP + Dense) |
| Số parameters | 136,420 |
| Input size | 224×224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | categorical_crossentropy |
| Số epoch đã chạy | 25 |
| Thời gian training | 7178.3 giây |
| Áp dụng class_weight | Có |

**Kết quả trên tập train/validation**

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 39.81% | 34.79% |
| Loss | 2.0044 | 2.1587 |

![Training curves](figures/baseline_cnn_training_curves.png)

**Kết quả trên tập test độc lập**

| Metric | Giá trị |
|---|---:|
| Accuracy | 40.55% |
| Precision (macro) | 0.4276 |
| Recall (macro) | 0.4005 |
| F1-score (macro) | 0.3845 |
| F1-score (weighted) | 0.3895 |
| Số ảnh test | 434 |
| Số ảnh dự đoán sai | 200 |

- **Class có kết quả tốt nhất:** soy beans
- **Class yếu nhất:** chilli pepper

**Phân tích overfitting/underfitting:** ⚠️ Underfitting

> Cả train (39.81%) và val accuracy (34.79%) đều thấp -> mô hình có dấu hiệu underfitting (chưa học được đặc trưng, có thể cần tăng epochs, giảm regularization, hoặc model chưa đủ phức tạp).

**Các cặp class bị nhầm lẫn nhiều nhất**

| Nhãn thật | Model dự đoán thành | Số lần |
|---|---|---:|
| corn | sweetcorn | 5 |
| spinach | peas | 5 |
| lettuce | spinach | 4 |
| mango | potato | 4 |
| onion | sweetpotato | 4 |

![Confusion matrix](figures/baseline_cnn_confusion_matrix.png)

![Dự đoán sai](figures/baseline_cnn_misclassified.png)

---

### Model 2 — MobileNetV2 (Transfer Learning)

**Cấu hình huấn luyện**

| Thông số | Giá trị |
|---|---|
| Architecture | MobileNetV2 (ImageNet pretrained) + GAP + Dense head |
| Số parameters | 2,595,172 |
| Input size | 224×224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | categorical_crossentropy |
| Số epoch đã chạy | 25 |
| Thời gian training | 3328.7 giây |
| Áp dụng class_weight | Có |

**Kết quả trên tập train/validation**

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 93.51% | 82.26% |
| Loss | 0.1904 | 0.6791 |

![Training curves](figures/mobilenetv2_transfer_training_curves.png)

**Kết quả trên tập test độc lập**

| Metric | Giá trị |
|---|---:|
| Accuracy | 80.88% |
| Precision (macro) | 0.8165 |
| Recall (macro) | 0.8022 |
| F1-score (macro) | 0.8019 |
| F1-score (weighted) | 0.8067 |
| Số ảnh test | 434 |
| Số ảnh dự đoán sai | 83 |

- **Class có kết quả tốt nhất:** banana
- **Class yếu nhất:** paprika

**Phân tích overfitting/underfitting:** ⚠️ Overfitting

> Train accuracy (93.51%) cao hơn val accuracy (82.26%) khoảng 11.25%, vượt ngưỡng 10% -> mô hình có dấu hiệu overfitting (học thuộc tập train, kém tổng quát hoá).

**Các cặp class bị nhầm lẫn nhiều nhất**

| Nhãn thật | Model dự đoán thành | Số lần |
|---|---|---:|
| corn | sweetcorn | 7 |
| bell pepper | capsicum | 3 |
| capsicum | bell pepper | 3 |
| lettuce | spinach | 3 |
| paprika | bell pepper | 3 |

![Confusion matrix](figures/mobilenetv2_transfer_confusion_matrix.png)

![Dự đoán sai](figures/mobilenetv2_transfer_misclassified.png)

---

### Model 3 — Classical SVM (HOG + PCA + SVM)

**Cấu hình huấn luyện**

| Thông số | Giá trị |
|---|---|
| Architecture | HOG(9 hướng, cell [8, 8]) -> PCA(50) + Color Hist RGB(32 bins, Hellinger x10) -> SVC RBF |
| Số parameters | 292,730 |
| Input size | 128×128 |
| Batch size | — |
| Optimizer | GridSearchCV (scoring = f1_macro) |
| Learning rate | — |
| Loss function | hinge (SVM) |
| Số epoch đã chạy | — |
| Thời gian training | 82.4 giây |
| Áp dụng class_weight | Không |

**Kết quả trên tập train/validation**

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 97.92% | 31.57% |
| Loss | — | — |

![GridSearchCV](figures/classical_svm_gridsearch.png)

**Kết quả trên tập test độc lập**

| Metric | Giá trị |
|---|---:|
| Accuracy | 34.10% |
| Precision (macro) | 0.3603 |
| Recall (macro) | 0.3382 |
| F1-score (macro) | 0.3372 |
| F1-score (weighted) | 0.3383 |
| Số ảnh test | 434 |
| Số ảnh dự đoán sai | 286 |

- **Class có kết quả tốt nhất:** carrot
- **Class yếu nhất:** capsicum

**Phân tích overfitting/underfitting:** ⚠️ Overfitting

> Train accuracy (97.92%) cao hơn val accuracy (31.57%) khoảng 66.35%, vượt ngưỡng 10% -> mô hình có dấu hiệu overfitting.

**Các cặp class bị nhầm lẫn nhiều nhất**

| Nhãn thật | Model dự đoán thành | Số lần |
|---|---|---:|
| apple | raddish | 4 |
| beetroot | turnip | 4 |
| soy beans | potato | 4 |
| capsicum | bell pepper | 3 |
| capsicum | spinach | 3 |

![Confusion matrix](figures/classical_svm_confusion_matrix.png)

![Dự đoán sai](figures/classical_svm_misclassified.png)

---

## 3. So sánh model

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | F1-score (weighted) | Training Time (s) | Num Parameters | Test Samples | Misclassified | Inference (ms/img) |
|---|---|---|---|---|---|---|---|---|---|---|
| classical_svm | 0.3410 | 0.3603 | 0.3382 | 0.3372 | 0.3383 | 82.4000 | 292730 | 434 | 286 | 5.6000 |
| baseline_cnn | 0.4055 | 0.4276 | 0.4005 | 0.3845 | 0.3895 | 7178.3000 | 136420 | 434 | 200 | nan |
| mobilenetv2_transfer | 0.8088 | 0.8165 | 0.8022 | 0.8019 | 0.8067 | 3328.7000 | 2595172 | 434 | 83 | nan |


**Quan sát tự động từ số liệu:**

- Accuracy cao nhất: **mobilenetv2_transfer** (80.88%)
- F1-score (macro) cao nhất: **mobilenetv2_transfer** (0.8019)
- Train nhanh nhất: **classical_svm** (82 giây)

> **Lưu ý khi chọn final model:** accuracy cao nhất KHÔNG đương nhiên là model tốt nhất.
> Nếu dataset mất cân bằng, F1-score (macro) phản ánh chất lượng công bằng hơn giữa các class.
> Cũng cần cân nhắc thời gian training, kích thước model, và tốc độ inference khi deploy.

---

## 4. Nhận xét & Kết luận _(bạn tự viết phần này)_

**4.1. Model nào tốt nhất và vì sao?**

_(dựa vào bảng so sánh mục 3, giải thích lựa chọn final model)_

**4.2. Vì sao một số class dễ bị nhầm lẫn?**

_(xem bảng cặp class nhầm lẫn + ảnh dự đoán sai ở mục 2 — ví dụ: hình dạng/màu sắc tương tự nhau)_

**4.3. Model có overfitting/underfitting không?**

_(xem phân tích tự động ở mục 2 và biểu đồ training curves)_

**4.4. Hạn chế của model**

_(ví dụ: dataset nhỏ, ảnh nền đơn giản không giống thực tế, một số class rất giống nhau...)_

**4.5. Hướng phát triển tiếp theo**

_(ví dụ: thu thập thêm ảnh thực tế, augmentation mạnh hơn, thử EfficientNet, deploy lên mobile bằng TensorFlow Lite...)_
