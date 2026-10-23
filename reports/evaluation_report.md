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
| Thời gian training | 2731.7 giây |
| Áp dụng class_weight | Có |

**Kết quả trên tập train/validation**

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 43.82% | 51.37% |
| Loss | 1.8371 | 1.4894 |

![Training curves](figures/baseline_cnn_training_curves.png)

**Kết quả trên tập test độc lập**

| Metric | Giá trị |
|---|---:|
| Accuracy | 52.57% |
| Precision (macro) | 0.5901 |
| Recall (macro) | 0.5490 |
| F1-score (macro) | 0.5218 |
| F1-score (weighted) | 0.5075 |
| Số ảnh test | 409 |
| Số ảnh dự đoán sai | 194 |

- **Class có kết quả tốt nhất:** grapes
- **Class yếu nhất:** corn

**Phân tích overfitting/underfitting:** ✅ Tổng quát hoá tốt

> Gap giữa train (43.82%) và val accuracy (51.37%) chỉ -7.55%, trong ngưỡng chấp nhận được -> mô hình tổng quát hoá tốt.

**Các cặp class bị nhầm lẫn nhiều nhất**

| Nhãn thật | Model dự đoán thành | Số lần |
|---|---|---:|
| tomato | apple | 11 |
| banana | apple | 10 |
| carrot | apple | 10 |
| cucumber | apple | 10 |
| corn | sweetcorn | 9 |

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
| Thời gian training | 1916.3 giây |
| Áp dụng class_weight | Có |

**Kết quả trên tập train/validation**

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 91.88% | 85.79% |
| Loss | 0.2339 | 0.3542 |

![Training curves](figures/mobilenetv2_transfer_training_curves.png)

**Kết quả trên tập test độc lập**

| Metric | Giá trị |
|---|---:|
| Accuracy | 86.31% |
| Precision (macro) | 0.9470 |
| Recall (macro) | 0.9054 |
| F1-score (macro) | 0.9055 |
| F1-score (weighted) | 0.8688 |
| Số ảnh test | 409 |
| Số ảnh dự đoán sai | 56 |

- **Class có kết quả tốt nhất:** watermelon
- **Class yếu nhất:** banana

**Phân tích overfitting/underfitting:** ✅ Tổng quát hoá tốt

> Gap giữa train (91.88%) và val accuracy (85.79%) chỉ 6.09%, trong ngưỡng chấp nhận được -> mô hình tổng quát hoá tốt.

**Các cặp class bị nhầm lẫn nhiều nhất**

| Nhãn thật | Model dự đoán thành | Số lần |
|---|---|---:|
| apple | banana | 10 |
| carrot | banana | 10 |
| cucumber | banana | 10 |
| tomato | banana | 10 |
| corn | sweetcorn | 7 |

![Confusion matrix](figures/mobilenetv2_transfer_confusion_matrix.png)

![Dự đoán sai](figures/mobilenetv2_transfer_misclassified.png)

---

## 3. So sánh model

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | Training Time (s) | Num Parameters |
|---|---|---|---|---|---|---|
| mobilenetv2_transfer | 0.8631 | 0.9470 | 0.9054 | 0.9055 | 1916.3000 | 2595172 |

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
