# Hướng dẫn train và so sánh 3 model (TN 3)

Mục tiêu: train lại cả 3 model trên **cùng một dataset sạch** rồi so sánh trên **cùng một tập test**, đúng như bản Word phân công.

| # | Model | Lệnh train | Thời gian (CPU) |
|---|---|---|---|
| 1 | HOG → PCA → SVM | `python -m src.training.train_svm` | 5–20 phút |
| 2 | CNN Baseline | `python -m src.training.train --model baseline` | 30–50 phút |
| 3 | MobileNetV2 | `python -m src.training.train --model transfer` | 30–60 phút |

> **Vì sao phải train lại CNN và MobileNetV2?** File Word ghi 2 model này "đã xong, không cần train lại". Tuy nhiên kết quả cũ trong `reports/` đã bị **trộn 150 ảnh nhiễu ngẫu nhiên** do `create_test_dataset.py` tạo ra, mỗi split 10 ảnh cho 5 class apple, banana, carrot, cucumber, tomato. Đó là lý do:
> - test có 409 ảnh thay vì 359;
> - 5 class trên có 20 ảnh test, trong khi các class khác chỉ có 10;
> - mỗi class trong 5 class đó có đúng 10 ảnh bị đoán nhầm sang cùng một class.
>
> Số liệu cũ không dùng để so sánh được. **Tuyệt đối không chạy `create_test_dataset.py` vào `data/raw` nữa.**

Mỗi bước bên dưới có ghi **"Kết quả đúng"** để bạn tự kiểm tra. Bước nào không ra đúng thì dừng lại, xem mục **Xử lý lỗi** ở cuối file.

---

## BƯỚC 0 — Chuyển project ra đường dẫn không dấu (5 phút)

Hiện project nằm ở `D:\Học máy\vegetable-classifier-main\vegetable-classifier-main`. Đường dẫn này có **dấu tiếng Việt**, **dấu cách** và **2 cấp thư mục lồng nhau**. TensorFlow và OpenCV trên Windows hay lỗi với đường dẫn có dấu, và file Word cũng quy định thư mục chung là `D:\vegetable-classifier\`.

1. Mở File Explorer và vào `D:\Học máy\vegetable-classifier-main\`.
2. Kéo thư mục con `vegetable-classifier-main` (thư mục chứa `src`, `configs`, `app`...) ra thẳng ổ `D:\`.
3. Đổi tên thư mục đó thành `vegetable-classifier`.
4. Xoá file tên `2.17.0` trong project. Đây là log rác sinh ra khi gõ `pip install tensorflow>=2.17.0` mà không có dấu ngoặc kép.

**Kết quả đúng:** có file `D:\vegetable-classifier\README.md` và thư mục `D:\vegetable-classifier\src\`.

Từ đây trở đi, mọi lệnh đều chạy trong **Command Prompt (cmd)** tại thư mục project:

```bat
D:
cd D:\vegetable-classifier
```

> Mẹo: trong File Explorer, mở `D:\vegetable-classifier`, gõ `cmd` vào thanh địa chỉ rồi Enter. Cửa sổ cmd sẽ mở đúng thư mục.

---

## BƯỚC 1 — Cài môi trường Python (10–20 phút, chỉ làm 1 lần)

### 1.1 Kiểm tra phiên bản Python

```bat
python --version
```

| Kết quả | Việc cần làm |
|---|---|
| `Python 3.10.x` đến `3.13.x` | OK, sang bước 1.2 |
| `3.14` trở lên, hoặc báo "không tìm thấy python" | Cài Python 3.12 từ python.org. Lúc cài **tick ô "Add python.exe to PATH"**, rồi mở lại cmd |

### 1.2 Tạo môi trường ảo (venv)

Môi trường ảo giúp thư viện của project không đụng đến thư viện của các project khác.

```bat
python -m venv venv
venv\Scripts\activate
```

**Kết quả đúng:** đầu dòng lệnh có chữ `(venv)`, ví dụ `(venv) D:\vegetable-classifier>`.

> ⚠️ **Mỗi lần mở cmd mới đều phải chạy lại `venv\Scripts\activate`.** Nếu không có `(venv)` ở đầu dòng thì các lệnh sau sẽ báo thiếu thư viện.
>
> Nếu bạn dùng PowerShell thay vì cmd, lệnh kích hoạt là `venv\Scripts\Activate.ps1`. Nếu bị báo lỗi "running scripts is disabled", chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần, hoặc chuyển sang dùng cmd cho đơn giản.

### 1.3 Cài thư viện

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Bước này tải khoảng 500 MB (TensorFlow là phần nặng nhất), mất 5–15 phút.

### 1.4 Kiểm tra cài đặt

```bat
python -c "import tensorflow as tf, sklearn, skimage, cv2; print('TF', tf.__version__, '| sklearn', sklearn.__version__, '| OK')"
```

**Kết quả đúng:** có dòng kết thúc bằng `| OK`. Các dòng cảnh báo `oneDNN` hay `I0000` phía trên là bình thường, bỏ qua được.

---

## BƯỚC 2 — Tải dataset (10–20 phút)

### Cách A — Tải thủ công (dễ nhất)

1. Vào https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition, đăng nhập và bấm **Download**. File `archive.zip` khoảng 2 GB.
2. Tạo thư mục `D:\vegetable-classifier\data\raw\`.
3. Giải nén `archive.zip` vào `data\raw\`.

### Cách B — Dùng Kaggle API

1. Trên kaggle.com, vào **Settings → API → Create New Token** để tải về file `kaggle.json`.
2. Chép file vào `C:\Users\<tên_bạn>\.kaggle\kaggle.json`.
3. Chạy lệnh:

```bat
python scripts\download_dataset.py
```

### Kiểm tra cấu trúc thư mục (bắt buộc)

Cấu trúc phải **đúng như sau**: 3 thư mục `train`, `validation`, `test` nằm **ngay trong** `data\raw`, mỗi thư mục có 36 thư mục class.

```
D:\vegetable-classifier\data\raw\
├── train\        apple\  banana\  beetroot\ ... (36 thư mục)
├── validation\   (36 thư mục)
└── test\         (36 thư mục)
```

Nếu sau khi giải nén bị thừa một cấp thư mục, ví dụ `data\raw\archive\train\...`, hãy chuyển 3 thư mục `train`, `validation`, `test` lên ngay dưới `data\raw\`.

---

## BƯỚC 3 — Kiểm tra và làm sạch dữ liệu (2–5 phút) ⭐

**Không bỏ qua bước này.** Đây chính là lỗi đã làm hỏng kết quả cũ.

### 3.1 Sao lưu kết quả cũ

Giữ lại kết quả cũ để làm slide so sánh "trước và sau khi làm sạch dữ liệu":

```bat
xcopy reports reports_cu_du_lieu_nhieu\ /E /I
```

### 3.2 Chạy kiểm tra

```bat
python scripts\check_duplicates.py
```

Script **chỉ đọc, không sửa gì**. Đọc kết quả theo bảng sau:

| Mục in ra | Kết quả đúng | Nếu khác |
|---|---|---|
| SỐ ẢNH | train ≈ 3115, validation ≈ 351, test ≈ 359 | Test = 409 nghĩa là có ảnh nhiễu, xem 3.3 |
| ẢNH NHIỄU NGẪU NHIÊN | "Không có. ✓" | Chạy 3.3 |
| ẢNH HỎNG | 0 | Chuyển các file đó ra khỏi `data\raw` |
| TRÙNG Y HỆT: train ∩ test | 0 hoặc rất ít | Ghi vào báo cáo: tập test bị rò rỉ một phần |
| TRÙNG Y HỆT: validation ∩ test | Ghi lại con số | Nếu lớn, ghi vào báo cáo: validation và test không độc lập |

Chi tiết từng file nằm trong `reports\duplicate_check.csv` (mở bằng Excel). **Chụp màn hình kết quả**, sẽ dùng làm slide phân tích dữ liệu.

### 3.3 Loại ảnh nhiễu (chỉ khi 3.2 báo có)

```bat
python scripts\check_duplicates.py --move-noise
```

Ảnh nhiễu được **chuyển** (không xoá) sang `data\_removed\`. Chạy lại bước 3.2 để chắc chắn mục ảnh nhiễu báo "Không có. ✓" và test còn khoảng 359 ảnh.

### 3.4 Kiểm tra TensorFlow đọc được dataset

```bat
python -m src.data.dataset
```

**Kết quả đúng:** có dòng `Số class: 36`, số ảnh 3 split khớp với bước 3.2, và `Batch shape: images=(32, 224, 224, 3)`.

---

## BƯỚC 4 — Train Model 1: HOG → PCA → SVM (5–20 phút)

Bước này dùng code của nhóm:
- `hog_features.py` (TN 2): mỗi ảnh thành vector **8.196 số**, gồm 8.100 số HOG và 96 số màu.
- `pca_wrapper.py` (TN 2): chuẩn hoá, rồi giảm xuống **200 chiều**.
- `classical_svm.py` (TN 1): GridSearchCV thử **C ∈ {0.1, 1, 10, 100} × gamma ∈ {scale, auto}**, 5-fold, chấm theo F1 macro.

Tham số lấy từ `configs\config.py` → `CONFIG["svm"]`.

### 4.1 Test nhanh 5 class (khoảng 1 phút)

Chạy thử trước để chắc chắn mọi thứ hoạt động:

```bat
python -m src.training.train_svm --classes 5 --tag quick
```

**Kết quả đúng:** cuối cùng in ra `[classical_svm_quick] Test accuracy ...%`, không có `Traceback`.

### 4.2 Train đầy đủ 36 class

```bat
python -m src.training.train_svm
```

Màn hình sẽ lần lượt hiện:

1. `Trích đặc trưng 3115 ảnh (train)...` có thanh tiến độ. Kết quả được lưu cache vào `checkpoints\features\`, nên lần chạy sau sẽ bỏ qua bước này.
2. `PCA fit xong: 8196 chiều → 200 chiều` và `Phương sai giữ lại: xx%`. **Ghi lại con số % này.**
3. `Bắt đầu GridSearchCV với 8 tổ hợp...`, rồi các dòng `[CV] END ...`. Đây là bước lâu nhất.
4. Khung tổng kết cuối:

```
[classical_svm] Test accuracy xx.xx% | F1 macro 0.xxxx
Train acc xx% | Val acc xx% -> overfitting / good_fit
Best: C=..., gamma=... | PCA giữ xx% phương sai
```

**Ghi lại:** C tốt nhất, gamma tốt nhất, Test accuracy, F1 macro, thời gian train. Bản Word yêu cầu đúng các số này.

Train acc gần 100% là bình thường với SVM: các ảnh train thường trở thành support vector. Hãy nhận xét dựa vào khoảng cách giữa train acc và val acc.

**File sinh ra:**
- `reports\classical_svm_summary.json`, `_metadata.json`, `_evaluation.json`, `_per_class_metrics.csv`
- `reports\figures\classical_svm_confusion_matrix.png`, `classical_svm_misclassified.png`
- `checkpoints\classical_svm.pkl`, `pca_model.pkl`, `svm_model.pkl` (gửi cho **TN 4** gắn vào Streamlit)

### 4.3 (Tuỳ chọn) Thử thêm tham số để có nội dung "tune"

```bat
python -m src.training.train_svm --c 1 10 100 --gamma scale 0.001 0.0001 --tag grid2
python -m src.training.train_svm --n-components 100 --tag pca100
```

Đặt `--tag` để **không ghi đè** kết quả chính. Đặc trưng đã được cache nên các lần sau chạy nhanh hơn nhiều. Ghi kết quả từng lần vào một bảng nhỏ để đưa vào slide "Quá trình tune".

> Nếu bước 4.2 lỗi ở phần của TN 1, có thể dùng bản dự phòng: `python -m src.training.train_svm --backend sklearn`. Bản này chỉ cần code HOG của TN 2.

---

## BƯỚC 5 — Train Model 2: CNN Baseline (30–50 phút)

```bat
python -m src.training.train --model baseline
```

Màn hình hiện từng epoch, tối đa 25 epoch, theo dạng:

```
Epoch 3/25
98/98 ━━━━━━━━━━━━ 95s  - accuracy: 0.31 - loss: 2.45 - val_accuracy: 0.38 - val_loss: 2.10
Epoch 3: val_accuracy improved from 0.33 to 0.38, saving model to ...baseline_cnn.keras
```

- Mỗi khi val_accuracy tăng, model được tự động lưu vào `checkpoints\baseline_cnn.keras`.
- **EarlyStopping** tự dừng nếu val_loss không giảm sau 5 epoch liên tiếp, nên dừng trước epoch 25 là bình thường.
- Đừng tắt cửa sổ cmd. Có thể thu nhỏ cửa sổ và cắm sạc cho laptop.

**Kết quả đúng:** dòng cuối có `Baseline CNN training xong sau ...s. Best val_accuracy=0.xx`.

**File sinh ra:** `reports\baseline_cnn_metadata.json`, `reports\figures\baseline_cnn_training_curves.png` (biểu đồ loss/accuracy theo epoch dùng cho báo cáo), `reports\logs\`.

---

## BƯỚC 6 — Train Model 3: MobileNetV2 Transfer Learning (30–60 phút)

```bat
python -m src.training.train --model transfer
```

- **Lần đầu cần internet** để tải trọng số ImageNet (khoảng 9 MB). Các lần sau đã có cache.
- Model train qua 2 giai đoạn, log sẽ in rõ:
  - `Giai đoạn 1/2: Feature extraction`: đóng băng MobileNetV2, chỉ train phần đầu ra, tối đa 25 epoch.
  - `Giai đoạn 2/2: Fine-tuning`: mở khoá các layer cuối, train với learning rate nhỏ, tối đa 15 epoch.

**Kết quả đúng:** có dòng `Transfer model training xong sau ...s. Best val_accuracy=0.xx`. Val accuracy nên cao hơn hẳn CNN.

**File sinh ra:** `checkpoints\mobilenetv2_transfer.keras`, `reports\mobilenetv2_transfer_metadata.json`, `reports\figures\mobilenetv2_transfer_training_curves.png`.

---

## BƯỚC 7 — Đánh giá CNN và MobileNetV2 trên tập test (2–5 phút)

```bat
python -m src.evaluation.evaluate --compare
```

Lệnh này load lại 2 checkpoint, chạy trên **test** (khoảng 359 ảnh) rồi sinh ra:
- confusion matrix
- ảnh dự đoán đúng và sai
- `reports\baseline_cnn_summary.json` và `reports\mobilenetv2_transfer_summary.json`

**Kết quả đúng:** có 2 dòng `[baseline_cnn] Test accuracy: ...` và `[mobilenetv2_transfer] Test accuracy: ...`.

> Lệnh này cũng ghi `model_comparison.csv` nhưng chỉ có 2 model. Bước 8 sẽ ghi lại file này với đủ 3 model.

---

## BƯỚC 8 — So sánh 3 model (dưới 1 phút) ⭐

```bat
python -m src.evaluation.compare_all
```

Màn hình in ra bảng so sánh:

```
      Display  Test Accuracy  Precision (macro)  Recall (macro)  F1-score (macro)  Training Time (s)  Test Samples
Classical SVM        0.xxxx             ...
 Baseline CNN        0.xxxx             ...
  MobileNetV2        0.xxxx             ...
```

**Kiểm tra quan trọng:** cột `Test Samples` phải **bằng nhau ở cả 3 dòng** (khoảng 359). Nếu có dòng `[!] CẢNH BÁO: các model đánh giá trên số ảnh test khác nhau`, nghĩa là còn dùng report cũ. Hãy chạy lại bước tương ứng (4, 7).

**File sinh ra (dùng cho PPT và báo cáo Word):**

| File | Dùng để |
|---|---|
| `reports\model_comparison.csv` | Bảng so sánh 3 model (mở bằng Excel) |
| `reports\figures\model_comparison_metrics.png` | Biểu đồ cột Accuracy/Precision/Recall/F1 |
| `reports\figures\model_comparison_time.png` | Biểu đồ thời gian train |
| `reports\figures\per_class_f1_comparison.png` | Bảng màu F1 từng class: class nào khó với cả 3 model |
| `reports\figures\*_confusion_matrix.png` | Confusion matrix từng model |
| `reports\evaluation_report.md` | Báo cáo tổng hợp. Mở bằng VS Code (Ctrl+Shift+V để xem) |

---

## BƯỚC 9 — Điền bảng so sánh trong Word

Lấy số từ `reports\model_comparison.csv` và các file `*_metadata.json`:

| Model | Accuracy | Precision | Recall | F1 (macro) | Thời gian train | Hyperparameter chính |
|---|---|---|---|---|---|---|
| HOG → PCA → SVM | | | | | | C = ?, gamma = ?, PCA 200 chiều (giữ ?% phương sai) |
| CNN Baseline | | | | | | lr 0.001, batch 32, ? epoch |
| MobileNetV2 | | | | | | lr 1e-3 → 1e-4, ? + ? epoch |

Gợi ý phần nhận xét (mỗi ý 2–3 câu, kèm số liệu):
1. **Model nào tốt nhất và vì sao:** đặc trưng học tự động (CNN) so với đặc trưng thủ công (HOG); lợi thế pretrain trên ImageNet.
2. **SVM:** train nhanh, không cần GPU, nhưng HOG khó phân biệt các class giống hình dạng (bell pepper, capsicum, paprika; corn và sweetcorn).
3. **Class khó:** lấy 5 class có F1 thấp nhất từ `per_class_f1_comparison.png`.
4. **Overfitting:** so train acc với val acc của từng model (có trong summary và metadata).
5. **Chất lượng dữ liệu:** 150 ảnh nhiễu đã loại ở bước 3, ảnh trùng giữa các split, và kết quả trước/sau khi làm sạch (so với `reports_cu_du_lieu_nhieu\`).

---

## Tóm tắt lệnh (theo đúng thứ tự)

```bat
D:
cd D:\vegetable-classifier
venv\Scripts\activate

python scripts\check_duplicates.py                  &REM 3. kiểm tra dữ liệu
python scripts\check_duplicates.py --move-noise     &REM    (chỉ khi có ảnh nhiễu)
python -m src.data.dataset                          &REM    kiểm tra TF đọc được dữ liệu

python -m src.training.train_svm --classes 5 --tag quick   &REM 4.1 test nhanh
python -m src.training.train_svm                    &REM 4.2 SVM
python -m src.training.train --model baseline       &REM 5. CNN
python -m src.training.train --model transfer       &REM 6. MobileNetV2
python -m src.evaluation.evaluate --compare         &REM 7. đánh giá CNN + MobileNetV2
python -m src.evaluation.compare_all                &REM 8. so sánh 3 model
```

Tổng thời gian khoảng **1,5–2,5 giờ** trên laptop không có GPU, phần lớn là bước 5 và 6. Có thể để máy chạy, nhưng tắt chế độ Sleep: **Settings → System → Power → Screen and sleep → Never**.

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `No module named 'src'` hoặc `'configs'` | Chạy lệnh sai thư mục | `cd D:\vegetable-classifier` rồi chạy lại. Phải dùng `python -m src....`, không dùng `python src\...` |
| `No module named 'tensorflow'` / `'skimage'` | Chưa activate venv | `venv\Scripts\activate`, kiểm tra có `(venv)` ở đầu dòng |
| `Could not find a version that satisfies tensorflow` | Python 3.14 hoặc Python 32-bit | Cài Python 3.12 bản 64-bit (bước 1.1) |
| `FileNotFoundError: Không tìm thấy ảnh nào...` | Sai cấu trúc `data\raw` | Xem lại phần kiểm tra cấu trúc ở bước 2 |
| `Unknown image file format` khi train CNN | Có ảnh hỏng | Chạy bước 3.2, chuyển các ảnh ở mục "ẢNH HỎNG" ra khỏi `data\raw` |
| `ResourceExhaustedError` / máy treo vì hết RAM | Batch quá lớn | Trong `configs\config.py`, sửa `"batch_size": 32` thành `16` |
| `URL fetch failure ... mobilenet_v2_weights` | Không có internet ở lần chạy MobileNetV2 đầu tiên | Kết nối mạng rồi chạy lại bước 6 |
| SVM: `[DỪNG] hog_features.py chưa được implement` | Code của TN 2 chưa được pull về | Lấy bản mới nhất của `src\features\` |
| SVM chạy quá 1 giờ | Máy yếu | Thêm `--max-per-class 60` để tune trên ít ảnh hơn, ghi rõ điều này trong báo cáo |
| Kết quả SVM thay đổi sau khi TN 2 sửa `COLOR_WEIGHT` | Đặc trưng thay đổi | Đây là hành vi đúng: cache tự tính lại khi tham số HOG hoặc màu thay đổi |

Nếu gặp lỗi khác: **chụp toàn bộ màn hình cmd**, nhất là đoạn từ chữ `Traceback` trở xuống, rồi gửi vào nhóm chat. Quy tắc nhóm: tự sửa không quá 30 phút.
