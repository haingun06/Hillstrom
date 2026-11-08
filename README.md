# Hillstrom – Python scripts tách từ Google Colab

Bộ mã này được tách trực tiếp từ notebook `Code (1).ipynb`, giữ thứ tự các cell và phần lớn logic/cấu hình gốc. Các đoạn Markdown trong notebook được chuyển thành comment để giữ lại giải thích và tiêu đề mục.

## 1. Cấu trúc thư mục

```text
hillstrom_python_project/
├── 01_prepare_data.py
├── 02_fs1_experiments.py
├── 03_fs2_experiments.py
├── 04_interpretation.py
├── 05_final_model.py
├── requirements.txt
├── data/
│   ├── raw/                 # đặt hillstrom.csv tại đây (hoặc thư mục gốc)
│   └── processed/           # train/test và outcomes do bước 01 tạo
├── figures/                 # bảng CSV và biểu đồ đầu ra
├── models/                  # mô hình .joblib
└── reports/
```

## 2. Cài đặt

Tạo môi trường ảo (khuyến nghị), sau đó chạy:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

## 3. Dữ liệu đầu vào

Đặt file `hillstrom.csv` vào một trong các vị trí:
- `data/raw/hillstrom.csv` (khuyến nghị)
- `data/hillstrom.csv`
- thư mục gốc dự án: `hillstrom.csv`

Nếu không tìm thấy, script 01 sẽ thử tải dữ liệu từ URL MineThatData như notebook gốc.

## 4. Thứ tự chạy

Mở terminal tại thư mục `hillstrom_python_project` và chạy lần lượt:

```bash
python 01_prepare_data.py
python 02_fs1_experiments.py
python 03_fs2_experiments.py
python 04_interpretation.py
python 05_final_model.py
```

**Không chạy file 02–05 trước file 01.** File 02/03 có các bước GridSearchCV/RandomizedSearchCV và SMOTENC, có thể chạy khá lâu. Nên chạy từng file, kiểm tra kết quả rồi mới chuyển bước tiếp theo.

## 5. Vai trò từng file

| File | Nội dung được gom từ notebook |
|---|---|
| `01_prepare_data.py` | Chương 4: audit dữ liệu, loại trùng, sửa `Surburban`, tạo `log_history`, `promo_match`, `both_gender`, `recency_group`, định nghĩa FS1/FS1P/FS2, chia train/test, pipeline và baseline, minh hoạ leakage |
| `02_fs1_experiments.py` | Chương 5: thí nghiệm FS1 (LR, DT, RF, GB) và so sánh 3 chiến lược mất cân bằng |
| `03_fs2_experiments.py` | Chương 5: thí nghiệm FS2 (LR, DT, RF, GB), chọn ngưỡng bằng OOF, bootstrap và biểu đồ |
| `04_interpretation.py` | Mục 5.6: Odds Ratio, VIF, cây Decision Tree diễn giải, impurity importance và permutation importance |
| `05_final_model.py` | Mục 5.10: tổng hợp kết quả, chọn mô hình cuối, lưu `.joblib`, xuất điểm dự đoán test |

## 6. Những điểm đã điều chỉnh khi chuyển từ Colab sang .py

1. Thêm phần xác định thư mục dự án ở đầu mỗi file để các đường dẫn tương đối được tính từ thư mục dự án, không phụ thuộc thư mục terminal hiện tại.
2. Thêm fallback cho `display()`/`Image` để có thể chạy ngoài notebook.
3. Bỏ các cell chỉ mount/sao lưu Google Drive (cell 37–39 của notebook), vì đây là thao tác riêng của Colab. Các kết quả vẫn được lưu vào `figures/` và `models/` trên máy.
4. Chuyển một số đường dẫn `/content/...` sang đường dẫn cục bộ.
5. Sửa lỗi cú pháp rõ ràng trong cell cuối của notebook: `for m in MODELS:A` thành `for m in MODELS:` để file 05 có thể chạy. Không thay đổi logic phần thân vòng lặp.

## 7. Lưu ý quan trọng

- Chạy từ file 01 để tạo `data/processed/train.csv`, `test.csv`, `outcomes_train.csv`, `outcomes_test.csv`.
- Các script dùng chung file kết quả trung gian trong `figures/` và mô hình trong `models/`; không xoá các thư mục này giữa các bước.
- Một số phần của notebook có fallback cấu hình khi thiếu file JSON/model đã lưu. Khi báo cáo kết quả chính thức, cần ưu tiên kết quả CV thực tế đã chạy, không xem các giá trị fallback là kết quả thực nghiệm.
- Notebook có các thí nghiệm được chia theo thứ tự cell; bộ script giữ cách tổ chức đó để gần notebook nhất có thể, chưa tái cấu trúc toàn bộ thành package/hàm dùng chung.
- Môi trường nên dùng scikit-learn tương đối mới vì notebook sử dụng `OneHotEncoder(sparse_output=...)`.
