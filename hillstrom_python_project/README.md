# Hillstrom – dự án Python (Chương 3-5)

Mô hình dự đoán khả năng mua hàng sau email khuyến mãi (Hillstrom E-Mail Analytics). Chạy mọi script **từ thư mục gốc dự án**, theo đúng thứ tự dưới đây.

## Thứ tự chạy và ánh xạ với báo cáo

| Script | Mục báo cáo | Việc làm | Đầu ra chính |
|---|---|---|---|
| `00_check_project.py` | – | Kiểm tra thư viện, tệp, và chiến lược 5.3 so với 5.4 | in ra màn hình |
| `01_prepare_data.py` | Chương 4 | Làm sạch, đặc trưng, chia 80/20, pipeline, mô hình cơ sở | `data/processed/`, `figures/T4_*.csv` |
| `02_fs1_experiments.py` | 5.2, 5.3 | LR/DT/RF/GB trên FS1; so sánh 3 chiến lược mất cân bằng | `figures/5_2_1_FS1/`, `best_strategy_FS1.json` |
| `03_fs2_experiments.py` | 5.4 | LR/DT/RF/GB trên FS2 với chiến lược đã chọn ở 5.3 | `figures/FS2_*.json`, `T5_4_1_*`, `T5_8_*`, `models/*_FS2_*.joblib` |
| `04_compare_fs1_fs2.py` | 5.5 | Bảng FS1 và FS2, chênh lệch bootstrap ghép cặp | `figures/5_5/` |
| `05_interpretation.py` | 5.6 | Odds Ratio, VIF, cây DT 3-4 tầng; RF impurity và permutation importance | `figures/5_6/` |
| `06_final_model.py` | 5.9 | Chọn mô hình cuối theo CV, lưu joblib, CSV dự đoán test | `models/final_model_FS2.joblib`, `figures/5_9/` |
| `07_uplift.py` | 5.7 | S-learner, đường Qini, chia nhóm uplift, uplift theo loại email | `figures/5_7/`, `models/uplift_slearner.joblib` |
| `08_robustness.py` | 5.8 | Bootstrap 1.000 lần, phân nhóm email, độ nhạy với bản ghi trùng | `figures/5_8/` |

`06` phải chạy trước `07` và `08` (hai script này dùng mô hình cuối). `common.py` chứa hàm dùng chung cho `04`-`08`.
Mục 5.1 (mô tả thiết lập) không có code riêng.

## Nguyên tắc dùng chung
- **Chiến lược mất cân bằng và siêu tham số không đặt cứng.** `04`-`08` đọc từ `best_strategy_FS1.json`, `FS2_LR_DT_cv_summary.json`, `FS2_best_params.json` và nạp mô hình `models/{MÔ HÌNH}_FS2_{chiến lược}.joblib`.
- **Ngưỡng F2 chỉ chọn trên dự đoán out-of-fold của train**; test chỉ dùng để chấm điểm.
- Mỗi script in cảnh báo nếu chiến lược FS2 ở 5.4 khác chiến lược chọn ở 5.3.

- **FS2 có 11 biến** = FS1 (7) + `segment`, `promo_match`, `recency_group`, `both_gender`. Vì `both_gender = mens + womens − 1`, VIF của ba cột này là vô hạn; mô hình Odds Ratio và cây DT ở `05` dùng `gender_profile` (Men only / Women only / Both) thay cho chúng.

## Những chỗ đã sửa so với bản gửi
1. `03_fs2_experiments.py`: bỏ phép `assert` cứng "LR = class_weight, DT = Không xử lý" và LR/DT ở 5.4.1 nay lấy `class_weight` theo `best_strategy_FS1.json` (trước đây luôn dùng `balanced` cho LR dù 5.3 chọn gì).
2. `01_prepare_data.py`: bản khử trùng lặp (`*_deduplicated_sensitivity.csv`) nay cũng được sửa "Surburban" thành "Suburban". Trước đó ba tệp này còn lỗi chính tả.

## Việc bạn cần làm trước khi lấy số liệu cho báo cáo
Các kết quả 5.4 trong bản gửi được tạo bằng chiến lược của lần chạy 5.3 **cũ** (LR: class_weight, RF: Không xử lý, GB: class_weight), không khớp 5.3 mới (LR: Không xử lý, RF: SMOTENC, GB: Không xử lý). Cách xử lý:
1. Chạy `00_check_project.py` để thấy các mô hình bị lệch.
2. Chạy lại `01` → `03` (nếu `02` chưa chạy lần cuối thì chạy cả `02`), sau đó `04` → `08`.
3. Các mô hình `LR_FS1_class_weight`, `GB_FS1_class_weight`, `RF_FS1_Không_xử_lý` và `final_model_FS2` trong thư mục `models/` là bản cũ (không dùng nữa); `final_model_FS2.joblib` sẽ được `06` ghi lại.

## Thư viện
`pip install -r requirements.txt` (cần `imbalanced-learn` cho SMOTENC và `statsmodels` cho OR/VIF).
