# Hillstrom Campaign Studio

Ứng dụng desktop dự đoán khả năng mua hàng theo loại email, mô phỏng hiệu quả chiến dịch và phân tích
dữ liệu Hillstrom (64.000 khách hàng).

## Cách chạy (người dùng)
- **Windows:** bấm đúp `run.bat`.
- **macOS / Linux:** mở Terminal tại thư mục này, chạy `sh run.sh`.

Lần đầu (cần Internet, vài phút): script tự cài [uv](https://docs.astral.sh/uv/) nếu máy chưa có, tự tải
Python 3.12 (không ảnh hưởng Python đang có trên máy), tạo `.venv` và cài đúng phiên bản thư viện đã khóa
trong `uv.lock`. Các lần sau mở ứng dụng ngay. Không cần tự cài Python hay thư viện.

## Cấu trúc thư mục

```
Dashboard/
├── run.bat / run.sh            Bấm để chạy ứng dụng (Windows / macOS, Linux)
├── README.md                   Tài liệu này
├── main.py                     Điểm khởi động (gọi hillstrom_app)
│
├── pyproject.toml              Khai báo thư viện + phiên bản Python (uv đọc file này)
├── uv.lock                     Phiên bản chính xác của mọi thư viện — không sửa tay
├── .python-version             Bản Python uv sẽ dùng (3.12)
├── requirements_desktop.txt    Danh sách thư viện cho ai cài thủ công bằng pip
│
├── data/
│   └── hillstrom.csv           Dữ liệu gốc 64.000 dòng (trang phân tích + mô phỏng Top-K)
├── models/
│   ├── final_model_FS2.joblib  Mô hình cuối (Logistic Regression, 11 biến FS2)
│   └── final_model_meta.json   Thông tin mô hình (tên, chiến lược, phiên bản scikit-learn…)
├── assets/
│   └── icons/                  Icon PNG, tên dạng <icon>_<tông màu>.png
│
└── hillstrom_app/              MÃ NGUỒN
    ├── paths.py                Đường dẫn tới data/, models/, assets/icons/ (sửa ở đây nếu dời thư mục)
    ├── core/                   Xử lý dữ liệu & mô hình — không phụ thuộc giao diện
    │   ├── data_logic.py       Đọc CSV, lọc theo nhóm, thống kê EDA, kiểm tra chất lượng dữ liệu
    │   └── model_service.py    Nạp mô hình, dự đoán xác suất mua hàng, mô phỏng chiến dịch Top-K
    └── ui/                     Giao diện PyQt6
        ├── main_window.py      Cửa sổ chính: sidebar + chuyển giữa các trang
        ├── prediction_page.py  Trang "Dự đoán & mô phỏng"
        ├── analytics_pages.py  Trang "Tổng quan chiến dịch", "Khám phá dữ liệu", "Chất lượng dữ liệu"
        ├── dialogs.py          Popup thông báo riêng của ứng dụng (lỗi hệ thống, thành công)
        ├── motion.py           Hiệu ứng: thẻ nổi, cuộn mượt, chuyển trang, dropdown, số liệu chạy
        └── theme.py            Design tokens (màu, bo góc, thời lượng hiệu ứng), font, nạp icon
```

`.venv/` (môi trường Python, tự tạo khi chạy) và `__pycache__/` không cần bàn giao.

## Dành cho người phát triển
- Chạy: `uv run main.py`. Chạy riêng một trang để thử: `uv run python -m hillstrom_app.ui.prediction_page`
  hoặc `uv run python -m hillstrom_app.ui.analytics_pages`.
- Thêm / nâng thư viện: `uv add <tên>` hoặc sửa `pyproject.toml` rồi `uv lock`. Nhớ cập nhật
  `requirements_desktop.txt` cho người dùng pip.
- **scikit-learn phải giữ 1.6.1** (bản Colab đã huấn luyện mô hình); vì vậy Python giới hạn 3.12–3.13.
- Đổi màu / bo góc / tốc độ hiệu ứng: sửa bảng `T`, `RADIUS`, `MOTION` trong `ui/theme.py`. Stylesheet viết
  dạng `@brand700`, `@gold400`… và được thay giá trị qua `qss()`. Màu biểu đồ: `SERIES` (đã kiểm tra mù màu).
- Icon nạp qua `theme.icon(name, tone)`; thiếu file đúng tông thì tự tô màu từ tông khác và in cảnh báo
  `[Hillstrom] Thiếu icon ...` ra terminal.
- Thông báo: lỗi nhập liệu hiển thị ngay tại ô (viền đỏ + dòng chữ, `Field.set_error`); chỉ lỗi hệ thống và
  thông báo thành công mới dùng popup (`ui/dialogs.py`). Không dùng `QMessageBox`.

## Mô hình
- LR, chiến lược "Không xử lý", 11 biến FS2: recency, log_history, mens, womens, zip_code, newbie, channel,
  segment, promo_match, recency_group, both_gender.
- `visit` và `spend` là kết quả sau chiến dịch, chỉ dùng cho EDA, không đưa vào mô hình.
