# Dashboard Hillstrom (mô hình cuối từ lần chạy Colab mới)

Chạy: `pip install -r requirements_desktop.txt` rồi `python main.py`

- Mô hình: LR, chiến lược "Không xử lý", 11 biến FS2 (`final_model_FS2.joblib`, `final_model_meta.json` lấy từ kết quả 06).
- Cần scikit-learn == 1.6.1 (đúng phiên bản Colab đã huấn luyện); bản khác sẽ lỗi khi nạp mô hình.
- `hillstrom.csv` là dữ liệu gốc 64.000 dòng dùng cho mô phỏng Top-K.

## Thay đổi so với dashboard cũ
- Xóa `generate_usecase.py`, hàm `show_usecase`, `__pycache__`, mô hình cũ.
- `model_service.py`: thứ tự 11 biến khớp pipeline; recency_group `0-3m` -> `1-3m`; tên mô hình đọc từ meta.
- Bỏ `imbalanced-learn` khỏi requirements (mô hình cuối không dùng).

## Sửa lỗi giao diện
- `ui_theme.py`: style Fusion + bảng màu sáng cố định (Windows để Dark mode không còn làm nền đen).
- Ô nhập số: không dùng `ButtonSymbols.NoButtons` (Qt 6.12 + stylesheet làm ô nhập co còn 1px, không gõ được);
  bấm vào ô là chọn hết để gõ đè; nhận cả dấu `.` và `,` làm dấu thập phân.
- Cuộn chuột qua combobox / ô số / thanh trượt không còn vô tình đổi giá trị.
- Trang dự đoán tự xếp 1 cột khi cửa sổ hẹp; nút "Xuất biểu đồ PNG" đã được gắn vào trang Tổng quan / Khám phá.
- Icon xanh lá: `assets/*_green.png`, `assets/*_sage.png` (tạo từ bộ `*_wine.png`).

## Giao diện v2 (design tokens + chuyển động)
- `ui_theme.py`: design tokens (màu, bo góc, thời lượng chuyển động). Stylesheet viết `@brand700`, `@gold400`…
  rồi qua `qss()`; đổi màu cả ứng dụng chỉ cần sửa bảng `T`. Màu biểu đồ: `SERIES` (đã kiểm tra mù màu).
- `motion.py`: thẻ nổi có bóng đổ tự vẽ (không dùng QGraphicsEffect để cuộn mượt), cuộn chuột mượt,
  chuyển trang mờ dần + trượt, khối chọn menu trượt, số liệu chạy (count-up), nút sáng lên khi rê chuột.
- `app.py`: chỉ vẽ ngay biểu đồ của trang đang xem; trang ẩn được vẽ dần lúc rảnh (đổi bộ lọc ~50 ms thay vì ~800 ms).
- Icon theo token: `assets/*_emerald.png`, `*_gold.png`, `*_indigo.png`, `*_ink.png`.
- Ô số: không tự bôi chọn (chỉ dấu nháy khi đang sửa; nút −/+ không để lại dấu nháy); gõ tự do,
  vượt giới hạn thì tự đưa về mức gần nhất khi xác nhận (vd. 68 -> 12).
- Dropdown (`motion.PolishedComboBox`): luôn mở ngay dưới ô, một khung bo góc có bóng đổ, giá trị đang
  chọn chỉ có chữ đậm + dấu ✓ (không tô nền), mục được rê chuột / chọn bằng phím mũi tên nổi lên nhẹ.
