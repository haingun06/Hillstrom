"""Điểm khởi động ứng dụng Hillstrom Campaign Studio.

Người dùng: bấm đúp run.bat (Windows) / chạy `sh run.sh` (macOS, Linux).
Lập trình viên: `uv run main.py`. Mã nguồn nằm trong thư mục hillstrom_app/.
"""
import sys

from hillstrom_app.ui.main_window import HillstromIntegrated, main  # noqa: F401
from hillstrom_app.ui.theme import apply_light_theme  # noqa: F401

if __name__ == '__main__':
    sys.exit(main())
