#!/usr/bin/env sh
# Hillstrom Campaign Studio - chạy ứng dụng trên macOS / Linux:  sh run.sh
# Lần đầu: tự cài uv (nếu chưa có), tải Python 3.12, tạo .venv và cài đúng phiên bản
# thư viện đã khóa trong uv.lock (cần Internet). Các lần sau: mở ứng dụng ngay.
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    if [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "[Hillstrom] Chưa có uv trên máy - đang cài đặt (chỉ cần làm 1 lần)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

echo "[Hillstrom] Đang kiểm tra môi trường - lần đầu có thể mất vài phút..."
exec uv run --locked main.py
