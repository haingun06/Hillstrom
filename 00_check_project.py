# -*- coding: utf-8 -*-
"""Kiểm tra nhanh cấu trúc và đầu vào trước khi chạy pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
print("Project:", ROOT)
for rel in ["hillstrom.csv", "data/raw/hillstrom.csv", "data/processed/train.csv",
            "data/processed/test.csv", "figures", "models"]:
    p = ROOT / rel
    print(f"{rel}: {'OK' if p.exists() else 'CHƯA CÓ'}")
