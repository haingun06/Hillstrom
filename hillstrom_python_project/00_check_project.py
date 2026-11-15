# -*- coding: utf-8 -*-
"""Kiểm tra nhanh cấu trúc dự án, thư viện và đầu vào trước khi chạy pipeline."""
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parent
print("Project:", ROOT)

print("\n[Thư viện]")
for mod in ["numpy", "pandas", "scipy", "matplotlib", "sklearn", "imblearn", "statsmodels", "joblib"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: {getattr(m, '__version__', 'OK')}")
    except Exception as e:
        print(f"  {mod}: THIẾU ({type(e).__name__}) -> pip install -r requirements.txt")

print("\n[Tệp và thư mục]")
for rel in ["data/raw/hillstrom.csv", "data/processed/train.csv", "data/processed/test.csv",
            "data/processed/train_deduplicated_sensitivity.csv", "data/processed/test_deduplicated_sensitivity.csv",
            "figures", "models", "common.py"]:
    print(f"  {rel}: {'OK' if (ROOT / rel).exists() else 'CHƯA CÓ'}")

print("\n[Script theo thứ tự chạy]")
for f in ["01_prepare_data.py", "02_fs1_experiments.py", "03_fs2_experiments.py", "04_compare_fs1_fs2.py",
          "05_interpretation.py", "06_final_model.py", "07_uplift.py", "08_robustness.py"]:
    print(f"  {f}: {'OK' if (ROOT / f).exists() else 'THIẾU'}")

try:
    import json
    fs1 = json.load(open(ROOT / "figures" / "best_strategy_FS1.json", encoding="utf-8"))
    lr_dt = json.load(open(ROOT / "figures" / "FS2_LR_DT_cv_summary.json", encoding="utf-8"))
    rf_gb = json.load(open(ROOT / "figures" / "FS2_best_params.json", encoding="utf-8"))
    fs2 = {"LR": lr_dt["LR"]["imbalance_strategy"], "DT": lr_dt["DT"]["imbalance_strategy"],
           "RF": rf_gb["RF"]["strategy"], "GB": rf_gb["GB"]["strategy"]}
    print("\n[Chiến lược mất cân bằng: 5.3 (FS1) so với 5.4 (FS2)]")
    for m in ["LR", "DT", "RF", "GB"]:
        flag = "OK" if fs1[m]["strategy"] == fs2[m] else "LỆCH -> chạy lại 03_fs2_experiments.py"
        print(f"  {m}: 5.3 = {fs1[m]['strategy']!r} | 5.4 = {fs2[m]!r}  {flag}")
except Exception as e:
    print("\n(Chưa kiểm tra được chiến lược 5.3/5.4:", type(e).__name__, ")")
