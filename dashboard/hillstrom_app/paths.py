from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # thư mục Dashboard/

DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
ICONS_DIR = ROOT / 'assets' / 'icons'

DATA_CSV = DATA_DIR / 'hillstrom.csv'              # dữ liệu gốc 64.000 dòng
MODEL_FILE = MODELS_DIR / 'final_model_FS2.joblib'  # mô hình cuối (scikit-learn 1.6.1)
MODEL_META = MODELS_DIR / 'final_model_meta.json'   # thông tin mô hình (tên, chiến lược…)
