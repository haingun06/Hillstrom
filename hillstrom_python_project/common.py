# -*- coding: utf-8 -*-
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"

SEED = 42
N_BOOT = 1000
TARGET = "conversion"
MODELS = ["LR", "DT", "RF", "GB"]
STRATS = ["Không xử lý", "class_weight", "SMOTENC"]

# Bộ đặc trưng (khớp 01_prepare_data.py)
FS1 = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel"]
FS1P = FS1 + ["segment", "promo_match"]
FS2 = FS1P + ["recency_group", "both_gender"]
LEAK = {"visit", "spend", "history_segment", "history", "row_id", TARGET}
assert len(FS2) == 11 and not (LEAK & set(FS2)), "FS2 phải có 11 biến và không chứa biến rò rỉ"

CAT_COLS = ["zip_code", "channel", "segment", "recency_group"]
BIN_COLS = ["mens", "womens", "newbie", "promo_match", "both_gender"]
NUM_COLS = ["recency", "log_history"]
LEVELS = {
    "zip_code": ["Rural", "Suburban", "Urban"],
    "channel": ["Multichannel", "Phone", "Web"],
    "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
    "recency_group": ["1-3m", "4-6m", "7-9m", "10-12m"],
}
CONTROL = "No E-Mail"
ARMS = ["Mens E-Mail", "Womens E-Mail"]


# ------------------------------------------------------------------------------
# Dữ liệu
# ------------------------------------------------------------------------------
def load_split(suffix=""):
    """Nạp train/test. suffix='' (mẫu chính 64.000 dòng) hoặc '_deduplicated_sensitivity'."""
    tr = pd.read_csv(DATA_DIR / f"train{suffix}.csv")
    te = pd.read_csv(DATA_DIR / f"test{suffix}.csv")
    for nm, d in (("train", tr), ("test", te)):
        missing = set(FS2 + [TARGET, "row_id"]) - set(d.columns)
        assert not missing, f"{nm}{suffix}.csv thiếu cột: {missing}"
        assert not ({"visit", "spend", "history_segment"} & set(d.columns)), f"Rò rỉ trong {nm}{suffix}.csv"
        for c, lv in LEVELS.items():
            assert set(d[c].unique()) <= set(lv), f"Mức lạ ở {c} trong {nm}{suffix}.csv"
    assert set(tr["row_id"]).isdisjoint(te["row_id"]), "row_id trùng giữa train và test"
    return tr, te


def cv_splitter():
    """Cùng 5 fold phân tầng với 5.2-5.4."""
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


# ------------------------------------------------------------------------------
# Chiến lược, siêu tham số và mô hình FS2 đã lưu
# ------------------------------------------------------------------------------
def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fs2_config():
    """Gộp cấu hình FS2 của 4 mô hình: chiến lược, CV AP, tham số tốt nhất (đọc từ 5.4.1 và 5.4.2)."""
    lr_dt = read_json(FIG_DIR / "FS2_LR_DT_cv_summary.json")
    rf_gb = read_json(FIG_DIR / "FS2_best_params.json")
    cfg = {}
    for m in ("LR", "DT"):
        d = lr_dt[m]
        cfg[m] = dict(strategy=d["imbalance_strategy"], cv_ap=d["cv_ap_mean"], cv_sd=d["cv_ap_std"],
                      params=d["best_params"], threshold_saved=d.get("threshold_f2_oof"))
    for m in ("RF", "GB"):
        d = rf_gb[m]
        cfg[m] = dict(strategy=d["strategy"], cv_ap=d["cv_ap_mean"], cv_sd=d["cv_ap_std"],
                      params=d["best_params"], threshold_saved=None)
    return cfg


def check_strategy_consistency(verbose=True):
    """So chiến lược dùng ở FS2 (5.4) với chiến lược chọn ở 5.3 (FS1). Trả về danh sách mô hình lệch."""
    fs1 = read_json(FIG_DIR / "best_strategy_FS1.json")
    cfg = fs2_config()
    bad = [m for m in MODELS if fs1[m]["strategy"] != cfg[m]["strategy"]]
    if verbose and bad:
        print("\n[CẢNH BÁO] Chiến lược ở 5.4 (FS2) khác chiến lược chọn ở 5.3 (FS1) với:", bad)
        for m in bad:
            print(f"   {m}: 5.3 -> {fs1[m]['strategy']!r} | 5.4 -> {cfg[m]['strategy']!r}")
        print("   Chạy lại 03_fs2_experiments.py sau khi 02_fs1_experiments.py chạy xong "
              "rồi chạy lại các script sau đó.\n")
    return bad


def model_path(m, strategy):
    return MODEL_DIR / f"{m}_FS2_{strategy.replace(' ', '_')}.joblib"


def load_fs2_model(m, cfg=None):
    cfg = cfg or fs2_config()
    p = model_path(m, cfg[m]["strategy"])
    if not p.exists():
        raise FileNotFoundError(f"Không thấy {p.name}. Chạy 03_fs2_experiments.py trước.")
    return joblib.load(p)


def fit_kwargs(model, strategy, y):
    """sample_weight chỉ dùng cho GB + class_weight (khớp 5.4.2)."""
    if model == "GB" and strategy == "class_weight":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def refit(est, model, strategy, X, y):
    """Huấn luyện lại một bản sao của mô hình đã chọn (cùng siêu tham số, cùng chiến lược)."""
    e = clone(est)
    e.fit(X, y, **fit_kwargs(model, strategy, y))
    return e


def oof_proba(est, model, strategy, X, y, cv=None):
    """Dự đoán out-of-fold trên TRAIN, huấn luyện lại từng fold (kể cả sample_weight/SMOTENC)."""
    cv = cv or cv_splitter()
    X = X.reset_index(drop=True)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y):
        e = clone(est)
        e.fit(X.iloc[tr], y[tr], **fit_kwargs(model, strategy, y[tr]))
        oof[va] = e.predict_proba(X.iloc[va])[:, 1]
    assert not np.isnan(oof).any()
    return oof


# ------------------------------------------------------------------------------
# Chỉ số
# ------------------------------------------------------------------------------
def best_thr_f2(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f2 = (5 * pr * rc / (4 * pr + rc + 1e-12))[:-1]
    return float(th[int(np.nanargmax(f2))]) if len(th) else 0.5


def lift_at_k(y, p, k=0.10):
    n = max(1, int(round(len(y) * k)))
    idx = np.argsort(-p, kind="mergesort")[:n]
    return float(y[idx].mean() / y.mean())


def gain_at_k(y, p, k=0.20):
    n = max(1, int(round(len(y) * k)))
    idx = np.argsort(-p, kind="mergesort")[:n]
    return float(y[idx].sum() / y.sum())


def boot_indices(y, n_boot=N_BOOT, seed=SEED):
    """Chỉ số bootstrap (cùng mẫu dùng cho mọi mô hình để so sánh ghép cặp). Bỏ mẫu không có ca dương."""
    rng = np.random.RandomState(seed)
    n = len(y)
    out = []
    while len(out) < n_boot:
        i = rng.randint(0, n, n)
        if y[i].sum() > 0:
            out.append(i)
    return out


def ci95(a, dig=3):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return f"[{lo:.{dig}f}; {hi:.{dig}f}]"


def bootstrap_metrics(y, p, idx_list):
    """ROC-AUC, PR-AUC, Lift@10% trên các mẫu bootstrap cho trước."""
    roc = np.array([roc_auc_score(y[i], p[i]) for i in idx_list])
    ap = np.array([average_precision_score(y[i], p[i]) for i in idx_list])
    lift = np.array([lift_at_k(y[i], p[i], 0.10) for i in idx_list])
    return roc, ap, lift
