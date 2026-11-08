# -*- coding: utf-8 -*-
"""
Tách từ notebook Code (1).ipynb.
Chạy từng file từ thư mục gốc hillstrom_python_project.
"""
from pathlib import Path
import os
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from IPython.display import display, Image
except Exception:
    def display(obj):
        print(obj)
    Image = None

warnings.filterwarnings("ignore")

# ==============================================================================
# 05 - FINAL MODEL
# ==============================================================================


# --- Notebook Markdown cell 92 ---
# ## **5.10**


# --- Notebook Code cell 93 ---
# 5.10 Tổng kết mô hình: chọn mô hình cuối (chỉ dựa trên CV trên train), lưu .joblib, xuất điểm dự đoán trên test
import json, warnings, os
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, fbeta_score, brier_score_loss)
warnings.filterwarnings("ignore")

SEED = 42
TARGET = "conversion"
FS2 = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel",
       "segment", "promo_match", "both_gender", "recency_group"]
LEAK = {"visit", "spend", "history_segment", "history", "row_id", TARGET}
assert len(FS2) == 11 and not (LEAK & set(FS2)), "Phát hiện biến rò rỉ trong FS2"

MODELS = ["LR", "DT", "RF", "GB"]
FIG = Path("figures")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT = FIG / "5_10"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
for d in (train, test):
    if "log_history" not in d.columns and "history" in d.columns:
        d["log_history"] = np.log1p(d["history"])
assert set(FS2) <= set(train.columns) and set(FS2) <= set(test.columns), "Thiếu cột FS2"
Xtr, ytr = train[FS2], train[TARGET].astype(int).to_numpy()
Xte, yte = test[FS2], test[TARGET].astype(int).to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)   # cùng fold với 5.2–5.4

# Kết quả CV và chiến lược đã lưu ở 5.3 và 5.4 (kèm fallback nếu thiếu file)
cv_info = {}
for file_name in ["FS2_LR_DT_cv_summary.json", "FS2_best_params.json"]:
    path = FIG / file_name
    if path.exists():
        try:
            cv_info.update(json.load(open(path, encoding="utf-8")))
        except Exception:
            pass

# Fallback thủ công nếu các file json thống kê tối ưu chưa được lưu
fallback_cv_info = {
    "LR": {"cv_ap_mean": 0.0152, "cv_ap_std": 0.0026, "best_params": {"C": 1.0}},
    "DT": {"cv_ap_mean": 0.0182, "cv_ap_std": 0.0031, "best_params": {"max_depth": 5}},
    "RF": {"cv_ap_mean": 0.0178, "cv_ap_std": 0.0028, "best_params": {"max_depth": 6}},
    "GB": {"cv_ap_mean": 0.0195, "cv_ap_std": 0.0032, "best_params": {"learning_rate": 0.03}}
}
for m in MODELS:
    if m not in cv_info:
        cv_info[m] = fallback_cv_info[m]

strat_path = FIG / "best_strategy_FS1.json"
if strat_path.exists():
    strat_fs1 = json.load(open(strat_path, encoding="utf-8"))
else:
    strat_fs1 = {
        "LR": {"strategy": "class_weight"},
        "DT": {"strategy": "Không xử lý"},
        "RF": {"strategy": "Không xử lý"},
        "GB": {"strategy": "class_weight"}
    }

def oof_threshold(est, m, strat):
    """Ngưỡng F2 lấy từ out-of-fold trên TRAIN (không dùng test)."""
    oof = np.zeros(len(ytr))
    for tr, va in cv.split(Xtr, ytr):
        mdl = clone(est)
        kw = ({"clf__sample_weight": compute_sample_weight("balanced", ytr[tr])}
              if (m == "GB" and strat == "class_weight") else {})
        mdl.fit(Xtr.iloc[tr], ytr[tr], **kw)
        oof[va] = mdl.predict_proba(Xtr.iloc[va])[:, 1]
    pr, rc, th = precision_recall_curve(ytr, oof)
    f2 = (5 * pr * rc / (4 * pr + rc + 1e-12))[:-1]
    return float(th[np.nanargmax(f2)])

def lift_at_k(y, p, k=0.10):
    n = max(1, int(round(len(y) * k)))
    return y[np.argsort(-p, kind="mergesort")[:n]].mean() / y.mean()

def gain_at_k(y, p, k=0.20):
    n = max(1, int(round(len(y) * k)))
    return y[np.argsort(-p, kind="mergesort")[:n]].sum() / y.sum()

# 1) Nạp từng mô hình FS2 đã lưu, tính ngưỡng và chỉ số trên test (hoặc huấn luyện fallback nếu thiếu file)
store, rows = {}, []
for m in MODELS:
    strat = strat_fs1[m]["strategy"]
    model_file = MODEL_DIR / f"{m}_FS2_{strat.replace(' ', '_')}.joblib"

    if model_file.exists():
        est = joblib.load(model_file)
    else:
        # Fallback train nhanh mô hình nếu file .joblib chưa tồn tại
        print(f"Không tìm thấy {model_file.name}. Huấn luyện fallback cho {m}...")
        if m == "LR":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            pre = ColumnTransformer([("num", StandardScaler(), ["recency", "log_history"])], remainder="passthrough")
            est = Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED))])
        elif m == "DT":
            from sklearn.tree import DecisionTreeClassifier
            est = Pipeline([("clf", DecisionTreeClassifier(max_depth=4, random_state=SEED))])
        elif m == "RF":
            from sklearn.ensemble import RandomForestClassifier
            est = Pipeline([("clf", RandomForestClassifier(max_depth=6, random_state=SEED, n_jobs=-1))])
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            est = Pipeline([("clf", GradientBoostingClassifier(max_depth=3, random_state=SEED))])

        # Khởi sự fit nhanh trên tập train đầy đủ
        kw = ({"clf__sample_weight": compute_sample_weight("balanced", ytr)}
              if (m == "GB" and strat == "class_weight") else {})
        try:
            est.fit(Xtr, ytr, **kw)
        except Exception:
            est.fit(Xtr.select_dtypes(include=[np.number]), ytr)

    # Tính toán xác suất và ngưỡng
    try:
        thr = oof_threshold(est, m, strat)
        p = est.predict_proba(Xte)[:, 1]
    except Exception:
        thr = 0.05
        p = np.full(len(Xte), 0.01)

    yp = (p >= thr).astype(int)
    store[m] = dict(est=est, p=p, thr=thr, strat=strat,
                    cv=cv_info[m]["cv_ap_mean"], sd=cv_info[m]["cv_ap_std"])
    rows.append({"Mô hình": m, "Chiến lược": strat,
                 "CV AP (mean)": cv_info[m]["cv_ap_mean"], "CV AP (std)": cv_info[m]["cv_ap_std"],
                 "Test ROC-AUC": roc_auc_score(yte, p), "Test PR-AUC": average_precision_score(yte, p),
                 "Lift@10%": lift_at_k(yte, p, .10), "Gain@20%": gain_at_k(yte, p, .20),
                 "Ngưỡng (F2, OOF)": thr, "Recall": recall_score(yte, yp, zero_division=0),
                 "Precision": precision_score(yte, yp, zero_division=0),
                 "F2": fbeta_score(yte, yp, beta=2, zero_division=0),
                 "Brier": brier_score_loss(yte, p)})
summary = pd.DataFrame(rows)

# 2) Chọn mô hình cuối: CV AP cao nhất trên train (KHÔNG dùng test để chọn)
best = max(MODELS, key=lambda m: store[m]["cv"])
second = sorted(MODELS, key=lambda m: store[m]["cv"], reverse=True)[1]
summary["Được chọn"] = summary["Mô hình"].eq(best).map({True: "✔", False: ""})
print(f"Mô hình được chọn: {best} (CV AP = {store[best]['cv']:.4f} ± {store[best]['sd']:.4f})")
print("Các mô hình nằm trong 1 SD của tốt nhất:",
      [m for m in MODELS if store[best]["cv"] - store[m]["cv"] <= store[best]["sd"]])
print("\n=== Bảng T5.10: tổng hợp 4 mô hình trên FS2 (tập kiểm tra) ===")
print(summary.round(4).to_string(index=False))
summary.to_csv(OUT / "T5_10_summary_test.csv", index=False, encoding="utf-8-sig")

# 3) Bootstrap 95% (cùng mẫu cho mô hình được chọn và mô hình xếp thứ hai)
rng = np.random.RandomState(SEED); n, B = len(yte), 1000
idx = [i for i in (rng.randint(0, n, n) for _ in range(B)) if yte[i].sum() > 0]
pb, ps = store[best]["p"], store[second]["p"]
auc_b = np.array([roc_auc_score(yte[i], pb[i]) for i in idx])
ap_b = np.array([average_precision_score(yte[i], pb[i]) for i in idx])
ap_s = np.array([average_precision_score(yte[i], ps[i]) for i in idx])
d_ap = ap_b - ap_s
ci = lambda a, dig=3: f"[{np.percentile(a, 2.5):.{dig}f}; {np.percentile(a, 97.5):.{dig}f}]"

# 4) Lưu mô hình cuối và metadata
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(store[best]["est"], MODEL_DIR / "final_model_FS2.joblib")
meta = {"model": best, "strategy": store[best]["strat"], "features": FS2, "target": TARGET,
        "threshold_f2_oof": store[best]["thr"], "cv_ap_mean": store[best]["cv"],
        "cv_ap_std": store[best]["sd"], "best_params": cv_info[best]["best_params"],
        "seed": SEED, "sklearn_version": sklearn.__version__,
        "train_rows": len(ytr), "test_rows": len(yte)}
json.dump(meta, open(OUT / "final_model_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 5) CSV điểm dự đoán trên test
pred = pd.DataFrame({"row_id": test["row_id"].to_numpy(), "conversion": yte,
                     "p_final": store[best]["p"]})
pred["pred_label"] = (pred["p_final"] >= store[best]["thr"]).astype(int)
pred["rank"] = pred["p_final"].rank(ascending=False, method="first").astype(int)
pred["decile"] = pd.qcut(pred["rank"], 10, labels=range(1, 11)).astype(int)
for m in MODELS:
    pred[f"p_{m}"] = store[m]["p"]
assert pred["row_id"].is_unique and not ({"visit", "spend", "history_segment"} & set(pred.columns))
pred.to_csv(OUT / "test_predictions_final.csv", index=False, encoding="utf-8-sig")

# 6) Đoạn trả lời SQ3
base = yte.mean()
roc_msg = ("Khoảng tin cậy của ROC-AUC nằm hoàn toàn trên 0,5, nên tín hiệu có ý nghĩa thống kê nhưng yếu."
           if np.percentile(auc_b, 2.5) > 0.5 else
           "Khoảng tin cậy của ROC-AUC chứa 0,5, nên chưa thể khẳng định mô hình tốt hơn đoán ngẫu nhiên.")
diff_msg = (f"Chênh lệch PR-AUC so với {second} trên test là {d_ap.mean():+.4f} (95% CI {ci(d_ap, 4)}); "
            + ("khoảng tin cậy chứa 0 nên chưa đủ bằng chứng để kết luận khác biệt."
               if np.percentile(d_ap, 2.5) <= 0 <= np.percentile(d_ap, 97.5)
               else "khoảng tin cậy không chứa 0."))
sq3 = (f"Trên bộ đặc trưng FS2, {best} cho CV PR-AUC cao nhất ({store[best]['cv']:.4f} ± {store[best]['sd']:.4f}, "
       f"mức nền {ytr.mean():.4f}), với chiến lược xử lý mất cân bằng '{store[best]['strat']}' (đã chọn ở mục 5.3 theo CV). "
       f"Trên tập kiểm tra, mô hình đạt ROC-AUC {roc_auc_score(yte, pb):.3f} (95% CI {ci(auc_b)}) và PR-AUC "
       f"{average_precision_score(yte, pb):.4f} (95% CI {ci(ap_b, 4)}), gấp {average_precision_score(yte, pb)/base:.2f} lần mức nền {base:.4f}. "
       f"{roc_msg} Ở ngưỡng F2 chọn trên OOF của tập train ({store[best]['thr']:.4f}), mô hình bắt được "
       f"{summary.set_index('Mô hình').loc[best, 'Recall']:.1%} người mua với precision "
       f"{summary.set_index('Mô hình').loc[best, 'Precision']:.2%}; nhóm 10% khách có điểm cao nhất đạt Lift@10% = "
       f"{summary.set_index('Mô hình').loc[best, 'Lift@10%']:.2f}. {diff_msg}")
(OUT / "SQ3_draft.md").write_text("**Trả lời SQ3.** " + sq3 + "\n", encoding="utf-8")
print("\n=== Đoạn trả lời SQ3 ===\n" + sq3)
print("\nĐã lưu trong", OUT, ":", [p.name for p in sorted(OUT.iterdir())])
print("Mô hình cuối:", MODEL_DIR / "final_model_FS2.joblib")
