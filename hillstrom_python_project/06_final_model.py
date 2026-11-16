# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys
import json
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score,
                             fbeta_score, brier_score_loss)

from common import (FIG_DIR, MODEL_DIR, SEED, TARGET, FS2, MODELS, LEAK, load_split, fs2_config,
                    load_fs2_model, check_strategy_consistency, oof_proba, best_thr_f2, lift_at_k,
                    gain_at_k, boot_indices, ci95, bootstrap_metrics)

OUT = FIG_DIR / "5_9"
OUT.mkdir(parents=True, exist_ok=True)

train, test = load_split()
Xtr, ytr = train[FS2], train[TARGET].astype(int).to_numpy()
Xte, yte = test[FS2], test[TARGET].astype(int).to_numpy()
assert len(FS2) == 11 and not (LEAK & set(FS2))
print(f"Train {len(ytr):,} ({ytr.sum()} người mua, {ytr.mean():.4%}) | Test {len(yte):,} ({yte.sum()} người mua, {yte.mean():.4%})")

cfg = fs2_config()
bad = check_strategy_consistency()

# 1) Nạp mô hình, tính ngưỡng (OOF trên train) và chỉ số trên test
store, rows = {}, []
for m in MODELS:
    strat = cfg[m]["strategy"]
    est = load_fs2_model(m, cfg)
    thr = best_thr_f2(ytr, oof_proba(est, m, strat, Xtr, ytr))      # chỉ dùng train
    p = est.predict_proba(Xte)[:, 1]                                 # test dùng một lần
    yp = (p >= thr).astype(int)
    store[m] = dict(est=est, p=p, thr=thr, strat=strat, cv=cfg[m]["cv_ap"], sd=cfg[m]["cv_sd"])
    rows.append({"Mô hình": m, "Chiến lược": strat, "CV AP (mean)": cfg[m]["cv_ap"], "CV AP (std)": cfg[m]["cv_sd"],
                 "Test ROC-AUC": roc_auc_score(yte, p), "Test PR-AUC": average_precision_score(yte, p),
                 "Lift@10%": lift_at_k(yte, p, .10), "Gain@20%": gain_at_k(yte, p, .20),
                 "Ngưỡng (F2, OOF)": thr, "Recall": recall_score(yte, yp, zero_division=0),
                 "Precision": precision_score(yte, yp, zero_division=0),
                 "F2": fbeta_score(yte, yp, beta=2, zero_division=0), "Brier": brier_score_loss(yte, p)})
summary = pd.DataFrame(rows)

# 2) Chọn mô hình cuối: CV AP cao nhất trên train
order = sorted(MODELS, key=lambda m: store[m]["cv"], reverse=True)
best, second = order[0], order[1]
near = [m for m in MODELS if store[best]["cv"] - store[m]["cv"] <= store[best]["sd"]]
summary["Được chọn"] = summary["Mô hình"].eq(best).map({True: "✔", False: ""})
print(f"\nMô hình được chọn: {best} (CV AP = {store[best]['cv']:.4f} ± {store[best]['sd']:.4f}), "
      f"chiến lược: {store[best]['strat']}")
print("Các mô hình nằm trong 1 SD của mô hình tốt nhất:", near)
print("\n=== Bảng 5.9: 4 mô hình trên FS2 (tập kiểm tra) ===")
print(summary.round(4).to_string(index=False))
summary.to_csv(OUT / "T5_9_summary_test.csv", index=False, encoding="utf-8-sig")

# 3) Bootstrap 1.000 lần (cùng mẫu cho mọi mô hình)
idx = boot_indices(yte)
b = {m: bootstrap_metrics(yte, store[m]["p"], idx) for m in MODELS}
d_ap = b[best][1] - b[second][1]
d_auc = b[best][0] - b[second][0]
ci_rows = [{"Mô hình": m, "Test ROC-AUC": roc_auc_score(yte, store[m]["p"]), "ROC-AUC 95% CI": ci95(b[m][0]),
            "Test PR-AUC": average_precision_score(yte, store[m]["p"]), "PR-AUC 95% CI": ci95(b[m][1], 4),
            "Lift@10%": lift_at_k(yte, store[m]["p"], .10), "Lift@10% 95% CI": ci95(b[m][2], 2)} for m in MODELS]
pd.DataFrame(ci_rows).to_csv(OUT / "T5_9b_bootstrap_ci_4models.csv", index=False, encoding="utf-8-sig")
print(f"\nChênh lệch {best} - {second}: ΔPR-AUC = {d_ap.mean():+.4f} {ci95(d_ap, 4)} | "
      f"ΔROC-AUC = {d_auc.mean():+.4f} {ci95(d_auc, 4)}")

# 4) Lưu mô hình cuối và siêu dữ liệu
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(store[best]["est"], MODEL_DIR / "final_model_FS2.joblib")
meta = {"model": best, "strategy": store[best]["strat"], "features": FS2, "target": TARGET,
        "threshold_f2_oof": store[best]["thr"], "cv_ap_mean": store[best]["cv"], "cv_ap_std": store[best]["sd"],
        "best_params": cfg[best]["params"], "models_within_1sd": near, "second_model": second,
        "seed": SEED, "sklearn_version": sklearn.__version__, "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
        "strategy_consistent_with_5_3": not bad}
json.dump(meta, open(OUT / "final_model_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 5) CSV điểm dự đoán trên test (không chứa visit/spend/history_segment)
pred = pd.DataFrame({"row_id": test["row_id"].to_numpy(), "conversion": yte, "p_final": store[best]["p"]})
pred["pred_label"] = (pred["p_final"] >= store[best]["thr"]).astype(int)
pred["rank"] = pred["p_final"].rank(ascending=False, method="first").astype(int)
pred["decile"] = pd.qcut(pred["rank"], 10, labels=range(1, 11)).astype(int)
for m in MODELS:
    pred[f"p_{m}"] = store[m]["p"]
assert pred["row_id"].is_unique and not ({"visit", "spend", "history_segment"} & set(pred.columns))
pred.to_csv(OUT / "test_predictions_final.csv", index=False, encoding="utf-8-sig")

# 6) Số liệu trả lời SQ3 (bản nháp sinh tự động từ kết quả; cần đọc lại trước khi đưa vào báo cáo)
base = yte.mean()
pb = store[best]["p"]
auc_b, ap_b = b[best][0], b[best][1]
sr = summary.set_index("Mô hình")
roc_msg = ("khoảng tin cậy của ROC-AUC nằm hoàn toàn trên 0,5 (tín hiệu có ý nghĩa thống kê nhưng yếu)"
           if np.percentile(auc_b, 2.5) > 0.5 else
           "khoảng tin cậy của ROC-AUC chứa 0,5 (chưa khẳng định được mô hình tốt hơn đoán ngẫu nhiên)")
diff_msg = ("khoảng tin cậy chứa 0 nên chưa đủ bằng chứng để kết luận khác biệt"
            if np.percentile(d_ap, 2.5) <= 0 <= np.percentile(d_ap, 97.5) else "khoảng tin cậy không chứa 0")
sq3 = (f"Trên FS2, {best} có CV PR-AUC cao nhất ({store[best]['cv']:.4f} ± {store[best]['sd']:.4f}; mức nền "
       f"{ytr.mean():.4f}), chiến lược xử lý mất cân bằng '{store[best]['strat']}'. Trên tập kiểm tra: ROC-AUC "
       f"{roc_auc_score(yte, pb):.3f} (95% CI {ci95(auc_b)}), PR-AUC {average_precision_score(yte, pb):.4f} "
       f"(95% CI {ci95(ap_b, 4)}), gấp {average_precision_score(yte, pb) / base:.2f} lần mức nền {base:.4f}; {roc_msg}. "
       f"Ở ngưỡng F2 chọn trên OOF của train ({store[best]['thr']:.4f}): recall {sr.loc[best, 'Recall']:.1%}, "
       f"precision {sr.loc[best, 'Precision']:.2%}; Lift@10% = {sr.loc[best, 'Lift@10%']:.2f}. "
       f"So với {second}, ΔPR-AUC = {d_ap.mean():+.4f} (95% CI {ci95(d_ap, 4)}); {diff_msg}. "
       f"Các mô hình nằm trong 1 SD của CV AP tốt nhất: {', '.join(near)}.")
(OUT / "SQ3_draft.md").write_text("**Trả lời SQ3 (nháp tự sinh).** " + sq3 + "\n", encoding="utf-8")
print("\n=== Đoạn số liệu SQ3 ===\n" + sq3)
print("\nĐã lưu:", sorted(p.name for p in OUT.iterdir()), "| mô hình cuối:", MODEL_DIR / "final_model_FS2.joblib")
if bad:
    print("\nLƯU Ý: chiến lược FS2 (5.4) chưa khớp 5.3 ở", bad, "- chạy lại 03 rồi chạy lại file này.")
