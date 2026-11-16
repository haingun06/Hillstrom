# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from common import (FIG_DIR, MODELS, TARGET, fs2_config, check_strategy_consistency, read_json,
                    boot_indices, lift_at_k, ci95)

OUT = FIG_DIR / "5_5"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
LABEL = {"LR": "Logistic Regression", "DT": "Decision Tree", "RF": "Random Forest", "GB": "Gradient Boosting"}

# ------------------------------------------------------------------------------
# 1. Nạp điểm dự đoán trên test và ghép theo row_id
# ------------------------------------------------------------------------------
fs1 = pd.read_csv(FIG_DIR / "test_scores_FS1_best.csv", encoding="utf-8-sig")
fs2_a = pd.read_csv(FIG_DIR / "FS2_LR_DT_test_scores.csv", encoding="utf-8-sig")
fs2_b = pd.read_csv(FIG_DIR / "test_scores_FS2.csv", encoding="utf-8-sig")

sc = fs1.rename(columns={f"p_{m}": f"FS1_{m}" for m in MODELS})
for part in (fs2_a, fs2_b):
    part = part.rename(columns={c: f"FS2_{c[2:]}" for c in part.columns if c.startswith("p_")})
    # kiểm tra nhãn khớp trước khi ghép
    chk = sc[["row_id", TARGET]].merge(part[["row_id", TARGET]], on="row_id", suffixes=("", "_b"))
    assert len(chk) == len(sc) and (chk[TARGET] == chk[f"{TARGET}_b"]).all(), "Nhãn/row_id không khớp giữa FS1 và FS2"
    sc = sc.merge(part.drop(columns=[TARGET]), on="row_id", how="inner")
assert len(sc) == len(fs1) and sc["row_id"].is_unique
y = sc[TARGET].astype(int).to_numpy()
print(f"Tập test: {len(y):,} khách, {y.sum()} người mua ({y.mean():.4%})")

strat1 = {m: v["strategy"] for m, v in read_json(FIG_DIR / "best_strategy_FS1.json").items()}
cfg2 = fs2_config()
strat2 = {m: cfg2[m]["strategy"] for m in MODELS}
bad = check_strategy_consistency()

# ------------------------------------------------------------------------------
# 2. Bảng chung FS1 và FS2 (điểm đơn + bootstrap ghép cặp)
# ------------------------------------------------------------------------------
idx = boot_indices(y)
cv1 = pd.read_csv(FIG_DIR / "T5_6_best_strategy_FS1.csv", encoding="utf-8-sig").set_index("Mô hình")


def metrics(p):
    return dict(roc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                l10=lift_at_k(y, p, .10), l20=lift_at_k(y, p, .20))


def boot(p):
    return dict(
        roc=np.array([roc_auc_score(y[i], p[i]) for i in idx]),
        ap=np.array([average_precision_score(y[i], p[i]) for i in idx]),
        l10=np.array([lift_at_k(y[i], p[i], .10) for i in idx]),
        l20=np.array([lift_at_k(y[i], p[i], .20) for i in idx]),
    )


rows, drows = [], []
store = {}
for m in MODELS:
    res = {}
    for fs in ("FS1", "FS2"):
        p = sc[f"{fs}_{m}"].to_numpy()
        res[fs] = (metrics(p), boot(p))
    store[m] = res
    for fs, strat, cv_ap, cv_sd in (("FS1", strat1[m], cv1.loc[m, "CV AP"], cv1.loc[m, "± std"]),
                                    ("FS2", strat2[m], cfg2[m]["cv_ap"], cfg2[m]["cv_sd"])):
        pt, bt = res[fs]
        rows.append({
            "Mô hình": m, "Bộ đặc trưng": fs, "Số biến": 7 if fs == "FS1" else 11, "Chiến lược": strat,
            "CV AP (train)": cv_ap, "± CV std": cv_sd,
            "ROC-AUC": pt["roc"], "ROC-AUC 95% CI": ci95(bt["roc"]),
            "PR-AUC": pt["ap"], "PR-AUC 95% CI": ci95(bt["ap"], 4),
            "Lift@10%": pt["l10"], "Lift@20%": pt["l20"],
        })
    for key, name, dig in (("roc", "ROC-AUC", 4), ("ap", "PR-AUC", 5), ("l10", "Lift@10%", 3), ("l20", "Lift@20%", 3)):
        d = res["FS2"][1][key] - res["FS1"][1][key]
        pt_d = res["FS2"][0][key] - res["FS1"][0][key]
        drows.append({"Mô hình": m, "Chỉ số": name,
                      "FS1": res["FS1"][0][key], "FS2": res["FS2"][0][key],
                      "Δ (FS2 − FS1)": pt_d, "Δ 95% CI (bootstrap ghép cặp)": ci95(d, dig),
                      "P(Δ ≤ 0)": float((d <= 0).mean()),
                      "Kết luận": ("FS2 tốt hơn (CI > 0)" if np.percentile(d, 2.5) > 0 else
                                   "FS2 kém hơn (CI < 0)" if np.percentile(d, 97.5) < 0 else
                                   "Chưa phân biệt được (CI chứa 0)")})

t55 = pd.DataFrame(rows)
t55d = pd.DataFrame(drows)
print("\n=== Bảng 5.5a: FS1 và FS2 trên tập kiểm tra ===")
print(t55.round(4).to_string(index=False))
print("\n=== Bảng 5.5b: chênh lệch FS2 - FS1 (bootstrap ghép cặp, 1.000 lần) ===")
print(t55d.round(4).to_string(index=False))
t55.to_csv(OUT / "T5_5a_fs1_vs_fs2_test.csv", index=False, encoding="utf-8-sig")
t55d.to_csv(OUT / "T5_5b_fs1_vs_fs2_delta.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------------------
# 3. Hình: ROC-AUC và PR-AUC theo mô hình; chênh lệch kèm khoảng tin cậy
# ------------------------------------------------------------------------------
x = np.arange(len(MODELS))
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, key, title, base in ((axs[0], "roc", "ROC-AUC", 0.5), (axs[1], "ap", "PR-AUC", y.mean())):
    for k, (fs, col) in enumerate((("FS1", "tab:gray"), ("FS2", "tab:blue"))):
        vals = [store[m][fs][0][key] for m in MODELS]
        lo = [store[m][fs][0][key] - np.percentile(store[m][fs][1][key], 2.5) for m in MODELS]
        hi = [np.percentile(store[m][fs][1][key], 97.5) - store[m][fs][0][key] for m in MODELS]
        ax.bar(x + (k - .5) * .38, vals, .38, color=col, alpha=.85, label=fs,
               yerr=[lo, hi], capsize=3, error_kw={"lw": 1})
    ax.axhline(base, color="k", ls="--", lw=1, label=("Ngẫu nhiên (0,5)" if key == "roc" else f"Mức nền ({base:.4f})"))
    ax.set_xticks(x, MODELS); ax.set_title(f"{title} trên tập kiểm tra (CI 95%)")
    ax.set_ylim(0.45 if key == "roc" else 0, None); ax.legend(fontsize=8)
ax = axs[2]
for k, (key, nm, col) in enumerate((("roc", "ROC-AUC", "tab:blue"), ("ap", "PR-AUC (×10)", "tab:orange"))):
    sc_f = 10 if key == "ap" else 1
    d_pt = [(store[m]["FS2"][0][key] - store[m]["FS1"][0][key]) * sc_f for m in MODELS]
    d_b = [(store[m]["FS2"][1][key] - store[m]["FS1"][1][key]) * sc_f for m in MODELS]
    ax.errorbar(x + (k - .5) * .2, d_pt, yerr=[[a - np.percentile(b, 2.5) for a, b in zip(d_pt, d_b)],
                                                [np.percentile(b, 97.5) - a for a, b in zip(d_pt, d_b)]],
                fmt="o", color=col, capsize=4, label=nm)
ax.axhline(0, color="red", ls="--", lw=1)
ax.set_xticks(x, MODELS); ax.set_title("Chênh lệch FS2 − FS1 (CI 95% ghép cặp)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "H5_5a_fs1_vs_fs2.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
for k, (fs, col) in enumerate((("FS1", "tab:gray"), ("FS2", "tab:blue"))):
    ax.bar(x + (k - .5) * .38, [store[m][fs][0]["l10"] for m in MODELS], .38, color=col, alpha=.85, label=fs)
ax.axhline(1, color="k", ls="--", lw=1, label="Chọn ngẫu nhiên (1,0)")
ax.set_xticks(x, MODELS); ax.set_ylabel("Lift@10%"); ax.set_title("Lift@10% của FS1 và FS2 trên tập kiểm tra"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "H5_5b_lift_fs1_vs_fs2.png", bbox_inches="tight"); plt.close(fig)

print("\nĐã lưu:", sorted(p.name for p in OUT.iterdir()))
if bad:
    print("LƯU Ý: chiến lược FS1 và FS2 chưa khớp nhau ở", bad, "- chưa dùng bảng này cho báo cáo.")
