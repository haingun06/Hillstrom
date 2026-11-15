# -*- coding: utf-8 -*-
"""
08_robustness.py - Mục 5.8: Kiểm tra độ bền
=============================================
Ba kiểm tra, áp dụng cho cả 4 mô hình FS2 của mục 5.4 (đặt mô hình cuối của 5.9 lên đầu):

1. Bootstrap 1.000 lần trên tập test: khoảng tin cậy 95% cho ROC-AUC, PR-AUC, Lift@10% và F2
   (ngưỡng F2 cố định, lấy từ OOF của train), cùng chênh lệch ghép cặp giữa mô hình cuối và từng
   mô hình còn lại (cùng mẫu bootstrap).
2. Phân nhóm trong tập test: toàn bộ, nhóm đã nhận email (Mens + Womens), từng loại email và nhóm
   không nhận email (đối chứng). Dùng lại mô hình đã huấn luyện, không huấn luyện lại.
3. Độ nhạy với bản ghi trùng lặp: mẫu chính (64.000 dòng) giữ nguyên bản ghi trùng vì dữ liệu không
   có mã khách hàng để xác nhận trùng thật; ở đây huấn luyện lại 4 mô hình (cùng siêu tham số và chiến
   lược) trên bản đã loại 6.562 dòng trùng (train_deduplicated_sensitivity.csv) và chấm trên
   test_deduplicated_sensitivity.csv (chia riêng, cùng seed), rồi so với kết quả mẫu chính.

Ngưỡng F2 luôn chọn trên OOF của train (không chọn trên test như bản cũ).
Chạy sau 06_final_model.py từ thư mục gốc dự án. Đầu ra: figures/5_8/.
"""
from pathlib import Path
import os
import sys
import json
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score

from common import (FIG_DIR, MODELS, TARGET, FS2, CONTROL, ARMS, load_split, fs2_config, load_fs2_model,
                    check_strategy_consistency, oof_proba, best_thr_f2, refit, lift_at_k, boot_indices,
                    bootstrap_metrics, ci95, read_json)

OUT = FIG_DIR / "5_8"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
COL = {"LR": "#1f77b4", "DT": "#2ca02c", "RF": "#ff7f0e", "GB": "#d62728"}

train, test = load_split()
Xtr, ytr = train[FS2], train[TARGET].astype(int).to_numpy()
Xte, yte = test[FS2], test[TARGET].astype(int).to_numpy()
cfg = fs2_config()
bad = check_strategy_consistency()
final = read_json(FIG_DIR / "5_9" / "final_model_meta.json")["model"]
ORDER = [final] + [m for m in MODELS if m != final]
print(f"Mô hình cuối (5.9): {final} | Test {len(yte):,} khách, {yte.sum()} người mua ({yte.mean():.4%})")

# ------------------------------------------------------------------------------
# 0. Mô hình đã huấn luyện, ngưỡng (OOF train) và điểm trên test
# ------------------------------------------------------------------------------
est, thr, p_te = {}, {}, {}
for m in MODELS:
    est[m] = load_fs2_model(m, cfg)
    thr[m] = best_thr_f2(ytr, oof_proba(est[m], m, cfg[m]["strategy"], Xtr, ytr))
    p_te[m] = est[m].predict_proba(Xte)[:, 1]

# ------------------------------------------------------------------------------
# 1. Bootstrap 1.000 lần trên tập test
# ------------------------------------------------------------------------------
idx = boot_indices(yte)
B = {m: bootstrap_metrics(yte, p_te[m], idx) for m in MODELS}
F2B = {m: np.array([fbeta_score(yte[i], (p_te[m][i] >= thr[m]).astype(int), beta=2, zero_division=0)
                    for i in idx]) for m in MODELS}
rows = []
for m in ORDER:
    rows.append({
        "Mô hình": m + (" (mô hình cuối)" if m == final else ""), "Chiến lược": cfg[m]["strategy"],
        "ROC-AUC (test)": roc_auc_score(yte, p_te[m]),
        "ROC-AUC Mean ± SD": f"{B[m][0].mean():.4f} ± {B[m][0].std():.4f}", "ROC-AUC 95% CI": ci95(B[m][0], 4),
        "PR-AUC (test)": average_precision_score(yte, p_te[m]),
        "PR-AUC Mean ± SD": f"{B[m][1].mean():.4f} ± {B[m][1].std():.4f}", "PR-AUC 95% CI": ci95(B[m][1], 4),
        "Lift@10% (test)": lift_at_k(yte, p_te[m], .10), "Lift@10% 95% CI": ci95(B[m][2], 2),
        "Ngưỡng F2 (OOF train)": thr[m],
        "F2 (test)": fbeta_score(yte, (p_te[m] >= thr[m]).astype(int), beta=2, zero_division=0),
        "F2 95% CI": ci95(F2B[m], 4),
        "P(ROC-AUC ≤ 0,5)": float((B[m][0] <= 0.5).mean())})
t1 = pd.DataFrame(rows)
print("\n=== Bảng 5.8a: bootstrap 1.000 lần trên tập test ===")
print(t1.round(4).to_string(index=False))
t1.to_csv(OUT / "T5_8_1_bootstrap_test_ci.csv", index=False, encoding="utf-8-sig")

drows = []
for m in ORDER[1:]:
    for k, nm, dig in ((0, "ROC-AUC", 4), (1, "PR-AUC", 5), (2, "Lift@10%", 3)):
        d = B[final][k] - B[m][k]
        drows.append({"So sánh": f"{final} − {m}", "Chỉ số": nm, "Δ (điểm đơn)":
                      [roc_auc_score(yte, p_te[final]) - roc_auc_score(yte, p_te[m]),
                       average_precision_score(yte, p_te[final]) - average_precision_score(yte, p_te[m]),
                       lift_at_k(yte, p_te[final], .1) - lift_at_k(yte, p_te[m], .1)][k],
                      "Δ 95% CI (bootstrap ghép cặp)": ci95(d, dig), "P(Δ ≤ 0)": float((d <= 0).mean()),
                      "Kết luận": ("khác biệt (CI không chứa 0)" if np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0
                                   else "chưa phân biệt được (CI chứa 0)")})
t1b = pd.DataFrame(drows)
print("\n=== Bảng 5.8b: chênh lệch ghép cặp ===")
print(t1b.round(5).to_string(index=False))
t1b.to_csv(OUT / "T5_8_1b_paired_differences.csv", index=False, encoding="utf-8-sig")

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, k, ttl, base in ((axs[0], 0, "ROC-AUC", .5), (axs[1], 1, "PR-AUC", yte.mean())):
    for m in ORDER:
        ax.hist(B[m][k], bins=40, alpha=.45, color=COL[m], density=True,
                label=f"{m} (TB {B[m][k].mean():.4f})")
    ax.axvline(base, color="k", ls="--", lw=1, label="Mức ngẫu nhiên" if k == 0 else f"Mức nền ({base:.4f})")
    ax.set_title(f"Phân phối bootstrap {ttl} ({len(idx):,} lần)"); ax.set_xlabel(ttl); ax.set_ylabel("Mật độ")
    ax.legend(fontsize=8)
for m in ORDER[1:]:
    axs[2].hist(B[final][1] - B[m][1], bins=40, alpha=.5, color=COL[m], density=True, label=f"{final} − {m}")
axs[2].axvline(0, color="red", ls="--", lw=1.4)
axs[2].set_title("Chênh lệch ghép cặp PR-AUC với mô hình cuối"); axs[2].set_xlabel("Δ PR-AUC"); axs[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "H5_8a_bootstrap_distribution.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------
# 2. Phân nhóm trong tập test (dùng lại mô hình đã huấn luyện)
# ------------------------------------------------------------------------------
seg = test["segment"].to_numpy()
groups = {"Toàn bộ tập test": np.ones(len(test), bool),
          "Đã nhận email (Mens + Womens)": np.isin(seg, ARMS),
          "Nhận Mens E-Mail": seg == "Mens E-Mail",
          "Nhận Womens E-Mail": seg == "Womens E-Mail",
          "Không nhận email (No E-Mail)": seg == CONTROL}
srows, sci = [], {}
for g, mask in groups.items():
    yg = yte[mask]
    ig = boot_indices(yg)
    for m in ORDER:
        pg = p_te[m][mask]
        roc_b, ap_b, l10_b = bootstrap_metrics(yg, pg, ig)
        ap = average_precision_score(yg, pg)
        sci[(g, m)] = (ap, np.percentile(ap_b, 2.5), np.percentile(ap_b, 97.5), yg.mean())
        srows.append({"Phân nhóm": g, "Mô hình": m, "Số khách": int(mask.sum()), "Số người mua": int(yg.sum()),
                      "Tỷ lệ mua nền": yg.mean(),
                      "ROC-AUC": roc_auc_score(yg, pg), "ROC-AUC 95% CI": ci95(roc_b, 3),
                      "PR-AUC": ap, "PR-AUC 95% CI": ci95(ap_b, 4), "PR-AUC / mức nền": ap / yg.mean(),
                      "Lift@10%": lift_at_k(yg, pg, .10), "Lift@10% 95% CI": ci95(l10_b, 2)})
t2 = pd.DataFrame(srows)
print("\n=== Bảng 5.8c: độ bền theo phân nhóm trong tập test ===")
print(t2.round(4).to_string(index=False))
t2.to_csv(OUT / "T5_8_2_robustness_subgroups.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(11, 5))
names = list(groups)
x = np.arange(len(names)); w = .8 / len(ORDER)
for k, m in enumerate(ORDER):
    v = np.array([sci[(g, m)][0] for g in names])
    lo = v - np.array([sci[(g, m)][1] for g in names]); hi = np.array([sci[(g, m)][2] for g in names]) - v
    ax.bar(x + (k - (len(ORDER) - 1) / 2) * w, v, w, color=COL[m], alpha=.85, yerr=[lo, hi], capsize=2, label=m)
ax.plot(x, [sci[(g, final)][3] for g in names], "k_", ms=34, mew=2, label="Tỷ lệ mua nền")
ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8.5)
ax.set_ylabel("PR-AUC"); ax.set_title("PR-AUC theo phân nhóm trong tập test (CI 95% bootstrap)"); ax.legend(fontsize=8, ncol=5)
fig.tight_layout(); fig.savefig(OUT / "H5_8b_subgroup_pr_auc.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------
# 3. Độ nhạy với bản ghi trùng lặp: huấn luyện lại trên bản đã khử trùng
# ------------------------------------------------------------------------------
dtr, dte = load_split("_deduplicated_sensitivity")
ydtr, ydte = dtr[TARGET].astype(int).to_numpy(), dte[TARGET].astype(int).to_numpy()
print(f"\nBản khử trùng: train {len(dtr):,} ({ydtr.sum()} người mua) | test {len(dte):,} ({ydte.sum()} người mua)")
idd = boot_indices(ydte)
drows, dstore = [], {}
for m in ORDER:
    t0 = time.time()
    e = refit(est[m], m, cfg[m]["strategy"], dtr[FS2], ydtr)
    pd_ = e.predict_proba(dte[FS2])[:, 1]
    r_b, a_b, l_b = bootstrap_metrics(ydte, pd_, idd)
    dstore[m] = (pd_, a_b)
    for nm, y_, p_, rb, ab, lb in (("Mẫu chính (64.000 dòng)", yte, p_te[m], *B[m]),
                                    ("Bản khử trùng (57.438 dòng)", ydte, pd_, r_b, a_b, l_b)):
        drows.append({"Mô hình": m, "Dữ liệu": nm, "Quy mô test": len(y_), "Người mua (test)": int(y_.sum()),
                      "Tỷ lệ mua nền": y_.mean(),
                      "ROC-AUC": roc_auc_score(y_, p_), "ROC-AUC 95% CI": ci95(rb, 3),
                      "PR-AUC": average_precision_score(y_, p_), "PR-AUC 95% CI": ci95(ab, 4),
                      "Lift@10%": lift_at_k(y_, p_, .10), "Lift@10% 95% CI": ci95(lb, 2)})
    print(f"  {m}: huấn luyện lại + đánh giá {time.time() - t0:.0f}s")
t3 = pd.DataFrame(drows)
print("\n=== Bảng 5.8d: mẫu chính và bản khử trùng ===")
print(t3.round(4).to_string(index=False))
t3.to_csv(OUT / "T5_8_3_main_vs_deduplicated.csv", index=False, encoding="utf-8-sig")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, key, ttl, base in ((axs[0], "ROC-AUC", "ROC-AUC", .5), (axs[1], "PR-AUC", "PR-AUC", None)):
    for k, (dn, col) in enumerate((("Mẫu chính (64.000 dòng)", "tab:blue"), ("Bản khử trùng (57.438 dòng)", "tab:gray"))):
        sub = t3[t3["Dữ liệu"] == dn].set_index("Mô hình").loc[ORDER]
        ax.bar(np.arange(len(ORDER)) + (k - .5) * .38, sub[key], .38, color=col, alpha=.85, label=dn)
    if base:
        ax.axhline(base, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(ORDER)), ORDER); ax.set_title(f"{ttl}: mẫu chính và bản khử trùng"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "H5_8c_main_vs_deduplicated.png", bbox_inches="tight"); plt.close(fig)

(OUT / "robustness_summary.json").write_text(json.dumps({
    "final_model": final, "n_boot": len(idx), "thresholds_f2_oof_train": thr,
    "strategies": {m: cfg[m]["strategy"] for m in MODELS}, "strategy_consistent_with_5_3": not bad,
    "test_n": int(len(yte)), "test_positives": int(yte.sum()),
    "dedup_test_n": int(len(ydte)), "dedup_test_positives": int(ydte.sum())}, ensure_ascii=False, indent=2))
print("\nĐã lưu:", sorted(p.name for p in OUT.iterdir()))
if bad:
    print("LƯU Ý: chiến lược FS2 (5.4) chưa khớp 5.3 ở", bad, "- chạy lại 03 rồi chạy lại 06, 07, 08.")
