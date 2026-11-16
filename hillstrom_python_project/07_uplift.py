# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys
import json
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

import common

# ==============================================================================
# 0. CẤU HÌNH
# ==============================================================================
SEED = 42
N_BOOT = 1000
TARGET = "conversion"
CONTROL = "No E-Mail"
ARMS = ["Mens E-Mail", "Womens E-Mail"]

PROC_DIR = Path("data") / "processed"
FIG_DIR = Path("figures")
OUT = FIG_DIR / "5_7"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Đặc trưng TRƯỚC can thiệp (FS1 + 2 biến phái sinh không phụ thuộc email)
PRE = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel",
       "recency_group", "both_gender"]
FS2 = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel",
       "segment", "promo_match", "recency_group", "both_gender"]
LEAK = {"visit", "spend", "history_segment", "history", "row_id", TARGET}
assert not (set(PRE) & (LEAK | {"segment", "promo_match"})), "PRE chứa biến rò rỉ/hậu can thiệp"
assert not (set(FS2) & LEAK), "FS2 chứa biến rò rỉ"

NUM_COLS = ["recency", "log_history"]
CAT_LEVELS = {
    "zip_code": ["Rural", "Suburban", "Urban"],
    "channel": ["Multichannel", "Phone", "Web"],
    "recency_group": ["1-3m", "4-6m", "7-9m", "10-12m"],
    "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
}

# Siêu tham số lấy từ mục 5.4 (figures/FS2_best_params.json); không tinh chỉnh lại theo
# tiêu chí uplift để tránh thêm một vòng lựa chọn trên dữ liệu ít người mua.
RF_PARAMS = dict(n_estimators=300, max_depth=6, min_samples_leaf=100, max_features="sqrt")
GB_PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.7)
bp_file = FIG_DIR / "FS2_best_params.json"
if bp_file.exists():
    bp = json.load(open(bp_file, encoding="utf-8"))
    RF_PARAMS.update(bp.get("RF", {}).get("best_params", {}))
    GB_PARAMS.update(bp.get("GB", {}).get("best_params", {}))

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})
# Mô hình phản hồi = mô hình cuối của mục 5.9
_meta = json.load(open(FIG_DIR / "5_9" / "final_model_meta.json", encoding="utf-8"))
FINAL_M, FINAL_STRAT = _meta["model"], _meta["strategy"]
FINAL_EST = joblib.load(MODEL_DIR / "final_model_FS2.joblib")
RESP_LABEL = f"Mô hình phản hồi {FINAL_M} (chọn theo p̂)"

print("scikit-learn", sklearn.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)

# ==============================================================================
# 1. NẠP DỮ LIỆU, BIẾN ĐIỀU TRỊ VÀ KIỂM TRA CÂN BẰNG NGẪU NHIÊN
# ==============================================================================
train = pd.read_csv(PROC_DIR / "train.csv")
test = pd.read_csv(PROC_DIR / "test.csv")
for nm, d in [("train", train), ("test", test)]:
    assert set(FS2 + [TARGET, "row_id"]) <= set(d.columns), f"Thiếu cột trong {nm}.csv"
    assert not ({"visit", "spend", "history_segment"} & set(d.columns)), f"Rò rỉ trong {nm}.csv"
    d["T"] = (d["segment"] != CONTROL).astype(int)
assert set(train["row_id"]).isdisjoint(test["row_id"])
y, t = train[TARGET].to_numpy(), train["T"].to_numpy()


def counts(d, label):
    g = d.groupby("segment")[TARGET].agg(["size", "sum", "mean"]).reindex([CONTROL] + ARMS)
    g.columns = ["Số khách", "Người mua", "Tỷ lệ mua"]
    g.loc["Nhận email (T=1)"] = [d["T"].sum(), d.loc[d["T"] == 1, TARGET].sum(),
                                 d.loc[d["T"] == 1, TARGET].mean()]
    g = g.reset_index().rename(columns={"segment": "Nhóm"})
    g.insert(0, "Tập", label)
    return g


t571 = pd.concat([counts(train, "Train"), counts(test, "Test")], ignore_index=True)
display(t571)
t571.to_csv(OUT / "T5_7_1_treatment_control_counts.csv", index=False, encoding="utf-8-sig")


def smd_table(d):
    Z = pd.get_dummies(d[PRE], columns=["zip_code", "channel", "recency_group"], dtype=float)
    rows = []
    for c in Z.columns:
        a, b = Z.loc[d["T"] == 1, c], Z.loc[d["T"] == 0, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"Biến": c, "Trung bình T=1": a.mean(), "Trung bình T=0": b.mean(),
                     "SMD": (a.mean() - b.mean()) / sd if sd > 0 else 0.0})
    return pd.DataFrame(rows)


smd = smd_table(train)
smd.to_csv(OUT / "T5_7_2_covariate_balance_smd.csv", index=False, encoding="utf-8-sig")
print(f"\nCân bằng hiệp biến T=1 vs T=0 (train): |SMD| lớn nhất = {smd['SMD'].abs().max():.4f} "
      f"(ngưỡng tham khảo 0,1)")


# ==============================================================================
# 2. HÀM TIỆN ÍCH: ATE, QINI, BOOTSTRAP
# ==============================================================================
def ate(y, t):
    return y[t == 1].mean() - y[t == 0].mean()


def strat_boot_idx(t, rng):
    i1, i0 = np.where(t == 1)[0], np.where(t == 0)[0]
    return np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]


def ate_ci(y, t, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(n_boot):
        i = strat_boot_idx(t, rng)
        est.append(ate(y[i], t[i]))
    return np.percentile(est, [2.5, 97.5])


def qini_curve(y, t, score):
    """Đường Qini (Radcliffe, 2007): Q(n) = Y_T(n) − Y_C(n)·N_T(n)/N_C(n).
    Trả về φ (tỷ lệ khách được chọn, 0..1) và Q (số đơn tăng thêm, quy mô nhóm điều trị)."""
    order = np.argsort(-np.asarray(score), kind="mergesort")
    y, t = np.asarray(y)[order], np.asarray(t)[order]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q = np.where(nc > 0, yt - yc * nt / np.maximum(nc, 1), 0.0)
    phi = np.arange(1, len(y) + 1) / len(y)
    return np.r_[0.0, phi], np.r_[0.0, q]


def qini_coef(y, t, score):
    """Hệ số Qini = diện tích giữa đường Qini và đường ngẫu nhiên, chia cho N_T
    (đơn vị: tỷ lệ mua tăng thêm). > 0 nghĩa là xếp hạng tốt hơn chọn ngẫu nhiên."""
    phi, q = qini_curve(y, t, score)
    return (getattr(np, "trapezoid", None) or np.trapz)(q - phi * q[-1], phi) / max(int(np.sum(t)), 1)


def q_at(y, t, score, k):
    phi, q = qini_curve(y, t, score)
    return q[int(round(k * (len(phi) - 1)))]


def bootstrap_qini(y, t, scores: dict, n_boot=N_BOOT, seed=SEED):
    """Bootstrap phân tầng theo T; dùng CHUNG mẫu cho mọi chiến lược (so sánh theo cặp)."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in scores}
    for _ in range(n_boot):
        i = strat_boot_idx(t, rng)
        for k, s in scores.items():
            out[k].append(qini_coef(y[i], t[i], s[i]))
    return {k: np.asarray(v) for k, v in out.items()}


def rate_diff_ci(y, t, z=1.96):
    y1, y0 = y[t == 1], y[t == 0]
    if len(y1) == 0 or len(y0) == 0:
        return np.nan, np.nan, np.nan
    d = y1.mean() - y0.mean()
    se = np.sqrt(y1.mean() * (1 - y1.mean()) / len(y1) + y0.mean() * (1 - y0.mean()) / len(y0))
    return d, d - z * se, d + z * se


def qini_table(y, t, scores, boot):
    rows = []
    for k, s in scores.items():
        q1 = q_at(y, t, s, 1.0)
        rows.append({"Chiến lược xếp hạng": k, "Hệ số Qini": qini_coef(y, t, s),
                     "Qini 95% CI": f"[{np.percentile(boot[k], 2.5):.5f}; {np.percentile(boot[k], 97.5):.5f}]",
                     "P(Qini ≤ 0)": (boot[k] <= 0).mean(),
                     **{f"Q@{int(p*100)}%": q_at(y, t, s, p) for p in (.1, .2, .3, .5)},
                     **{f"Tỷ lệ đơn tăng thêm @{int(p*100)}%": q_at(y, t, s, p) / q1 for p in (.2, .3, .5)},
                     "Q@100% (gửi tất cả)": q1})
    return pd.DataFrame(rows)


# ==============================================================================
# 3. S-LEARNER
# ==============================================================================
def make_pre(features):
    num = [c for c in features if c in NUM_COLS]
    cat = [c for c in features if c in CAT_LEVELS]
    binary = [c for c in features if c not in num + cat]
    return ColumnTransformer(
        [("num", StandardScaler(), num), ("bin", "passthrough", binary),
         ("cat", OneHotEncoder(categories=[CAT_LEVELS[c] for c in cat], drop="first",
                               sparse_output=False), cat)], remainder="drop")


class SLearner:
    """Một mô hình duy nhất μ(x, T). Với Logistic Regression, thêm tương tác T × x để uplift
    có thể khác nhau theo đặc điểm khách hàng (nếu không, LR chỉ cộng một hằng số vào
    log-odds và τ̂ chỉ phụ thuộc vào mức xác suất nền)."""

    def __init__(self, base, interactions=False):
        self.base, self.interactions = base, interactions

    def _design(self, X, t):
        Z = self.pre_.transform(X[PRE])
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        return np.hstack([Z, t] + ([Z * t] if self.interactions else []))

    def fit(self, X, t, y):
        self.pre_ = make_pre(PRE).fit(X[PRE])
        self.model_ = clone(self.base).fit(self._design(X, t), y)
        return self

    def predict_mu(self, X, t):
        return self.model_.predict_proba(self._design(X, np.full(len(X), t)))[:, 1]

    def predict_uplift(self, X):
        return self.predict_mu(X, 1) - self.predict_mu(X, 0)


BASES = {
    "LR (tương tác T×x)": (LogisticRegression(C=1.0, max_iter=3000, random_state=SEED), True),
    "Random Forest": (RandomForestClassifier(**RF_PARAMS, n_jobs=-1, random_state=SEED), False),
    "Gradient Boosting": (GradientBoostingClassifier(**GB_PARAMS, random_state=SEED), False),
}


def fit_response(X, y_):
    """Huấn luyện lại mô hình cuối của 5.9 (cùng siêu tham số và chiến lược mất cân bằng)."""
    return common.refit(FINAL_EST, FINAL_M, FINAL_STRAT, X[FS2], y_)


def response_score(est, X):
    """Điểm xếp hạng của mô hình phản hồi: xác suất mua dự đoán NẾU khách được gửi email, trung bình
    của hai loại email (segment đặt giả định, promo_match tính lại). Không dùng segment thực tế vì nó
    chính là biến điều trị T: nếu dùng, p̂ cao sẽ dồn vào người đã nhận email và đường Qini bị méo
    (nhóm đầu gần như chỉ có khách T = 1)."""
    out = []
    for a in ARMS:
        Z = X[FS2].copy()
        Z["segment"] = a
        Z["promo_match"] = (((a == "Mens E-Mail") & (Z["mens"] == 1)) |
                            ((a == "Womens E-Mail") & (Z["womens"] == 1))).astype(int)
        out.append(est.predict_proba(Z)[:, 1])
    return np.mean(out, axis=0)


# ==============================================================================
# 4. DỰ ĐOÁN OUT-OF-FOLD VÀ CHỌN BASE LEARNER THEO QINI
# ==============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
strata = 2 * t + y                      # phân tầng đồng thời theo T và nhãn mua
fold_idx = list(cv.split(train, strata))
for a, b in fold_idx:
    assert y[b][t[b] == 0].sum() > 0, "Fold không có người mua trong nhóm đối chứng"

tau_oof = {n: np.zeros(len(train)) for n in BASES}
mu1_oof = {n: np.zeros(len(train)) for n in BASES}
mu0_oof = {n: np.zeros(len(train)) for n in BASES}
fold_rows = []
for name, (base, inter) in BASES.items():
    t0 = time.time()
    for k, (a, b) in enumerate(fold_idx, 1):
        m = SLearner(base, inter).fit(train.iloc[a], t[a], y[a])
        Xb = train.iloc[b]
        mu1_oof[name][b], mu0_oof[name][b] = m.predict_mu(Xb, 1), m.predict_mu(Xb, 0)
        tau_oof[name][b] = mu1_oof[name][b] - mu0_oof[name][b]
        fold_rows.append({"Base learner": name, "Fold": k,
                          "Qini (fold)": qini_coef(y[b], t[b], tau_oof[name][b])})
    print(f"[CV] {name}: {time.time() - t0:.0f}s")

p_resp = np.zeros(len(train))
for k, (a, b) in enumerate(fold_idx, 1):
    p_resp[b] = response_score(fit_response(train.iloc[a], y[a]), train.iloc[b])
    fold_rows.append({"Base learner": RESP_LABEL, "Fold": k,
                      "Qini (fold)": qini_coef(y[b], t[b], p_resp[b])})

folds = pd.DataFrame(fold_rows)
cv_tab = (folds.groupby("Base learner")["Qini (fold)"].agg(["mean", "std"])
          .rename(columns={"mean": "Qini theo fold (mean)", "std": "Qini theo fold (std)"}))
all_scores = {**tau_oof, RESP_LABEL: p_resp}
cv_tab["Qini OOF (gộp 5 fold)"] = [qini_coef(y, t, all_scores[n]) for n in cv_tab.index]
cv_tab["Số fold Qini > 0"] = folds.assign(pos=folds["Qini (fold)"] > 0).groupby(
    "Base learner")["pos"].sum()
cv_tab["τ̂ trung bình"] = [tau_oof[n].mean() if n in tau_oof else np.nan for n in cv_tab.index]
cv_tab["Tỷ lệ τ̂ < 0"] = [(tau_oof[n] < 0).mean() if n in tau_oof else np.nan for n in cv_tab.index]
cv_tab = cv_tab.sort_values("Qini theo fold (mean)", ascending=False)
BEST = [n for n in cv_tab.index if n in BASES][0]
cv_tab["Được chọn"] = np.where(cv_tab.index == BEST, "✔", "")
display(cv_tab.round(5))
cv_tab.reset_index().to_csv(OUT / "T5_7_3_cv_qini_base_learners.csv", index=False,
                            encoding="utf-8-sig")
folds.to_csv(OUT / "T5_7_3b_cv_qini_by_fold.csv", index=False, encoding="utf-8-sig")
print(f"\nBase learner được chọn (Qini trung bình theo fold cao nhất): {BEST}")
tau = tau_oof[BEST]
base, inter = BASES[BEST]

# ==============================================================================
# 5. ATE QUAN SÁT VÀ ĐƯỜNG QINI (TRAIN, OUT-OF-FOLD)
# ==============================================================================
ci = ate_ci(y, t)
rand = np.random.default_rng(SEED).random(len(train))
scores = {"S-learner": tau, RESP_LABEL: p_resp, "Ngẫu nhiên": rand}
boot = bootstrap_qini(y, t, scores)
t575 = qini_table(y, t, scores, boot)
diff = boot["S-learner"] - boot[RESP_LABEL]
qdiff = {"mean": float(diff.mean()), "ci_low": float(np.percentile(diff, 2.5)),
         "ci_high": float(np.percentile(diff, 97.5)), "P(diff<=0)": float((diff <= 0).mean())}
display(t575.round(4))
t575.to_csv(OUT / "T5_7_5_qini_oof.csv", index=False, encoding="utf-8-sig")
print(f"Chênh lệch Qini (S-learner − phản hồi), bootstrap theo cặp: {qdiff['mean']:.5f} "
      f"[{qdiff['ci_low']:.5f}; {qdiff['ci_high']:.5f}], P(≤0) = {qdiff['P(diff<=0)']:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
for k, col in [("S-learner", "tab:blue"), (RESP_LABEL, "tab:red")]:
    phi, q = qini_curve(y, t, scores[k])
    lab = f"S-learner – {BEST}" if k == "S-learner" else k
    ax.plot(phi * 100, q, color=col, lw=1.8, label=f"{lab} (Qini = {qini_coef(y, t, scores[k]):.4f})")
phi, q = qini_curve(y, t, tau)
ax.plot([0, 100], [0, q[-1]], "--", color="black", lw=1.2,
        label=f"Chọn ngẫu nhiên (Q@100% = {q[-1]:.0f} đơn)")
ax.axhline(0, color="grey", lw=.8)
ax.set_xlabel("Tỷ lệ khách hàng được chọn theo điểm (%)"); ax.set_ylabel("Số đơn tăng thêm ước tính, Q(φ)")
ax.set_title("(a) Đường Qini – train, dự đoán out-of-fold"); ax.legend(loc="upper left", fontsize=8.3)
ax = axes[1]
for name, col in zip(BASES, ["tab:green", "tab:purple", "tab:orange"]):
    phi, q = qini_curve(y, t, tau_oof[name])
    ax.plot(phi * 100, q, color=col, lw=1.6, label=f"{name} (Qini = {qini_coef(y, t, tau_oof[name]):.4f})")
ax.plot([0, 100], [0, q[-1]], "--", color="black", lw=1.2, label="Chọn ngẫu nhiên")
ax.axhline(0, color="grey", lw=.8)
ax.set_xlabel("Tỷ lệ khách hàng được chọn theo τ̂ (%)")
ax.set_title("(b) So sánh ba base learner của S-learner"); ax.legend(loc="upper left", fontsize=8.3)
fig.tight_layout(); fig.savefig(OUT / "H5_7a_qini_curves.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4))
for k, col in [("S-learner", "tab:blue"), (RESP_LABEL, "tab:red"), ("Ngẫu nhiên", "grey")]:
    ax.hist(boot[k], bins=40, alpha=.45, color=col, label=k)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Hệ số Qini (1.000 mẫu bootstrap phân tầng theo T)"); ax.set_ylabel("Tần suất")
ax.set_title("Độ bất định của hệ số Qini (train, out-of-fold)"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "H5_7b_qini_bootstrap.png", bbox_inches="tight"); plt.close(fig)


# ==============================================================================
# 6. CHIA KHÁCH HÀNG THEO MỨC UPLIFT
# ==============================================================================
def bucket_table(y, t, tau, labels, order):
    out = []
    for gname in order:
        m = labels == gname
        d, lo, hi = rate_diff_ci(y[m], t[m])
        out.append({"Nhóm": gname, "Số khách": int(m.sum()), "N (T=1)": int(t[m].sum()),
                    "N (T=0)": int((1 - t[m]).sum()),
                    "Người mua T=1": int(y[m & (t == 1)].sum()),
                    "Người mua T=0": int(y[m & (t == 0)].sum()),
                    "Tỷ lệ mua T=1": y[m & (t == 1)].mean(), "Tỷ lệ mua T=0": y[m & (t == 0)].mean(),
                    "Uplift quan sát": d, "CI thấp": lo, "CI cao": hi, "τ̂ trung bình": tau[m].mean()})
    return pd.DataFrame(out)


def deciles(v):
    return np.ceil(pd.Series(v).rank(method="first", ascending=False) / len(v) * 10).astype(int).to_numpy()


dec = deciles(tau)
t576 = bucket_table(y, t, tau, dec, range(1, 11))
display(t576.round(4))
t576.to_csv(OUT / "T5_7_6_uplift_by_decile.csv", index=False, encoding="utf-8-sig")
rho_dec = spearmanr(t576["τ̂ trung bình"], t576["Uplift quan sát"])

fig, ax = plt.subplots(figsize=(8, 4.4))
x = t576["Nhóm"].to_numpy()
ax.bar(x, t576["Uplift quan sát"] * 100, color="tab:blue", alpha=.7, label="Uplift quan sát (T=1 − T=0)")
ax.errorbar(x, t576["Uplift quan sát"] * 100,
            yerr=[(t576["Uplift quan sát"] - t576["CI thấp"]) * 100,
                  (t576["CI cao"] - t576["Uplift quan sát"]) * 100],
            fmt="none", ecolor="black", capsize=3, lw=1)
ax.plot(x, t576["τ̂ trung bình"] * 100, "o-", color="tab:orange", label="τ̂ dự đoán trung bình")
ax.axhline(ate(y, t) * 100, ls="--", color="grey", label=f"ATE toàn train = {ate(y, t)*100:.2f} điểm %")
ax.axhline(0, color="black", lw=.8)
ax.set_xticks(range(1, 11)); ax.set_xlabel("Decile theo τ̂ (1 = uplift dự đoán cao nhất)")
ax.set_ylabel("Chênh lệch tỷ lệ mua (điểm %)")
ax.set_title("Uplift quan sát và dự đoán theo decile (train OOF, KTC 95%)"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "H5_7c_uplift_by_decile.png", bbox_inches="tight"); plt.close(fig)

# Bốn nhóm hành động theo phân vị τ̂ OOF trên train: A top 20%, B 20–50%, C 50–80%,
# D 20% thấp nhất. Ngưỡng τ̂ được lưu lại và áp nguyên cho test / khách hàng mới.
q80, q50, q20 = np.quantile(tau, [.8, .5, .2])
TIERS = ["A. Uplift cao", "B. Uplift khá", "C. Uplift thấp", "D. Uplift ≈ 0"]


def tier_of(v):
    return np.select([v >= q80, v >= q50, v >= q20], TIERS[:3], TIERS[3])


tier = tier_of(tau)
t577 = bucket_table(y, t, tau, tier, TIERS)
t577["Tỷ trọng khách"] = t577["Số khách"] / len(train)
t577["Đơn tăng thêm ước tính nếu gửi cả nhóm"] = t577["Uplift quan sát"] * t577["Số khách"]
display(t577.round(4))
t577.to_csv(OUT / "T5_7_7_uplift_tiers.csv", index=False, encoding="utf-8-sig")
tier_cut = {"A (τ̂ ≥)": float(q80), "B (τ̂ ≥)": float(q50), "C (τ̂ ≥)": float(q20)}
print("Ngưỡng τ̂:", {k: round(v, 5) for k, v in tier_cut.items()})

prof_df = train.assign(tier=tier, tau=tau, history=np.expm1(train["log_history"]))
prof = prof_df.groupby("tier").agg(
    **{"Số khách": ("row_id", "size"), "τ̂ trung bình": ("tau", "mean"),
       "τ̂ min": ("tau", "min"), "τ̂ max": ("tau", "max"),
       "history TB (USD)": ("history", "mean"), "history trung vị": ("history", "median"),
       "recency TB (tháng)": ("recency", "mean"), "Tỷ lệ newbie": ("newbie", "mean"),
       "Tỷ lệ mua đồ nam": ("mens", "mean"), "Tỷ lệ mua đồ nữ": ("womens", "mean"),
       "Tỷ lệ both_gender": ("both_gender", "mean")}).reindex(TIERS)
for c in ["Web", "Phone", "Multichannel"]:
    prof[f"Kênh {c}"] = prof_df.assign(v=prof_df["channel"].eq(c)).groupby("tier")["v"].mean()
for z in ["Urban", "Suburban", "Rural"]:
    prof[f"Khu vực {z}"] = prof_df.assign(v=prof_df["zip_code"].eq(z)).groupby("tier")["v"].mean()
display(prof.round(3).T)
prof.reset_index().rename(columns={"tier": "Nhóm"}).to_csv(
    OUT / "T5_7_8_tier_profiles.csv", index=False, encoding="utf-8-sig")

short = [s.split(". ")[0] for s in TIERS]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
ax.bar(np.arange(4) - .19, t577["Tỷ lệ mua T=0"] * 100, .38, color="tab:gray", alpha=.8, label="Không nhận (T=0)")
ax.bar(np.arange(4) + .19, t577["Tỷ lệ mua T=1"] * 100, .38, color="tab:blue", alpha=.8, label="Nhận email (T=1)")
for i, r in t577.iterrows():
    ax.text(i, max(r["Tỷ lệ mua T=1"], r["Tỷ lệ mua T=0"]) * 100 + .05,
            f"{r['Uplift quan sát']*100:+.2f}", ha="center", fontsize=8.5)
ax.set_ylim(0, max(t577["Tỷ lệ mua T=1"].max(), t577["Tỷ lệ mua T=0"].max()) * 100 * 1.25)
ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_ylabel("Tỷ lệ mua (%)")
ax.set_title("(a) Tỷ lệ mua theo nhóm (số: uplift, điểm %)"); ax.legend(fontsize=8)
ax = axes[1]
ax.bar(range(4), prof["history TB (USD)"], color="tab:purple", alpha=.75)
ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_title("(b) Chi tiêu lịch sử trung bình (USD)")
ax = axes[2]
bottom = np.zeros(4)
for c, col in zip(["Kênh Web", "Kênh Phone", "Kênh Multichannel"], ["tab:cyan", "tab:orange", "tab:gray"]):
    ax.bar(range(4), prof[c] * 100, bottom=bottom, color=col, alpha=.8, label=c[5:])
    bottom += prof[c].to_numpy() * 100
ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_title("(c) Cơ cấu kênh mua (%)")
ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Bốn nhóm khách hàng theo uplift dự đoán (A: top 20%, B: 20–50%, C: 50–80%, D: 20% thấp nhất)",
             y=1.02)
fig.tight_layout(); fig.savefig(OUT / "H5_7d_uplift_tiers.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.hist(tau * 100, bins=60, color="tab:blue", alpha=.75)
for v, s in [(q80, "A"), (q50, "B"), (q20, "C")]:
    ax.axvline(v * 100, color="black", ls=":", lw=1)
    ax.text(v * 100, ax.get_ylim()[1] * .92, f" {s}")
ax.axvline(ate(y, t) * 100, color="tab:red", ls="--", label=f"ATE quan sát = {ate(y, t)*100:.2f} điểm %")
ax.axvline(0, color="black", lw=.8)
ax.set_xlabel("τ̂ (điểm %)"); ax.set_ylabel("Số khách")
ax.set_title("Phân phối uplift dự đoán (train, out-of-fold)"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "H5_7e_tau_distribution.png", bbox_inches="tight"); plt.close(fig)

# ==============================================================================
# 7. PHÂN TÍCH BỔ SUNG: UPLIFT THEO LOẠI EMAIL
# ==============================================================================


def with_segment(X, seg):
    Z = X.copy()
    Z["segment"] = seg
    Z["promo_match"] = (((seg == "Mens E-Mail") & (Z["mens"] == 1)) |
                        ((seg == "Womens E-Mail") & (Z["womens"] == 1))).astype(int)
    return Z


class MultiArmSLearner:
    """S-learner đa nhánh: segment (3 mức) là biến điều trị, promo_match tính lại theo segment giả định.
    Thiết kế: [B, S, promo_match] (+ B × S_j nếu có tương tác), với B = đặc trưng trước can thiệp (PRE
    đã mã hóa), S = 2 cột one-hot của segment (mức tham chiếu No E-Mail)."""

    def __init__(self, base, interactions=False):
        self.base, self.interactions = base, interactions

    def _design(self, X):
        B = self.pre_.transform(X[PRE])
        S = np.column_stack([(X["segment"] == a).to_numpy(float) for a in ARMS])
        pm = X[["promo_match"]].to_numpy(float)
        parts = [B, S, pm]
        if self.interactions:
            parts += [B * S[:, [j]] for j in range(S.shape[1])]
        return np.hstack(parts)

    def fit(self, X, y):
        self.pre_ = make_pre(PRE).fit(X[PRE])
        self.model_ = clone(self.base).fit(self._design(X), y)
        return self

    def mu(self, X, seg):
        return self.model_.predict_proba(self._design(with_segment(X, seg)))[:, 1]


tau_arm = {a: np.zeros(len(train)) for a in ARMS}
for a_idx, b_idx in fold_idx:
    ma = MultiArmSLearner(base, inter).fit(train.iloc[a_idx], y[a_idx])
    Xb = train.iloc[b_idx]
    mc = ma.mu(Xb, CONTROL)
    for a in ARMS:
        tau_arm[a][b_idx] = ma.mu(Xb, a) - mc
best_arm = np.where(tau_arm["Mens E-Mail"] >= tau_arm["Womens E-Mail"], "Mens E-Mail", "Womens E-Mail")

arm_rows, arm_curves = [], {}
for a in ARMS:
    m = train["segment"].isin([a, CONTROL]).to_numpy()
    ya, ta = y[m], (train.loc[m, "segment"] == a).astype(int).to_numpy()
    d, lo, hi = rate_diff_ci(ya, ta)
    bq = bootstrap_qini(ya, ta, {"s": tau_arm[a][m]})["s"]
    arm_curves[a] = (ya, ta, tau_arm[a][m])
    arm_rows.append({"Loại email": a, "Số khách (nhánh + đối chứng)": int(m.sum()),
                     "ATE quan sát": d, "ATE 95% CI": f"[{lo:.4f}; {hi:.4f}]",
                     "τ̂ trung bình": tau_arm[a].mean(),
                     "Qini (nhánh vs No E-Mail)": qini_coef(ya, ta, tau_arm[a][m]),
                     "Qini 95% CI": f"[{np.percentile(bq, 2.5):.5f}; {np.percentile(bq, 97.5):.5f}]",
                     "P(Qini ≤ 0)": (bq <= 0).mean(),
                     "Tỷ lệ khách có τ̂ nhánh này cao hơn": (best_arm == a).mean()})
t579 = pd.DataFrame(arm_rows)
gender = np.select([train["both_gender"] == 1, train["mens"] == 1], ["Both", "Men only"], "Women only")
t579b = pd.DataFrame({
    "Hồ sơ giới tính": gender, "τ̂ Mens": tau_arm["Mens E-Mail"], "τ̂ Womens": tau_arm["Womens E-Mail"],
    "Khuyến nghị Mens": best_arm == "Mens E-Mail"}).groupby("Hồ sơ giới tính").agg(
    **{"Số khách": ("τ̂ Mens", "size"), "τ̂ Mens TB": ("τ̂ Mens", "mean"),
       "τ̂ Womens TB": ("τ̂ Womens", "mean"), "Tỷ lệ khuyến nghị Mens": ("Khuyến nghị Mens", "mean")})
for a in ARMS:
    obs = []
    for gp in t579b.index:
        m = (gender == gp) & train["segment"].isin([a, CONTROL]).to_numpy()
        obs.append(rate_diff_ci(y[m], (train.loc[m, "segment"] == a).astype(int).to_numpy())[0])
    t579b[f"Uplift quan sát {a.split()[0]}"] = obs
display(t579.round(5)); display(t579b.round(5))
t579.to_csv(OUT / "T5_7_9_uplift_by_email_type.csv", index=False, encoding="utf-8-sig")
t579b.reset_index().to_csv(OUT / "T5_7_9b_email_uplift_by_gender_profile.csv", index=False,
                           encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 4.4))
for a, col in zip(ARMS, ["tab:blue", "tab:pink"]):
    ya, ta, s = arm_curves[a]
    phi, q = qini_curve(ya, ta, s)
    ax.plot(phi * 100, q, color=col, lw=1.8, label=f"{a} (Qini = {qini_coef(ya, ta, s):.4f})")
    ax.plot([0, 100], [0, q[-1]], "--", color=col, lw=1)
ax.axhline(0, color="grey", lw=.8)
ax.set_xlabel("Tỷ lệ khách được chọn theo τ̂ của từng loại email (%)")
ax.set_ylabel("Số đơn tăng thêm ước tính, Q(φ)")
ax.set_title("Đường Qini theo loại email (train OOF; nét đứt: ngẫu nhiên)"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "H5_7f_qini_by_email_type.png", bbox_inches="tight"); plt.close(fig)

# ==============================================================================
# 8. ĐÁNH GIÁ MỘT LẦN TRÊN TẬP KIỂM TRA
# ==============================================================================
final = SLearner(base, inter).fit(train, t, y)
joblib.dump({"model": final, "tier_cutoffs": tier_cut, "features": PRE, "base_learner": BEST},
            MODEL_DIR / "uplift_slearner.joblib")
y_te, t_te = test[TARGET].to_numpy(), test["T"].to_numpy()
mu1_te, mu0_te = final.predict_mu(test, 1), final.predict_mu(test, 0)
tau_te = mu1_te - mu0_te

p_resp_te = response_score(FINAL_EST, test)   # mô hình cuối của 5.9 (đã huấn luyện trên toàn bộ train)

scores_te = {"S-learner": tau_te, RESP_LABEL: p_resp_te,
             "Ngẫu nhiên": np.random.default_rng(SEED).random(len(test))}
boot_te = bootstrap_qini(y_te, t_te, scores_te)
t5710 = qini_table(y_te, t_te, scores_te, boot_te)
ci_te = ate_ci(y_te, t_te)
t5710["ATE quan sát (test)"] = ate(y_te, t_te)
t5710["ATE 95% CI"] = f"[{ci_te[0]:.4f}; {ci_te[1]:.4f}]"
display(t5710.round(4))
t5710.to_csv(OUT / "T5_7_10_qini_test.csv", index=False, encoding="utf-8-sig")
diff_te = boot_te["S-learner"] - boot_te[RESP_LABEL]
qdiff_te = {"mean": float(diff_te.mean()), "ci_low": float(np.percentile(diff_te, 2.5)),
            "ci_high": float(np.percentile(diff_te, 97.5)), "P(diff<=0)": float((diff_te <= 0).mean())}

tier_te = tier_of(tau_te)
t5711 = bucket_table(y_te, t_te, tau_te, tier_te, TIERS)
t5711["Tỷ trọng khách"] = t5711["Số khách"] / len(test)
display(t5711.round(4))
t5711.to_csv(OUT / "T5_7_11_uplift_tiers_test.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
for k, col in [("S-learner", "tab:blue"), (RESP_LABEL, "tab:red")]:
    phi, q = qini_curve(y_te, t_te, scores_te[k])
    ax.plot(phi * 100, q, color=col, lw=1.8, label=f"{k} (Qini = {qini_coef(y_te, t_te, scores_te[k]):.4f})")
ax.plot([0, 100], [0, q[-1]], "--", color="black", lw=1.2, label=f"Chọn ngẫu nhiên (Q@100% = {q[-1]:.0f} đơn)")
ax.axhline(0, color="grey", lw=.8)
ax.set_xlabel("Tỷ lệ khách hàng được chọn theo điểm (%)"); ax.set_ylabel("Số đơn tăng thêm ước tính, Q(φ)")
ax.set_title("(a) Đường Qini trên tập kiểm tra"); ax.legend(loc="upper left", fontsize=8.3)
ax = axes[1]
ax.bar(np.arange(4) - .19, t5711["Tỷ lệ mua T=0"] * 100, .38, color="tab:gray", alpha=.8, label="Không nhận (T=0)")
ax.bar(np.arange(4) + .19, t5711["Tỷ lệ mua T=1"] * 100, .38, color="tab:blue", alpha=.8, label="Nhận email (T=1)")
for i, r in t5711.iterrows():
    ax.text(i, max(r["Tỷ lệ mua T=1"], r["Tỷ lệ mua T=0"]) * 100 + .05,
            f"{r['Uplift quan sát']*100:+.2f}\n(C={r['Người mua T=0']})", ha="center", fontsize=8)
ax.set_ylim(0, max(t5711["Tỷ lệ mua T=1"].max(), t5711["Tỷ lệ mua T=0"].max()) * 100 * 1.3)
ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_ylabel("Tỷ lệ mua (%)")
ax.set_title("(b) Bốn nhóm uplift trên test (C = số người mua nhóm đối chứng)"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "H5_7g_test_qini_tiers.png", bbox_inches="tight"); plt.close(fig)

# ==============================================================================
# 9. XUẤT ĐIỂM UPLIFT VÀ TÓM TẮT
# ==============================================================================
out = train[["row_id", "segment", "T", TARGET]].copy()
out["mu_T1"], out["mu_T0"], out["tau_hat"] = mu1_oof[BEST], mu0_oof[BEST], tau
out["decile_tau"], out["uplift_tier"] = dec, tier
out["tau_mens"], out["tau_womens"], out["best_email"] = tau_arm["Mens E-Mail"], tau_arm["Womens E-Mail"], best_arm
out["p_response_model"] = p_resp
out.to_csv(OUT / "train_oof_uplift_scores.csv", index=False, encoding="utf-8-sig")
out_te = test[["row_id", "segment", "T", TARGET]].assign(
    mu_T1=mu1_te, mu_T0=mu0_te, tau_hat=tau_te, decile_tau=deciles(tau_te), uplift_tier=tier_te,
    p_response_model=p_resp_te)
out_te.to_csv(OUT / "test_uplift_scores.csv", index=False, encoding="utf-8-sig")

summary = {
    "treatment_definition": "T=1: Mens/Womens E-Mail; T=0: No E-Mail",
    "evaluation": "Chọn base learner bằng Qini 5-fold OOF trên train; test dùng một lần",
    "features": PRE, "base_learner": BEST, "rf_params": RF_PARAMS, "gb_params": GB_PARAMS,
    "cv_qini": cv_tab.reset_index().round(6).to_dict(orient="records"),
    "ate_train": float(ate(y, t)), "ate_train_ci": [float(ci[0]), float(ci[1])],
    "tau_mean_oof": float(tau.mean()),
    "qini_oof": dict(zip(t575["Chiến lược xếp hạng"], t575["Hệ số Qini"].astype(float))),
    "qini_diff_oof": qdiff,
    "decile_spearman": {"rho": float(rho_dec.statistic), "p": float(rho_dec.pvalue)},
    "tier_cutoffs": tier_cut, "max_abs_smd": float(smd["SMD"].abs().max()),
    "ate_test": float(ate(y_te, t_te)), "ate_test_ci": [float(ci_te[0]), float(ci_te[1])],
    "qini_test": dict(zip(t5710["Chiến lược xếp hạng"], t5710["Hệ số Qini"].astype(float))),
    "qini_diff_test": qdiff_te,
    "seed": SEED, "n_boot": N_BOOT, "sklearn_version": sklearn.__version__,
}
json.dump(summary, open(OUT / "uplift_summary.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print(f"\nSpearman (τ̂ dự đoán vs uplift quan sát theo decile): {rho_dec.statistic:.3f} (p = {rho_dec.pvalue:.3f})")
print("Đã lưu kết quả vào", OUT, "và mô hình vào", MODEL_DIR / "uplift_slearner.joblib")
