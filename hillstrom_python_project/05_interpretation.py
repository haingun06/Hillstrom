# -*- coding: utf-8 -*-
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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, brier_score_loss)

from common import (FIG_DIR, SEED, TARGET, FS1, FS2, CAT_COLS, BIN_COLS, NUM_COLS, LEVELS, LEAK,
                    load_split, fs2_config, load_fs2_model, check_strategy_consistency, lift_at_k)

OUT = FIG_DIR / "5_6"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
N_REPEATS = 30
FEATURE_SETS = {"FS1": FS1, "FS2": FS2}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300})

train, test = load_split()
ytr = train[TARGET].astype(int)
yte = test[TARGET].astype(int).to_numpy()
assert not (LEAK & set(FS2)), "FS2 chứa biến rò rỉ"
print(f"Train {len(train):,} ({ytr.sum()} người mua) | Test {len(test):,} ({yte.sum()} người mua)")
pd.DataFrame({"Tập": ["Train", "Test", "Tổng"],
              "Số quan sát": [len(train), len(test), len(train) + len(test)],
              "Số chuyển đổi": [int(ytr.sum()), int(yte.sum()), int(ytr.sum() + yte.sum())],
              "Tỷ lệ chuyển đổi (%)": [100 * ytr.mean(), 100 * yte.mean(),
                                        100 * (ytr.sum() + yte.sum()) / (len(train) + len(test))]}
             ).to_csv(TAB / "T5_6_1_data_split.csv", index=False, encoding="utf-8-sig")
check_strategy_consistency()

# ==============================================================================
# 5.6.1.a VIF (trên Train, sau mã hóa One-hot drop='first', số liên tục đã chuẩn hóa)
# ==============================================================================
GENDER_LEVELS = ["Men only", "Women only", "Both"]
ALL_LEVELS = {**LEVELS, "gender_profile": GENDER_LEVELS}
GENDER_COLS = ("mens", "womens", "both_gender")


def add_gender_profile(frame, feats):
    """Thay mens/womens/both_gender bằng gender_profile (loại phụ thuộc tuyến tính chính xác)."""
    f = frame.copy()
    f["gender_profile"] = np.select([(f["mens"] == 1) & (f["womens"] == 1), f["mens"] == 1],
                                    ["Both", "Men only"], default="Women only")
    if "both_gender" in f.columns:
        assert (f["both_gender"] == f["mens"] + f["womens"] - 1).all()
    return f, [c for c in feats if c not in GENDER_COLS] + ["gender_profile"]


def design_matrix(frame, features, fit_on=None, collapse_gender=False):
    """Ma trận thiết kế: số -> chuẩn hóa; nhị phân giữ nguyên; phân loại -> One-hot bỏ mức đầu.
    collapse_gender=True: dùng gender_profile thay cho mens/womens/both_gender (cho OR và cây DT)."""
    if collapse_gender:
        frame, features = add_gender_profile(frame, features)
        fit_on = frame if fit_on is None else add_gender_profile(fit_on, features)[0]
    fit_on = frame if fit_on is None else fit_on
    num = [c for c in features if c in NUM_COLS]
    cat = [c for c in features if c in ALL_LEVELS]
    binc = [c for c in features if c not in num + cat]
    parts, names = [], []
    if num:
        parts.append(StandardScaler().fit(fit_on[num].astype(float)).transform(frame[num].astype(float)))
        names += num
    if binc:
        parts.append(frame[binc].astype(float).to_numpy())
        names += binc
    if cat:
        enc = OneHotEncoder(categories=[ALL_LEVELS[c] for c in cat], drop="first", sparse_output=False,
                            handle_unknown="error").fit(fit_on[cat])
        parts.append(enc.transform(frame[cat]))
        names += enc.get_feature_names_out(cat).tolist()
    return pd.DataFrame(np.column_stack(parts).astype(float), columns=names, index=frame.index)


vif_frames = {}
for fs, feats in FEATURE_SETS.items():
    X = design_matrix(train, feats)
    Xc = sm.add_constant(X, has_constant="add")
    rank = np.linalg.matrix_rank(Xc.to_numpy())
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        v = float(variance_inflation_factor(Xc.to_numpy(), i))
        lab = ("Vô hạn - phụ thuộc tuyến tính" if np.isinf(v) else "Cao (>10)" if v > 10 else
               "Cần xem xét (5-10)" if v > 5 else "Không cao (<5)")
        rows.append({"Bộ đặc trưng": fs, "Biến mã hóa": col, "VIF": v, "Đánh giá": lab})
    vif_frames[fs] = pd.DataFrame(rows)
    print(f"VIF {fs}: {X.shape[1]} cột, rank (có hằng số) = {rank}/{Xc.shape[1]}")
vif_full = pd.concat(vif_frames.values(), ignore_index=True)
print(vif_full.round(3).to_string(index=False))
vif_full.to_csv(TAB / "T5_6_1_VIF_full_FS1_FS2.csv", index=False, encoding="utf-8-sig")

v1, v2 = vif_frames["FS1"].set_index("Biến mã hóa"), vif_frames["FS2"].set_index("Biến mã hóa")
vif_cmp = pd.DataFrame([{
    "Biến mã hóa": n, "VIF_FS1": v1["VIF"].get(n, np.nan), "VIF_FS2": v2["VIF"].get(n, np.nan),
    "Phạm vi": "Chung" if n in v1.index and n in v2.index else "Chỉ FS2"}
    for n in sorted(set(v1.index) | set(v2.index))])
vif_cmp["Chênh lệch (FS2-FS1)"] = vif_cmp["VIF_FS2"] - vif_cmp["VIF_FS1"]
vif_cmp.to_csv(TAB / "T5_6_1_VIF_comparison_FS2_vs_FS1.csv", index=False, encoding="utf-8-sig")

# ==============================================================================
# 5.6.1.b Odds Ratio bằng statsmodels.Logit (không class_weight: mục tiêu là hệ số và CI)
# ==============================================================================
def fit_logit(feats):
    X = design_matrix(train, feats, collapse_gender=True)
    m = sm.Logit(ytr, sm.add_constant(X, has_constant="add").astype(float)).fit(disp=False, maxiter=300)
    if not m.mle_retvals.get("converged", False):
        raise RuntimeError(f"Logit chưa hội tụ ({feats})")
    return m, X


or_models, or_tables = {}, {}
for fs, feats in FEATURE_SETS.items():
    m, X = fit_logit(feats)
    ci = m.conf_int(alpha=0.05)
    t = pd.DataFrame({"Bộ đặc trưng": fs, "Biến": m.params.index, "Hệ số beta": m.params.values,
                      "Odds Ratio": np.exp(m.params.values), "CI 95% dưới": np.exp(ci[0].values),
                      "CI 95% trên": np.exp(ci[1].values), "p-value": m.pvalues.values})
    t = t[t["Biến"] != "const"].copy()
    t["Ý nghĩa 5%"] = np.where(t["p-value"] < 0.05, "Có", "Chưa có bằng chứng")
    or_models[fs], or_tables[fs] = m, t
    print(f"\nOdds Ratio {fs} (hội tụ; {X.shape[1]} hệ số; McFadden R2 = {m.prsquared:.4f})")
    print(t.round(4).to_string(index=False))
    t.to_csv(TAB / f"T5_6_1_OR_{fs}.csv", index=False, encoding="utf-8-sig")
pd.concat(or_tables.values(), ignore_index=True).to_csv(
    TAB / "T5_6_1_OR_FS2_primary_FS1_baseline.csv", index=False, encoding="utf-8-sig")

# Độ nhạy: bỏ recency_group (trùng một phần thông tin với recency) để xem OR của recency
m_s, _ = fit_logit([c for c in FS2 if c != "recency_group"])
ci_s = m_s.conf_int(alpha=0.05)
sens = pd.DataFrame({"Biến": m_s.params.index, "Odds Ratio": np.exp(m_s.params.values),
                     "CI 95% dưới": np.exp(ci_s[0].values), "CI 95% trên": np.exp(ci_s[1].values),
                     "p-value": m_s.pvalues.values})
sens = sens[sens["Biến"] == "recency"].copy()
sens["OR FS2 chính (có recency_group)"] = or_tables["FS2"].loc[or_tables["FS2"]["Biến"] == "recency", "Odds Ratio"].iloc[0]
print("\nĐộ nhạy - OR của recency khi bỏ recency_group:\n", sens.round(4).to_string(index=False))
sens.to_csv(TAB / "T5_6_1_OR_FS2_sensitivity_no_recency_group.csv", index=False, encoding="utf-8-sig")

o1, o2 = or_tables["FS1"].set_index("Biến"), or_tables["FS2"].set_index("Biến")
or_cmp = pd.DataFrame([{
    "Biến": n, "OR_FS1": o1["Odds Ratio"].get(n, np.nan), "OR_FS2": o2["Odds Ratio"].get(n, np.nan),
    "p_FS1": o1["p-value"].get(n, np.nan), "p_FS2": o2["p-value"].get(n, np.nan),
    "Phạm vi": "Chung" if n in o1.index and n in o2.index else "Chỉ FS2"}
    for n in sorted(set(o1.index) | set(o2.index))])
or_cmp["Chênh lệch OR (FS2-FS1)"] = or_cmp["OR_FS2"] - or_cmp["OR_FS1"]
or_cmp.to_csv(TAB / "T5_6_1_OR_comparison_FS2_vs_FS1.csv", index=False, encoding="utf-8-sig")

# ==============================================================================
# 5.6.1.c Decision Tree diễn giải (độ sâu 3-4), cùng cấu hình cho FS1 và FS2
# ==============================================================================
# Tỷ lệ mua ~0,9% nên min_samples_leaf=100 có thể làm cây nông hơn 3 tầng; khi đó giảm dần
# min_samples_leaf (100 -> 60 -> 40 -> 30) cho tới khi độ sâu đạt 3-4, và ghi lại giá trị đã dùng.
LEAF_CANDIDATES = [100, 60, 40, 30]


def fit_tree(fs, feats):
    tr_f, tr_feats = add_gender_profile(train, feats)
    te_f, _ = add_gender_profile(test, feats)
    num = [c for c in tr_feats if c in NUM_COLS]
    cat = [c for c in tr_feats if c in ALL_LEVELS]
    binc = [c for c in tr_feats if c not in num + cat]
    tr = []
    if cat:
        tr.append(("cat", OneHotEncoder(categories=[ALL_LEVELS[c] for c in cat], drop="first",
                                        handle_unknown="error", sparse_output=False), cat))
    if binc:
        tr.append(("bin", "passthrough", binc))
    if num:
        tr.append(("num", "passthrough", num))
    for leaf in LEAF_CANDIDATES:
        pre = ColumnTransformer(tr, remainder="drop")
        pipe = Pipeline([("pre", pre), ("dt", DecisionTreeClassifier(
            criterion="gini", max_depth=4, min_samples_leaf=leaf, class_weight=None, random_state=SEED))])
        t0 = time.perf_counter()
        pipe.fit(tr_f[tr_feats], ytr)
        secs = time.perf_counter() - t0
        tree = pipe.named_steps["dt"]
        if 3 <= tree.get_depth() <= 4:
            break
    else:
        print(f"[CẢNH BÁO] Cây {fs} chỉ đạt độ sâu {tree.get_depth()} ngay cả với min_samples_leaf={leaf}")
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    imp = pd.DataFrame({"Cột mã hóa": names, "Gini importance": tree.feature_importances_})
    imp["Đặc trưng"] = imp["Cột mã hóa"].map(
        lambda n: next((c for c in cat if n.startswith(c + "_")), n))
    imp_g = imp.groupby("Đặc trưng", as_index=False)["Gini importance"].sum().sort_values(
        "Gini importance", ascending=False)
    p = pipe.predict_proba(te_f[tr_feats])[:, 1]
    pred = (p >= .5).astype(int)
    met = {"Bộ đặc trưng": fs, "Số biến gốc": len(feats), "min_samples_leaf": leaf,
           "Độ sâu thực tế": int(tree.get_depth()), "Số nút": int(tree.tree_.node_count),
           "Số nút lá": int(tree.get_n_leaves()),
           "Accuracy Train": accuracy_score(ytr, pipe.predict(tr_f[tr_feats])),
           "Accuracy Test": accuracy_score(yte, pred), "ROC-AUC Test": roc_auc_score(yte, p),
           "Average Precision Test": average_precision_score(yte, p),
           "Lift@10% Test": lift_at_k(yte, p, .10), "Brier Test": brier_score_loss(yte, p),
           "Số dự báo mua tại ngưỡng 0,5": int(pred.sum()), "Thời gian fit (giây, tham khảo)": secs}
    return pipe, names, imp_g, met


trees, tnames, timp, tmet = {}, {}, {}, {}
for fs, feats in FEATURE_SETS.items():
    trees[fs], tnames[fs], timp[fs], tmet[fs] = fit_tree(fs, feats)
    print(f"\nDecision Tree {fs}:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in tmet[fs].items()})
    print(timp[fs].round(4).to_string(index=False))
    timp[fs].to_csv(TAB / f"T5_6_1_DT_importance_{fs}.csv", index=False, encoding="utf-8-sig")
tmet_df = pd.DataFrame(tmet.values())
tmet_df.to_csv(TAB / "T5_6_1_DT_metrics_FS2_FS1.csv", index=False, encoding="utf-8-sig")
cols = ["Accuracy Test", "ROC-AUC Test", "Average Precision Test", "Lift@10% Test", "Brier Test"]
pd.DataFrame([{"Chỉ số": c, "FS1": tmet["FS1"][c], "FS2": tmet["FS2"][c],
               "Chênh lệch (FS2-FS1)": tmet["FS2"][c] - tmet["FS1"][c]} for c in cols]
             ).to_csv(TAB / "T5_6_1_DT_metric_comparison_FS2_vs_FS1.csv", index=False, encoding="utf-8-sig")

for fs in ("FS2", "FS1"):
    tree = trees[fs].named_steps["dt"]
    fig, ax = plt.subplots(figsize=(24, 13))
    plot_tree(tree, feature_names=tnames[fs], class_names=["Không mua", "Mua"], filled=True, rounded=True,
              impurity=True, proportion=True, max_depth=4, fontsize=8, ax=ax)
    ax.set_title(f"Decision Tree diễn giải - {fs} | độ sâu = {tree.get_depth()} | "
                 f"min_samples_leaf = {tmet[fs]['min_samples_leaf']}", fontsize=15, pad=18)
    fig.tight_layout()
    fig.savefig(OUT / ("H5_6_1a_DecisionTree_FS2.png" if fs == "FS2" else "H5_6_1a_DecisionTree_FS1_doi_chieu.png"),
                bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(18, 15), gridspec_kw={"height_ratios": [1.35, 1]})
for ax, fs in zip(axes[0], ("FS2", "FS1")):
    d = or_tables[fs].sort_values("Odds Ratio").copy()
    yy = np.arange(len(d)); x = d["Odds Ratio"].to_numpy()
    ax.errorbar(x, yy, xerr=[x - d["CI 95% dưới"].to_numpy(), d["CI 95% trên"].to_numpy() - x], fmt="o",
                capsize=3, color="#1f77b4")
    ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log")
    ax.set_yticks(yy, d["Biến"].tolist(), fontsize=8)
    ax.set_xlabel("Odds Ratio (thang log; CI 95%)"); ax.set_title(f"Logistic Regression - {fs}"); ax.grid(axis="x", alpha=.25)
for ax, fs in zip(axes[1], ("FS2", "FS1")):
    d = timp[fs].sort_values("Gini importance")
    ax.barh(d["Đặc trưng"], d["Gini importance"], color="#2ca02c", alpha=.85)
    ax.set_xlabel("Gini importance (tổng = 1)"); ax.set_title(f"Decision Tree - {fs}"); ax.grid(axis="x", alpha=.25)
fig.suptitle("Diễn giải Logistic Regression và Decision Tree: FS2 chính, FS1 đối chiếu", fontsize=16, y=1.01)
fig.text(.5, .005, "LR: OR có điều kiện, CI 95%; biến liên tục tính theo +1 độ lệch chuẩn. "
         "DT: Gini importance. Hai đại lượng không cùng thang đo.", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "H5_6_1b_LR_DT_FS2_chinh_FS1_doi_chieu.png", bbox_inches="tight"); plt.close(fig)

# ==============================================================================
# 5.6.2 Random Forest (FS2): impurity importance và permutation importance
# ==============================================================================
cfg = fs2_config()
rf_pipe = load_fs2_model("RF", cfg)
clf = rf_pipe.named_steps["clf"]
Xte = test[FS2].copy()
p_rf = rf_pipe.predict_proba(Xte)[:, 1]
base_auc, base_ap = roc_auc_score(yte, p_rf), average_precision_score(yte, p_rf)
print(f"\nRF/FS2 ({cfg['RF']['strategy']}) trên test: ROC-AUC = {base_auc:.4f} | PR-AUC = {base_ap:.4f} "
      f"(mức nền {yte.mean():.4f}); tham số: {cfg['RF']['params']}")

enc_names, owner = [], []
for c in CAT_COLS:
    for lv in LEVELS[c][1:]:
        enc_names.append(f"{c} = {lv}"); owner.append(c)
for c in BIN_COLS + NUM_COLS:
    enc_names.append(c); owner.append(c)
assert len(enc_names) == clf.n_features_in_ == 16, \
    f"Số cột mã hóa ({len(enc_names)}) không khớp mô hình ({clf.n_features_in_}); mô hình có thể là bản FS2 cũ (10 biến)"
M = np.zeros((len(FS2), len(enc_names)))
for j, o in enumerate(owner):
    M[FS2.index(o), j] = 1
imp_mean = M @ clf.feature_importances_
imp_std = (np.vstack([t.feature_importances_ for t in clf.estimators_]) @ M.T).std(axis=0)
imp_df = pd.DataFrame({"Feature": FS2, "Impurity": imp_mean, "Std_across_trees": imp_std}
                      ).sort_values("Impurity", ascending=False).reset_index(drop=True)
imp_df["Share_%"] = imp_df["Impurity"] * 100
imp_df.to_csv(TAB / "T5_6_2a_rf_impurity_importance.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"Encoded_column": enc_names, "Impurity": clf.feature_importances_}
             ).sort_values("Impurity", ascending=False).to_csv(
    TAB / "T5_6_2a2_rf_impurity_by_encoded_column.csv", index=False, encoding="utf-8-sig")
print("\nImpurity importance (gộp về 11 đặc trưng gốc):\n", imp_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 5)); d = imp_df.iloc[::-1]
ax.barh(d["Feature"], d["Impurity"], xerr=d["Std_across_trees"], color="#1f77b4", alpha=.85, capsize=3)
ax.set(xlabel="Mean decrease in impurity (tổng các cột one-hot)", title="Random Forest - impurity importance (FS2)")
ax.grid(axis="x", alpha=.3); fig.tight_layout()
fig.savefig(OUT / "H5_6_2a_rf_impurity_importance.png", bbox_inches="tight"); plt.close(fig)

perm = permutation_importance(rf_pipe, Xte, yte, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                              n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1)
perm_df = pd.DataFrame({"Feature": FS2,
                        "PR_drop_mean": perm["pr_auc"].importances_mean, "PR_drop_std": perm["pr_auc"].importances_std,
                        "ROC_drop_mean": perm["roc_auc"].importances_mean, "ROC_drop_std": perm["roc_auc"].importances_std}
                       ).sort_values("PR_drop_mean", ascending=False).reset_index(drop=True)
perm_df["Clearly_above_0 (mean > 2*std)"] = perm_df["PR_drop_mean"] > 2 * perm_df["PR_drop_std"]
perm_df.to_csv(TAB / "T5_6_2b_permutation_importance.csv", index=False, encoding="utf-8-sig")
print("\nPermutation importance (test):\n", perm_df.round(5).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for a, (mc, sc, ttl, col) in zip(axs, [("PR_drop_mean", "PR_drop_std", "Giảm PR-AUC", "#d62728"),
                                       ("ROC_drop_mean", "ROC_drop_std", "Giảm ROC-AUC", "#2ca02c")]):
    dd = perm_df.sort_values(mc)
    a.barh(dd["Feature"], dd[mc], xerr=dd[sc], color=col, alpha=.85, capsize=3)
    a.axvline(0, color="k", lw=1); a.set_xlabel(ttl); a.grid(axis="x", alpha=.3)
axs[0].set_title("(a) PR-AUC (chỉ số chính)"); axs[1].set_title("(b) ROC-AUC")
fig.suptitle(f"Permutation importance của Random Forest trên tập test (FS2, {N_REPEATS} lần)")
fig.tight_layout(); fig.savefig(OUT / "H5_6_2b_permutation_importance.png", bbox_inches="tight"); plt.close(fig)

GROUPS = {"Khuyến mãi (segment + promo_match)": ["segment", "promo_match"],
          "Recency (recency + recency_group)": ["recency", "recency_group"],
          "Lịch sử mua theo giới (mens + womens + both_gender)": ["mens", "womens", "both_gender"]}
rng = np.random.RandomState(SEED)
rows = []
for g, gc in GROUPS.items():
    d_ap, d_auc = [], []
    for _ in range(N_REPEATS):
        Xp = Xte.copy()
        Xp[gc] = Xte[gc].iloc[rng.permutation(len(Xp))].to_numpy()   # hoán vị CÙNG LÚC cả nhóm
        pp = rf_pipe.predict_proba(Xp)[:, 1]
        d_ap.append(base_ap - average_precision_score(yte, pp))
        d_auc.append(base_auc - roc_auc_score(yte, pp))
    rows.append({"Group": g, "PR_drop_mean": np.mean(d_ap), "PR_drop_std": np.std(d_ap),
                 "ROC_drop_mean": np.mean(d_auc), "ROC_drop_std": np.std(d_auc)})
group_df = pd.DataFrame(rows).sort_values("PR_drop_mean", ascending=False)
group_df.to_csv(TAB / "T5_6_2c_grouped_permutation_importance.csv", index=False, encoding="utf-8-sig")
print("\nGrouped permutation importance:\n", group_df.round(5).to_string(index=False))

# Đối chiếu hai phương pháp
mm = imp_df.set_index("Feature").join(perm_df.set_index("Feature"))
rho, pv = spearmanr(mm["Impurity"], mm["PR_drop_mean"])
summary = {
    "train_n": int(len(train)), "train_conversions": int(ytr.sum()),
    "test_n": int(len(test)), "test_conversions": int(yte.sum()),
    "feature_sets": {"FS1": FS1, "FS2": FS2}, "primary_feature_set": "FS2", "baseline_feature_set": "FS1",
    "or_models": {fs: {"converged": bool(or_models[fs].mle_retvals.get("converged", False)),
                       "n_parameters_including_intercept": int(len(or_models[fs].params)),
                       "pseudo_r2_mcfadden": float(or_models[fs].prsquared),
                       "log_likelihood": float(or_models[fs].llf), "class_weight": None} for fs in FEATURE_SETS},
    "sensitivity_FS2_without_recency_group": {
        "recency_OR": float(sens["Odds Ratio"].iloc[0]),
        "recency_CI95": [float(sens["CI 95% dưới"].iloc[0]), float(sens["CI 95% trên"].iloc[0])],
        "recency_p": float(sens["p-value"].iloc[0])},
    "vif": {"FS1_infinite_count": int(np.isinf(vif_frames["FS1"]["VIF"]).sum()),
            "FS2_infinite_count": int(np.isinf(vif_frames["FS2"]["VIF"]).sum()),
            "FS2_high_vif_features": vif_frames["FS2"].loc[vif_frames["FS2"]["VIF"] > 10, "Biến mã hóa"].tolist()},
    "tree_metrics": tmet,
    "rf": {"strategy": cfg["RF"]["strategy"], "params": cfg["RF"]["params"],
           "test_roc_auc": base_auc, "test_pr_auc": base_ap, "baseline": float(yte.mean()),
           "impurity_top3": imp_df["Feature"].head(3).tolist(),
           "permutation_top3": perm_df["Feature"].head(3).tolist(),
           "n_clear_above_0": int(perm_df["Clearly_above_0 (mean > 2*std)"].sum()),
           "spearman_rho_impurity_vs_permutation": float(rho), "spearman_p": float(pv),
           "permutation_nonpositive": perm_df.loc[perm_df["PR_drop_mean"] <= 0, "Feature"].tolist(),
           "top_group": group_df.iloc[0]["Group"]}}
(OUT / "summary_5_6.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

print("\n=============== SỐ LIỆU ĐỂ ĐIỀN BÁO CÁO 5.6 ===============")
print(f"RF test: ROC-AUC = {base_auc:.3f}, PR-AUC = {base_ap:.4f}, mức nền = {yte.mean():.4f}")
print("Impurity top-3:", [(f, f"{v:.1%}") for f, v in zip(imp_df.Feature.head(3), imp_df.Impurity.head(3))])
print("Permutation (PR-AUC) top-3:", [(f, f"{v:.4f}") for f, v in zip(perm_df.Feature.head(3), perm_df.PR_drop_mean.head(3))])
print(f"Số đặc trưng có mean > 2*std: {summary['rf']['n_clear_above_0']}/{len(FS2)} | "
      f"Spearman rho = {rho:.2f} (p = {pv:.3f})")
print("Nhóm quan trọng nhất:", group_df.iloc[0]["Group"], f"(giảm PR-AUC = {group_df.iloc[0]['PR_drop_mean']:.4f})")
print("\nĐã lưu:", sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()))
