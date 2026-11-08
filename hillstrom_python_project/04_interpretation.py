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
# 04 - MODEL INTERPRETATION
# ==============================================================================


# --- Notebook Markdown cell 67 ---
# ## **5.6.1** Odds Ratio, VIF và diễn giải LR/DT
#
# Thiết kế phân tích: FS2 là bộ đặc trưng chính; FS1 là bộ cơ sở dùng để đối chiếu. Hai bộ được xử lý riêng trên cùng tập Train/Test và cùng quy tắc tiền xử lý.
#
# Đầu ra: (1) bảng Odds Ratio và VIF; (2) cây quyết định độ sâu tối đa 4; (3) hình phân tích LR và DT; (4) các bảng so sánh FS2–FS1.
#
# Phạm vi: Chỉ triển khai mục 5.6.1. Không sửa hoặc chạy phần RF/Permutation Importance của mục 5.6.2.
#
# Nguyên tắc: Không đưa visit, spend, history_segment, row_id hoặc conversion vào X. conversion chỉ là biến mục tiêu.


# --- Notebook Code cell 68 ---
# 1. Thư viện, cấu hình và thư mục đầu ra
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, brier_score_loss
from IPython.display import display

SEED = 42
TARGET = "conversion"
OUTPUT_DIR = Path("figures/5_6_1_FS2_FS1")
TABLE_DIR = OUTPUT_DIR / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FS1 = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel"]
FS2 = FS1 + ["segment", "promo_match", "both_gender", "recency_group"]
FEATURE_SETS = {"FS1": FS1, "FS2": FS2}
NUM_COLS = ["recency", "log_history"]
CATEGORY_LEVELS = {
    "zip_code": ["Rural", "Suburban", "Urban"],
    "channel": ["Multichannel", "Phone", "Web"],
    "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
    "recency_group": ["0-3m", "4-6m", "7-9m", "10-12m"],
    "gender_profile": ["Men only", "Women only", "Both"],
}
print("Thư mục đầu ra:", OUTPUT_DIR.resolve())


# --- Notebook Markdown cell 69 ---
# ## Nạp dữ liệu và tái tạo đúng tập Train/Test
# Ưu tiên dùng data/processed/train.csv và test.csv nếu đã được tạo bởi notebook Chương 4. Nếu chưa có, notebook tái tạo dữ liệu từ CSV Hillstrom theo đúng quy tắc: khử trùng lặp trên 12 cột gốc, chuẩn hóa Surburban, tạo đặc trưng và chia 80/20 phân tầng với random_state=42.


# --- Notebook Code cell 70 ---
# 2.1 Nạp Train/Test có sẵn trong kernel để tránh lỗi FileNotFoundError
PROC_DIR = Path("data/processed")
train_path, test_path = PROC_DIR / "train.csv", PROC_DIR / "test.csv"

# Tìm kiếm các DataFrame đã load sẵn trong globals
train_df = globals().get('train_df', globals().get('train54', globals().get('train53')))
test_df = globals().get('test_df', globals().get('test54', globals().get('test53')))

if train_df is not None and test_df is not None:
    data_source = "Sử dụng trực tiếp DataFrame có sẵn trong bộ nhớ kernel"
else:
    if train_path.exists() and test_path.exists():
        train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
        data_source = "Đọc từ data/processed/train.csv và test.csv"
    else:
        candidates = [Path("hillstrom.csv"), Path("hillstrom(1).csv"),
                      Path("data/hillstrom.csv"), Path("data/raw/hillstrom.csv"),
                      Path("/content/hillstrom.csv"), Path("/mnt/data/hillstrom(1).csv")]
        raw_path = next((p for p in candidates if p.exists()), None)
        if raw_path is None:
            raise FileNotFoundError("Không tìm thấy Train/Test đã xử lý hoặc CSV Hillstrom. Hãy đặt hillstrom.csv cạnh notebook hoặc chạy Chương 4 trước.")
        raw = pd.read_csv(raw_path)
        raw_cols = ["recency", "history_segment", "history", "mens", "womens", "zip_code",
                    "newbie", "channel", "segment", "visit", "conversion", "spend"]
        missing = sorted(set(raw_cols) - set(raw.columns))
        if missing: raise ValueError(f"CSV Hillstrom thiếu cột: {missing}")
        raw = raw[raw_cols].copy()
        raw.insert(0, "row_id", np.arange(len(raw)))
        clean = raw.drop_duplicates(subset=raw_cols).copy()
        clean["zip_code"] = clean["zip_code"].replace({"Surburban": "Suburban"})
        clean["log_history"] = np.log1p(clean["history"])
        clean["promo_match"] = (((clean["segment"] == "Mens E-Mail") & (clean["mens"] == 1)) |
                                ((clean["segment"] == "Womens E-Mail") & (clean["womens"] == 1))).astype(int)
        clean["both_gender"] = ((clean["mens"] == 1) & (clean["womens"] == 1)).astype(int)
        clean["recency_group"] = pd.cut(clean["recency"], bins=[0,3,6,9,12],
                                        labels=["0-3m","4-6m","7-9m","10-12m"]).astype(str)
        train_df, test_df = train_test_split(clean, test_size=0.20, random_state=SEED, stratify=clean[TARGET])
        data_source = f"{raw_path} (tái tạo theo pipeline Chương 4)"

for split_name, frame in [("Train", train_df), ("Test", test_df)]:
    missing = sorted(set(FS2 + [TARGET]) - set(frame.columns))
    if missing: raise ValueError(f"{split_name} thiếu cột: {missing}")
    if frame[TARGET].isna().any(): raise ValueError(f"{split_name} có conversion bị thiếu")
    assert set(frame[TARGET].unique()) <= {0,1}
    assert not ({"visit", "spend", "history_segment"} & set(FS2))
    assert "row_id" not in FS2 and TARGET not in FS2
if "row_id" in train_df.columns and "row_id" in test_df.columns:
    assert set(train_df.row_id).isdisjoint(set(test_df.row_id)), "Train/Test bị trùng row_id"

summary_data = pd.DataFrame({
    "Tập": ["Train", "Test", "Tổng"],
    "Số quan sát": [len(train_df), len(test_df), len(train_df)+len(test_df)],
    "Số chuyển đổi": [int(train_df[TARGET].sum()), int(test_df[TARGET].sum()), int(train_df[TARGET].sum()+test_df[TARGET].sum())],
    "Tỷ lệ chuyển đổi (%)": [100*train_df[TARGET].mean(), 100*test_df[TARGET].mean(),
        100*(train_df[TARGET].sum()+test_df[TARGET].sum())/(len(train_df)+len(test_df))]
})
print("Nguồn dữ liệu:", data_source)
display(summary_data.round(3))
summary_data.to_csv(TABLE_DIR / "T5_6_1_data_split.csv", index=False, encoding="utf-8-sig")


# --- Notebook Markdown cell 71 ---
# ## Chuẩn hóa biểu diễn nhóm giới tính
# Trong dữ liệu Hillstrom, mọi khách hàng có mens=1 hoặc womens=1, do đó both_gender = mens + womens - 1. Ba biến này tạo phụ thuộc tuyến tính chính xác khi mô hình có hệ số chặn. VIF trên FS2 đầy đủ giữ nguyên ba biến để phát hiện vấn đề. Riêng mô hình OR và cây DT, ba cột được thay bằng biến phân loại tương đương gender_profile gồm Men only, Women only, Both; cách mã hóa này bảo toàn thông tin nhóm giới tính nhưng tránh đưa các cột dư thừa vào mô hình.


# --- Notebook Code cell 72 ---
# 3.1 Tạo gender_profile và xác nhận quan hệ đại số
for frame in (train_df, test_df):
    frame["gender_profile"] = np.select(
        [(frame["mens"] == 1) & (frame["womens"] == 1),
         (frame["mens"] == 1) & (frame["womens"] == 0)],
        ["Both", "Men only"], default="Women only")
    assert (frame["both_gender"] == frame["mens"] + frame["womens"] - 1).all()
print("Phân bố gender_profile trong Train:")
display(train_df["gender_profile"].value_counts().rename_axis("Nhóm").to_frame("Số khách"))


# --- Notebook Markdown cell 73 ---
# ## Tính VIF trên FS1 và FS2 đầy đủ
# VIF được tính trên ma trận sau mã hóa One-hot (drop='first'), biến số được chuẩn hóa và biến nhị phân giữ nguyên, chỉ sử dụng tập Train. Hệ số chặn được thêm vào phép tính nhưng không đưa vào bảng. FS2 đầy đủ vẫn giữ mens, womens, both_gender để phát hiện quan hệ dư thừa trong cấu trúc đặc trưng gốc.


# --- Notebook Code cell 74 ---
# 4.1 Hàm mã hóa ma trận gốc để tính VIF
def make_original_vif_matrix(frame, features):
    numeric = [c for c in features if c in NUM_COLS]
    categorical = [c for c in features if c in CATEGORY_LEVELS and c != "gender_profile"]
    binary = [c for c in features if c not in numeric + categorical]
    pieces, names = [], []
    if numeric:
        pieces.append(StandardScaler().fit_transform(frame[numeric].astype(float)))
        names.extend(numeric)
    if binary:
        pieces.append(frame[binary].astype(float).to_numpy())
        names.extend(binary)
    if categorical:
        enc = OneHotEncoder(categories=[CATEGORY_LEVELS[c] for c in categorical], drop="first",
                            handle_unknown="error", sparse_output=False)
        pieces.append(enc.fit_transform(frame[categorical]))
        names.extend(enc.get_feature_names_out(categorical).tolist())
    return pd.DataFrame(np.column_stack(pieces).astype(float), columns=names, index=frame.index)

vif_frames = {}
for fs_name, features in FEATURE_SETS.items():
    Xv = make_original_vif_matrix(train_df, features)
    Xv_const = sm.add_constant(Xv, has_constant="add")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, col in enumerate(Xv_const.columns):
            if col == "const": continue
            value = variance_inflation_factor(Xv_const.to_numpy(), i)
            label = ("Vô hạn – phụ thuộc tuyến tính" if np.isinf(value) else
                     ("Cao (>10)" if value > 10 else ("Cần xem xét (5–10)" if value > 5 else "Không cao (<5)")))
            rows.append({"Bộ đặc trưng": fs_name, "Biến mã hóa": col, "VIF": float(value), "Đánh giá": label})
    vif_frames[fs_name] = pd.DataFrame(rows)
    rank = np.linalg.matrix_rank(Xv_const.to_numpy())
    print(f"{fs_name}: {Xv.shape[1]} cột đầu vào; rank ma trận có hằng số = {rank}/{Xv_const.shape[1]}")
vif_full = pd.concat(vif_frames.values(), ignore_index=True)
display(vif_full.round(3))
vif_full.to_csv(TABLE_DIR / "T5_6_1_VIF_full_FS1_FS2.csv", index=False, encoding="utf-8-sig")


# --- Notebook Markdown cell 75 ---
# ## Ước lượng Odds Ratio bằng statsmodels
# Hai mô hình Logistic Regression được ước lượng riêng trên Train bằng statsmodels.Logit, không dùng class_weight, vì mục tiêu là ước lượng hệ số và khoảng tin cậy thống kê, không thay thế mô hình phân loại đã tinh chỉnh ở 5.4.1.
# - recency, log_history: chuẩn hóa theo Train; OR ứng với mức tăng 1 độ lệch chuẩn.
# - Biến nhị phân: OR so sánh giá trị 1 với 0.
# - Biến phân loại: One-hot, bỏ mức tham chiếu đầu tiên.
# - Mức tham chiếu: gender_profile=Men only, zip_code=Rural, channel=Multichannel, segment=No E-Mail, recency_group=0-3m.
# - FS2 giữ recency_group trong mô hình OR để phản ánh đầy đủ FS2; VIF cao giữa recency và recency_group được ghi nhận và dùng để giới hạn diễn giải.


# --- Notebook Code cell 76 ---
# 5.1 Xây dựng ma trận OR không dư thừa ở nhóm giới tính
def make_or_frame(frame, features):
    X = frame[features].copy()
    X["gender_profile"] = frame["gender_profile"]
    X = X.drop(columns=[c for c in ["mens", "womens", "both_gender"] if c in X.columns])
    return X

def build_or_matrix(frame, features):
    Xraw = make_or_frame(frame, features)
    numeric = [c for c in NUM_COLS if c in Xraw.columns]
    categorical = [c for c in ["gender_profile", "zip_code", "channel", "segment", "recency_group"] if c in Xraw.columns]
    binary = [c for c in Xraw.columns if c not in numeric + categorical]
    arrays, names = [], []
    if numeric:
        scaler = StandardScaler().fit(Xraw[numeric])
        arrays.append(scaler.transform(Xraw[numeric])); names.extend(numeric)
    if binary:
        arrays.append(Xraw[binary].astype(float).to_numpy()); names.extend(binary)
    if categorical:
        enc = OneHotEncoder(categories=[CATEGORY_LEVELS[c] for c in categorical], drop="first",
                            handle_unknown="error", sparse_output=False).fit(Xraw[categorical])
        arrays.append(enc.transform(Xraw[categorical])); names.extend(enc.get_feature_names_out(categorical).tolist())
    return pd.DataFrame(np.column_stack(arrays), columns=names, index=frame.index)

or_models, or_tables, or_matrices = {}, {}, {}
for fs_name, features in FEATURE_SETS.items():
    X_or = build_or_matrix(train_df, features)
    X_or_const = sm.add_constant(X_or, has_constant="add")
    model = sm.Logit(train_df[TARGET].astype(int), X_or_const.astype(float)).fit(disp=False, maxiter=300)
    if not model.mle_retvals.get("converged", False): raise RuntimeError(f"Logit chưa hội tụ trên {fs_name}")
    ci = model.conf_int(alpha=0.05)
    result = pd.DataFrame({
        "Bộ đặc trưng": fs_name, "Biến": model.params.index,
        "Hệ số beta": model.params.values, "Odds Ratio": np.exp(model.params.values),
        "CI 95% dưới": np.exp(ci[0].values), "CI 95% trên": np.exp(ci[1].values),
        "p-value": model.pvalues.values})
    result = result[result["Biến"] != "const"].copy()
    result["Ý nghĩa 5%"] = np.where(result["p-value"] < 0.05, "Có", "Chưa có bằng chứng")
    or_models[fs_name], or_tables[fs_name], or_matrices[fs_name] = model, result, X_or
    print(f"{fs_name}: hội tụ={model.mle_retvals['converged']}; n={len(train_df):,}; số hệ số giải thích={X_or.shape[1]}")
    display(result.round(4))
    result.to_csv(TABLE_DIR / f"T5_6_1_OR_{fs_name}.csv", index=False, encoding="utf-8-sig")
or_combined = pd.concat(or_tables.values(), ignore_index=True)
or_combined.to_csv(TABLE_DIR / "T5_6_1_OR_FS2_primary_FS1_baseline.csv", index=False, encoding="utf-8-sig")


# --- Notebook Markdown cell 77 ---
# ## Phân tích độ nhạy: FS2 khi loại recency_group khỏi mô hình OR
# VIF cho thấy recency và các dummy của recency_group cùng mang thông tin thời gian mua gần nhất. Mô hình OR chính vẫn báo cáo đầy đủ FS2; mô hình độ nhạy dưới đây bỏ recency_group để kiểm tra hệ số recency có thay đổi đáng kể hay không. Đây là kiểm tra độ ổn định, không thay thế kết quả chính.


# --- Notebook Code cell 78 ---
FS2_NO_RECENCY_GROUP = [c for c in FS2 if c != "recency_group"]
X_sens = build_or_matrix(train_df, FS2_NO_RECENCY_GROUP)
X_sens_const = sm.add_constant(X_sens, has_constant="add")
model_sens = sm.Logit(train_df[TARGET].astype(int), X_sens_const.astype(float)).fit(disp=False, maxiter=300)
ci_sens = model_sens.conf_int(alpha=0.05)
sensitivity_or = pd.DataFrame({
    "Biến": model_sens.params.index, "Odds Ratio": np.exp(model_sens.params.values),
    "CI 95% dưới": np.exp(ci_sens[0].values), "CI 95% trên": np.exp(ci_sens[1].values),
    "p-value": model_sens.pvalues.values})
sensitivity_or = sensitivity_or[sensitivity_or["Biến"] == "recency"].copy()
sensitivity_or["OR FS2 chính (có recency_group)"] = or_tables["FS2"].loc[or_tables["FS2"]["Biến"] == "recency", "Odds Ratio"].iloc[0]
display(sensitivity_or.round(4))
sensitivity_or.to_csv(TABLE_DIR / "T5_6_1_OR_FS2_sensitivity_no_recency_group.csv", index=False, encoding="utf-8-sig")


# --- Notebook Markdown cell 79 ---
# ## So sánh Odds Ratio giữa FS2 và FS1
# Chỉ so sánh trực tiếp các hệ số có cùng cách mã hóa và cùng mức tham chiếu. Chênh lệch OR được tính là OR_FS2 - OR_FS1; đây là thay đổi về mối liên hệ có điều kiện khi đưa thêm biến vào mô hình, không phải thước đo hiệu năng dự đoán và không phải tác động nhân quả.


# --- Notebook Code cell 80 ---
or1, or2 = or_tables["FS1"].set_index("Biến"), or_tables["FS2"].set_index("Biến")
all_terms = sorted(set(or1.index) | set(or2.index))
rows = []
for term in all_terms:
    a = or1.loc[term] if term in or1.index else None
    b = or2.loc[term] if term in or2.index else None
    rows.append({"Biến": term, "OR_FS1": np.nan if a is None else a["Odds Ratio"],
        "OR_FS2": np.nan if b is None else b["Odds Ratio"],
        "Chênh lệch OR (FS2-FS1)": np.nan if a is None or b is None else b["Odds Ratio"]-a["Odds Ratio"],
        "p_FS1": np.nan if a is None else a["p-value"], "p_FS2": np.nan if b is None else b["p-value"],
        "Phạm vi": "Chung" if a is not None and b is not None else "Chỉ FS2"})
or_compare = pd.DataFrame(rows)
display(or_compare.round(4))
or_compare.to_csv(TABLE_DIR / "T5_6_1_OR_comparison_FS2_vs_FS1.csv", index=False, encoding="utf-8-sig")


# --- Notebook Markdown cell 81 ---
# ## Huấn luyện cây Decision Tree diễn giải và đánh giá trên Test
# Cây được huấn luyện riêng cho mục đích diễn giải, với cùng cấu hình trên hai bộ: criterion='gini', max_depth=4, min_samples_leaf=100, class_weight=None, random_state=42. Biến số không cần chuẩn hóa. Nhóm giới tính được mã hóa thành gender_profile để tránh biến chỉ báo dư thừa.
#
# Đây là cây diễn giải 4 tầng, không thay thế Decision Tree đã tinh chỉnh ở các mục 5.2/5.4.1. Chất lượng trên Test được báo cáo bằng ROC-AUC, Average Precision, Lift@10% và Brier; Accuracy được kèm để cho thấy giới hạn của nó trên dữ liệu mất cân bằng.


# --- Notebook Code cell 82 ---
TREE_PARAMS = dict(criterion="gini", max_depth=4, min_samples_leaf=100, class_weight=None, random_state=SEED)

def make_tree_frame(frame, features):
    X = frame[features].copy()
    X["gender_profile"] = frame["gender_profile"]
    return X.drop(columns=[c for c in ["mens", "womens", "both_gender"] if c in X.columns])

def fit_interpretation_tree(fs_name, features):
    Xtr, Xte = make_tree_frame(train_df, features), make_tree_frame(test_df, features)
    numeric = [c for c in NUM_COLS if c in Xtr.columns]
    categorical = [c for c in ["gender_profile", "zip_code", "channel", "segment", "recency_group"] if c in Xtr.columns]
    binary = [c for c in Xtr.columns if c not in numeric + categorical]
    transformers = []
    if categorical:
        transformers.append(("cat", OneHotEncoder(categories=[CATEGORY_LEVELS[c] for c in categorical],
            drop="first", handle_unknown="error", sparse_output=False), categorical))
    if binary: transformers.append(("bin", "passthrough", binary))
    if numeric: transformers.append(("num", "passthrough", numeric))
    pre = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline([("pre", pre), ("dt", DecisionTreeClassifier(**TREE_PARAMS))])
    t0=time.perf_counter(); model.fit(Xtr, train_df[TARGET].astype(int)); fit_seconds=time.perf_counter()-t0
    p_test = model.predict_proba(Xte)[:,1]
    pred_test = (p_test >= .5).astype(int)
    tree = model.named_steps["dt"]
    depth = int(tree.get_depth())
    if not 3 <= depth <= 4: raise RuntimeError(f"Cây {fs_name} đạt độ sâu {depth}, ngoài yêu cầu 3–4")
    names = [n.split("__",1)[-1] for n in pre.get_feature_names_out()]
    importance = pd.DataFrame({"Cột mã hóa": names, "Gini importance": tree.feature_importances_})
    def base_feature(name):
        for cat in categorical:
            if name.startswith(cat + "_"): return cat
        return name
    importance["Đặc trưng"] = importance["Cột mã hóa"].map(base_feature)
    importance_grouped = importance.groupby("Đặc trưng",as_index=False)["Gini importance"].sum().sort_values("Gini importance",ascending=False)
    prevalence=float(test_df[TARGET].mean()); n_top=max(1,int(round(len(test_df)*.10)))
    top_idx=np.argsort(-p_test,kind="mergesort")[:n_top]
    metrics={"Bộ đặc trưng":fs_name,"Số biến gốc":len(features),"Độ sâu thực tế":depth,
        "Số nút":int(tree.tree_.node_count),"Số nút lá":int(tree.get_n_leaves()),
        "Accuracy Train":accuracy_score(train_df[TARGET],model.predict(Xtr)),
        "Accuracy Test":accuracy_score(test_df[TARGET],pred_test),
        "ROC-AUC Test":roc_auc_score(test_df[TARGET],p_test),
        "Average Precision Test":average_precision_score(test_df[TARGET],p_test),
        "Lift@10% Test":float(test_df[TARGET].to_numpy()[top_idx].mean()/prevalence),
        "Brier Test":brier_score_loss(test_df[TARGET],p_test),
        "Số dự báo mua tại ngưỡng 0,5":int(pred_test.sum()),
        "Thời gian fit (giây, tham khảo)":fit_seconds}
    return model, names, importance_grouped, metrics

tree_models, tree_feature_names, tree_importance, tree_metrics = {}, {}, {}, {}
for fs_name, features in FEATURE_SETS.items():
    model,names,imp,metrics=fit_interpretation_tree(fs_name,features)
    tree_models[fs_name],tree_feature_names[fs_name],tree_importance[fs_name],tree_metrics[fs_name]=model,names,imp,metrics
    print(fs_name,metrics); display(imp)
    imp.to_csv(TABLE_DIR/f"T5_6_1_DT_importance_{fs_name}.csv",index=False,encoding="utf-8-sig")
tree_metrics_df=pd.DataFrame(tree_metrics.values())
display(tree_metrics_df.round(5))
tree_metrics_df.to_csv(TABLE_DIR/"T5_6_1_DT_metrics_FS2_FS1.csv",index=False,encoding="utf-8-sig")


# --- Notebook Markdown cell 83 ---
# ## Tạo các hình đầu ra
# - Hình 5.6.1a chính: cây DT trên FS2, độ sâu thực tế 3–4.
# - Hình đối chiếu: cây DT trên FS1 với cùng cấu hình.
# - Hình 5.6.1b: LR thể hiện bằng OR và CI 95%; DT thể hiện bằng Gini importance. Hai thang đo khác bản chất, chỉ đặt cạnh nhau để bổ sung góc nhìn, không so sánh trị số trực tiếp.


# --- Notebook Code cell 84 ---
# 8.1 Lưu cây FS2 (chính) và FS1 (đối chiếu)
for fs_name in ["FS2","FS1"]:
    pipe=tree_models[fs_name]; tree=pipe.named_steps["dt"]
    fig,ax=plt.subplots(figsize=(24,13))
    plot_tree(tree,feature_names=tree_feature_names[fs_name],class_names=["Không mua","Mua"],
        filled=True,rounded=True,impurity=True,proportion=True,max_depth=4,fontsize=8,ax=ax)
    ax.set_title(f"Decision Tree diễn giải – {fs_name} | độ sâu={tree.get_depth()} | min_samples_leaf=100",fontsize=15,pad=18)
    fig.tight_layout()
    name="H5_6_1a_DecisionTree_FS2_depth4.png" if fs_name=="FS2" else "H5_6_1a_DecisionTree_FS1_depth4_doi_chieu.png"
    fig.savefig(OUTPUT_DIR/name,dpi=300,bbox_inches="tight"); plt.close(fig)
    print("Đã lưu:",OUTPUT_DIR/name)


# --- Notebook Code cell 85 ---
# 8.2 Hình LR–DT: OR (LR) và Gini importance (DT), tách rõ FS2/FS1
fig,axes=plt.subplots(2,2,figsize=(18,15),gridspec_kw={"height_ratios":[1.35,1]})
for ax,fs_name in zip(axes[0],["FS2","FS1"]):
    d=or_tables[fs_name].sort_values("Odds Ratio",ascending=True).copy(); y=np.arange(len(d))
    x=d["Odds Ratio"].to_numpy(); lo=d["CI 95% dưới"].to_numpy(); hi=d["CI 95% trên"].to_numpy()
    ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt="o",capsize=3,color="#1f77b4")
    ax.axvline(1.0,color="black",linestyle="--",linewidth=1); ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(d["Biến"].tolist(),fontsize=8)
    ax.set_xlabel("Odds Ratio (thang log; CI 95%)"); ax.set_title(f"Logistic Regression – {fs_name}"); ax.grid(axis="x",alpha=.25)
for ax,fs_name in zip(axes[1],["FS2","FS1"]):
    d=tree_importance[fs_name].sort_values("Gini importance",ascending=True)
    ax.barh(d["Đặc trưng"],d["Gini importance"],color="#2ca02c",alpha=.85)
    ax.set_xlabel("Gini importance (tổng = 1; không có đơn vị)"); ax.set_title(f"Decision Tree – {fs_name}"); ax.grid(axis="x",alpha=.25)
fig.suptitle("Diễn giải Logistic Regression và Decision Tree: FS2 chính, FS1 đối chiếu",fontsize=16,y=1.01)
fig.text(.5,.005,"LR: OR có điều kiện và CI 95%; số liên tục tính theo +1 độ lệch chuẩn. DT: Gini importance là tỷ trọng giảm impurity. Hai đại lượng không cùng thang đo.",ha="center",fontsize=9)
fig.tight_layout(); fig_path=OUTPUT_DIR/"H5_6_1b_LR_DT_FS2_chinh_FS1_doi_chieu.png"
fig.savefig(fig_path,dpi=300,bbox_inches="tight"); plt.close(fig)
print("Đã lưu:",fig_path)


# --- Notebook Markdown cell 86 ---
# ## So sánh định lượng FS2 và FS1 theo từng khung đầu ra
# Với OR, so sánh biến chung và liệt kê biến chỉ có trong FS2; không diễn giải chênh lệch OR như độ chính xác. Với VIF, so sánh các cột chung và các cột phát sinh do FS2. Với cây DT, dùng cùng cấu hình và cùng tập Test, so sánh ROC-AUC, Average Precision, Lift@10%, Brier, Accuracy và cấu trúc cây.


# --- Notebook Code cell 87 ---
# 9.1 So sánh VIF trên các cột mã hóa
v1,v2=vif_frames["FS1"].set_index("Biến mã hóa"),vif_frames["FS2"].set_index("Biến mã hóa")
rows=[]
for name in sorted(set(v1.index)|set(v2.index)):
    a=v1.loc[name,"VIF"] if name in v1.index else np.nan
    b=v2.loc[name,"VIF"] if name in v2.index else np.nan
    rows.append({"Biến mã hóa":name,"VIF_FS1":a,"VIF_FS2":b,
        "Chênh lệch (FS2-FS1)":b-a if np.isfinite(a) and np.isfinite(b) else np.nan,
        "Phạm vi":"Chung" if name in v1.index and name in v2.index else "Chỉ FS2"})
vif_compare=pd.DataFrame(rows); display(vif_compare.round(3))
vif_compare.to_csv(TABLE_DIR/"T5_6_1_VIF_comparison_FS2_vs_FS1.csv",index=False,encoding="utf-8-sig")

# 9.2 Chênh lệch chỉ số cây trên cùng tập Test
m1,m2=tree_metrics["FS1"],tree_metrics["FS2"]
metric_cols=["Accuracy Test","ROC-AUC Test","Average Precision Test","Lift@10% Test","Brier Test"]
metric_compare=pd.DataFrame([{"Chỉ số":c,"FS1":m1[c],"FS2":m2[c],"Chênh lệch (FS2-FS1)":m2[c]-m1[c]} for c in metric_cols])
display(metric_compare.round(6))
metric_compare.to_csv(TABLE_DIR/"T5_6_1_DT_metric_comparison_FS2_vs_FS1.csv",index=False,encoding="utf-8-sig")


# --- Notebook Markdown cell 88 ---
# ## Tổng hợp và lưu kết quả


# --- Notebook Code cell 89 ---
summary={
 "data_source":data_source,"seed":SEED,"train_n":int(len(train_df)),"train_conversions":int(train_df[TARGET].sum()),
 "test_n":int(len(test_df)),"test_conversions":int(test_df[TARGET].sum()),
 "feature_sets":{"FS1":FS1,"FS2":FS2},"primary_feature_set":"FS2","baseline_feature_set":"FS1",
 "or_models":{fs:{"converged":bool(or_models[fs].mle_retvals.get("converged",False)),
   "n_parameters_including_intercept":int(len(or_models[fs].params)),"pseudo_r2_mcfadden":float(or_models[fs].prsquared),
   "log_likelihood":float(or_models[fs].llf),"gender_encoding":"gender_profile replaces mens/womens/both_gender",
   "class_weight":None} for fs in ["FS1","FS2"]},
 "sensitivity_FS2_without_recency_group": {"recency_OR":float(sensitivity_or["Odds Ratio"].iloc[0]),
   "recency_CI95":[float(sensitivity_or["CI 95% dưới"].iloc[0]),float(sensitivity_or["CI 95% trên"].iloc[0])],
   "recency_p":float(sensitivity_or["p-value"].iloc[0])},
 "vif":{"FS1_infinite_count":int(np.isinf(vif_frames["FS1"]["VIF"]).sum()),
   "FS2_infinite_count":int(np.isinf(vif_frames["FS2"]["VIF"]).sum()),
   "FS2_high_vif_features":vif_frames["FS2"].loc[vif_frames["FS2"]["VIF"]>10,"Biến mã hóa"].tolist()},
 "tree_params":TREE_PARAMS,"tree_metrics":tree_metrics}
(OUTPUT_DIR/"summary_5_6_1_FS2_primary_FS1_baseline.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print("Hoàn tất 5.6.1. Các đầu ra:")
for p in sorted(OUTPUT_DIR.rglob("*")):
    if p.is_file(): print("-",p)


# --- Notebook Markdown cell 90 ---
# ## **5.6.2**


# --- Notebook Code cell 91 ---
# ===================== 5.6.0 Thiết lập =====================
import warnings
from pathlib import Path
import joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

SEED, N_REPEATS = 42, 30
FIG56 = Path("figures/5_6"); FIG56.mkdir(parents=True, exist_ok=True)

FS1 = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel"]
FS2 = FS1 + ["segment", "promo_match", "both_gender", "recency_group"]
CAT_COLS = ["zip_code", "channel", "segment", "recency_group"]
BIN_COLS = ["mens", "womens", "newbie", "promo_match", "both_gender"]
NUM_COLS = ["recency", "log_history"]
LEVELS = {"zip_code": ["Rural", "Suburban", "Urban"],
          "channel": ["Multichannel", "Phone", "Web"],
          "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
          "recency_group": ["0-3m", "4-6m", "7-9m", "10-12m"]}
assert not {"visit", "spend", "history_segment", "history", "row_id", "conversion"} & set(FS2)

# Dữ liệu test (FS2) đọc lại từ file để không phụ thuộc biến cũ trong kernel
test56 = pd.read_csv("data/processed/test.csv")
if "log_history" not in test56.columns:
    test56["log_history"] = np.log1p(test56["history"])
Xte56 = test56[FS2].copy()
yte56 = test56["conversion"].astype(int).to_numpy()

# Mô hình RF trên FS2 (đã tinh chỉnh ở 5.4.2)
fitted_models = globals().get("fitted54", {})

if isinstance(fitted_models, dict) and "RF" in fitted_models:
    rf_pipe = fitted_models["RF"]
    print("Đã lấy model RF từ fitted54.")
else:
    rf_paths = sorted(Path("models").glob("RF_FS2_*.joblib"))

    if rf_paths:
        rf_path = rf_paths[0]
        rf_pipe = joblib.load(rf_path)
        print("Đã load:", rf_path)
    else:
        print("Không tìm thấy model RF đã lưu. Đang huấn luyện model Random Forest fallback...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        from imblearn.pipeline import Pipeline as ImbPipeline

        train56 = pd.read_csv("data/processed/train.csv")

        if "log_history" not in train56.columns:
            train56["log_history"] = np.log1p(train56["history"])

        Xtr56 = train56[FS2].copy()
        ytr56 = train56["conversion"].astype(int).to_numpy()

        enc = ColumnTransformer(
            [
                ("cat", OrdinalEncoder(
                    categories=[LEVELS[c] for c in CAT_COLS]
                ), CAT_COLS),
                ("bin", "passthrough", BIN_COLS),
                ("num", "passthrough", NUM_COLS)
            ],
            remainder="drop"
        )

        pre = ColumnTransformer(
            [
                ("oh", OneHotEncoder(
                    categories=[
                        [float(i) for i in range(len(LEVELS[c]))]
                        for c in CAT_COLS
                    ],
                    drop="first",
                    sparse_output=False
                ), list(range(4))),
                ("bin", "passthrough", list(range(4, 9))),
                ("num", "passthrough", [9, 10])
            ],
            remainder="drop"
        )

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=50,
            random_state=SEED,
            n_jobs=-1
        )

        rf_pipe = ImbPipeline([
            ("enc", enc),
            ("pre", pre),
            ("clf", clf)
        ])

        rf_pipe.fit(Xtr56, ytr56)
        print("Đã huấn luyện xong model Random Forest fallback.")
        # Huấn luyện fallback sử dụng dữ liệu train
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        from imblearn.pipeline import Pipeline as ImbPipeline

        train56 = pd.read_csv("data/processed/train.csv")
        if "log_history" not in train56.columns:
            train56["log_history"] = np.log1p(train56["history"])
        Xtr56 = train56[FS2]
        ytr56 = train56["conversion"].astype(int).to_numpy()

        enc = ColumnTransformer(
            [("cat", OrdinalEncoder(categories=[LEVELS[c] for c in CAT_COLS]), CAT_COLS),
             ("bin", "passthrough", BIN_COLS), ("num", "passthrough", NUM_COLS)], remainder="drop")
        pre = ColumnTransformer(
            [("oh", OneHotEncoder(categories=[[float(i) for i in range(len(LEVELS[c]))] for c in CAT_COLS],
                                  drop="first", sparse_output=False), list(range(4))),
             ("bin", "passthrough", list(range(4, 9))),
             ("num", "passthrough", [9, 10])], remainder="drop")

        clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50, random_state=SEED, n_jobs=-1)
        rf_pipe = ImbPipeline([("enc", enc), ("pre", pre), ("clf", clf)])
        rf_pipe.fit(Xtr56, ytr56)
        print("Đã huấn luyện xong model Random Forest fallback.")

clf = rf_pipe.named_steps["clf"]
p_rf = rf_pipe.predict_proba(Xte56)[:, 1]
base_auc = roc_auc_score(yte56, p_rf); base_ap = average_precision_score(yte56, p_rf)
print(f"RF/FS2 trên test: ROC-AUC = {base_auc:.4f} | PR-AUC = {base_ap:.4f} "
      f"(mức nền PR-AUC = {yte56.mean():.4f})")

# ===================== 5.6.1 Impurity importance của RF =====================
enc_names, owner = [], []
for c in CAT_COLS:
    for lv in LEVELS[c][1:]:
        enc_names.append(f"{c} = {lv}"); owner.append(c)
for c in BIN_COLS + NUM_COLS:
    enc_names.append(c); owner.append(c)
assert len(enc_names) == clf.n_features_in_ == 16, "Số cột không khớp pipeline"

M = np.zeros((len(FS2), len(enc_names)))
for j, o in enumerate(owner):
    M[FS2.index(o), j] = 1

imp_mean = M @ clf.feature_importances_
per_tree = np.vstack([t.feature_importances_ for t in clf.estimators_]) @ M.T
imp_std = per_tree.std(axis=0)

imp_df = (pd.DataFrame({"Feature": FS2, "Impurity": imp_mean, "Std_across_trees": imp_std})
          .sort_values("Impurity", ascending=False).reset_index(drop=True))
imp_df["Share_%"] = imp_df["Impurity"] * 100
imp_df.to_csv(FIG56 / "T5_6a_rf_impurity_importance.csv", index=False, encoding="utf-8-sig")

detail_df = (pd.DataFrame({"Encoded_column": enc_names, "Impurity": clf.feature_importances_})
             .sort_values("Impurity", ascending=False))
detail_df.to_csv(FIG56 / "T5_6a2_rf_impurity_by_encoded_column.csv", index=False, encoding="utf-8-sig")
print("\n=== Bảng 5.6a: Impurity importance (gộp về 11 đặc trưng gốc) ===")
display(imp_df.round(4))

fig, ax = plt.subplots(figsize=(7.5, 5))
d = imp_df.iloc[::-1]
ax.barh(d["Feature"], d["Impurity"], xerr=d["Std_across_trees"], color="#1f77b4", alpha=.85, capsize=3)
ax.set(xlabel="Mean decrease in impurity (summed over one-hot columns)",
       title="Random Forest impurity-based feature importance (FS2)")
ax.grid(axis="x", alpha=.3); fig.tight_layout()
fig.savefig(FIG56 / "H5_6a_rf_impurity_importance.png", dpi=300, bbox_inches="tight"); plt.show()

# ===================== 5.6.2 Permutation importance (tập test) =====================
perm = permutation_importance(
    rf_pipe, Xte56, yte56,
    scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
    n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1)

perm_df = pd.DataFrame({
    "Feature": FS2,
    "PR_drop_mean": perm["pr_auc"].importances_mean,  "PR_drop_std": perm["pr_auc"].importances_std,
    "ROC_drop_mean": perm["roc_auc"].importances_mean, "ROC_drop_std": perm["roc_auc"].importances_std,
}).sort_values("PR_drop_mean", ascending=False).reset_index(drop=True)
perm_df["Clearly_above_0 (mean > 2*std)"] = perm_df["PR_drop_mean"] > 2 * perm_df["PR_drop_std"]
perm_df.to_csv(FIG56 / "T5_6b_permutation_importance.csv", index=False, encoding="utf-8-sig")
print("\n=== Bảng 5.6b: Permutation importance (giảm điểm khi xáo trộn một đặc trưng) ===")
display(perm_df.round(5))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for a, (mcol, scol, ttl, colr) in zip(axs, [("PR_drop_mean", "PR_drop_std", "Drop in PR-AUC", "#d62728"),
                                            ("ROC_drop_mean", "ROC_drop_std", "Drop in ROC-AUC", "#2ca02c")]):
    dd = perm_df.sort_values(mcol).reset_index(drop=True)
    a.barh(dd["Feature"], dd[mcol], xerr=dd[scol], color=colr, alpha=.85, capsize=3)
    a.axvline(0, color="k", lw=1); a.set_xlabel(ttl); a.grid(axis="x", alpha=.3)
axs[0].set_title("(a) PR-AUC (primary metric)"); axs[1].set_title("(b) ROC-AUC")
fig.suptitle(f"Permutation importance of the Random Forest on the test set (FS2, {N_REPEATS} repeats)")
fig.tight_layout(); fig.savefig(FIG56 / "H5_6b_permutation_importance.png", dpi=300, bbox_inches="tight"); plt.show()

# ---- Permutation theo NHÓM (đặc trưng tương quan hoán vị cùng lúc) ----
GROUPS = {"Promotion (segment + promo_match)": ["segment", "promo_match"],
          "Recency (recency + recency_group)": ["recency", "recency_group"],
          "Gender purchase history (mens + womens + both_gender)": ["mens", "womens", "both_gender"]}
rng = np.random.RandomState(SEED)
rows = []
for g, cols in GROUPS.items():
    d_ap, d_auc = [], []
    for _ in range(N_REPEATS):
        Xp = Xte56.copy()
        idx = rng.permutation(len(Xp))
        Xp[cols] = Xte56[cols].iloc[idx].to_numpy()
        pp = rf_pipe.predict_proba(Xp)[:, 1]
        d_ap.append(base_ap - average_precision_score(yte56, pp))
        d_auc.append(base_auc - roc_auc_score(yte56, pp))
    rows.append({"Group": g, "PR_drop_mean": np.mean(d_ap), "PR_drop_std": np.std(d_ap),
                 "ROC_drop_mean": np.mean(d_auc), "ROC_drop_std": np.std(d_auc)})
group_df = pd.DataFrame(rows).sort_values("PR_drop_mean", ascending=False)
group_df.to_csv(FIG56 / "T5_6c_grouped_permutation_importance.csv", index=False, encoding="utf-8-sig")
print("\n=== Bảng 5.6c: Grouped permutation importance ===")
display(group_df.round(5))

# ===================== 5.6.3 Đối chiếu hai phương pháp + khối số liệu để điền báo cáo =====================
m = imp_df.set_index("Feature").join(perm_df.set_index("Feature"))
rho, pval = spearmanr(m["Impurity"], m["PR_drop_mean"])
top_i = imp_df["Feature"].head(3).tolist(); top_p = perm_df["Feature"].head(3).tolist()
n_clear = int(perm_df["Clearly_above_0 (mean > 2*std)"].sum())
print("\n================ FILL-IN VALUES CHO BÁO CÁO 5.6 ================")
print(f"RF test: ROC-AUC = {base_auc:.3f}, PR-AUC = {base_ap:.4f}, baseline = {yte56.mean():.4f}")
cvinfo_data = globals().get("cvinfo", {})

if isinstance(cvinfo_data, dict) and "RF" in cvinfo_data:
    rf_info = cvinfo_data["RF"]

    if isinstance(rf_info, dict) and "best_params" in rf_info:
        print("RF best params:", rf_info["best_params"])
    else:
        print("Không tìm thấy best_params của RF.")
else:
    print("Không có cvinfo trong bộ nhớ. Bỏ qua thông tin RF best params.")
print("Impurity top-3:", [(f, f"{v:.1%}") for f, v in zip(imp_df['Feature'].head(3), imp_df['Impurity'].head(3))])
print("Permutation (PR-AUC) top-3:", [(f, f"{v:.4f}") for f, v in zip(perm_df['Feature'].head(3), perm_df['PR_drop_mean'].head(3))])
print(f"Số đặc trưng có mean > 2*std (PR-AUC): {n_clear}/{len(FS2)}")
print(f"Spearman rho giữa 2 phương pháp = {rho:.2f} (p = {pval:.3f})")
print("Top-3 trùng nhau:", sorted(set(top_i) & set(top_p)))
print("Đặc trưng có permutation importance <= 0:", perm_df.loc[perm_df['PR_drop_mean'] <= 0, 'Feature'].tolist())
print("Nhóm quan trọng nhất (grouped):", group_df.iloc[0]["Group"], f"(PR drop = {group_df.iloc[0]['PR_drop_mean']:.4f})")
