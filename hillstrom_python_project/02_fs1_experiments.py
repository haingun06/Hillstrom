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
# 02 - FS1 EXPERIMENTS
# ==============================================================================


# --- Notebook Code cell 27 ---
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from packaging.version import Version
from IPython.display import display, Image

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

print("Python notebook sẵn sàng.")
print("scikit-learn:", sklearn.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)


# --- Notebook Code cell 28 ---
import os
from pathlib import Path
import json
from packaging.version import Version
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import pandas as pd

# Sử dụng trực tiếp biến FS1 và FS2 đã có sẵn trong globals()
# Thư mục đầu ra riêng cho mục 5.2.1 để không ghi đè kết quả của các mục khác.
OUTPUT_DIR = Path('figures/5_2_1_FS1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chốt đúng không gian FS1 đã định nghĩa ở Chương 4: 7 đặc trưng hồ sơ khách hàng.
EXPECTED_FS1 = [
    'recency', 'log_history', 'mens', 'womens',
    'zip_code', 'newbie', 'channel'
]
RANDOM_STATE = 42

FS1 = [
    'recency', 'log_history', 'mens', 'womens',
    'zip_code', 'newbie', 'channel'
]

FS1P = FS1 + ['segment', 'promo_match']

FS2 = FS1P + ['both_gender', 'recency_group']

FEATURE_SETS = {
    'FS1': FS1,
    'FS1P': FS1P,
    'FS2': FS2
}

PROC_DIR = Path('data/processed')

train_path = PROC_DIR / 'train.csv'
test_path = PROC_DIR / 'test.csv'

if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        "Không tìm thấy train.csv hoặc test.csv. "
        "Hãy chạy 01_prepare_data.py trước."
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Đã nạp dữ liệu Train/Test thành công.")
if list(FS1) != EXPECTED_FS1:
    raise ValueError(
        f"Định nghĩa FS1 không đúng như phương pháp đã chốt. "
        f"Yêu cầu: {EXPECTED_FS1}; hiện tại: {list(FS1)}"
    )

FEATURE_SET_NAME = 'FS1'
FS1_FEATURES = list(FS1)
CONTINUOUS_FEATURES = ['recency', 'log_history']
BINARY_FEATURES = ['mens', 'womens', 'newbie']
CATEGORICAL_FEATURES = ['zip_code', 'channel']
TARGET = 'conversion'

# Kiểm tra đầu vào: chỉ chứa FS1 và nhãn; không đưa đặc trưng FS2/hậu can thiệp vào X.
required_cols = FS1_FEATURES + [TARGET]
for split_name, split_df in [('train', train_df), ('test', test_df)]:
    missing_cols = [col for col in required_cols if col not in split_df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột trong {split_name}: {missing_cols}")
    if split_df[required_cols].isna().any().any():
        raise ValueError(f"Có giá trị thiếu trong FS1/target của {split_name}.")
    if split_df[TARGET].nunique() != 2:
        raise ValueError(f"Tập {split_name} phải có đủ hai lớp 0 và 1.")

forbidden_cols = {
    'segment', 'promo_match', 'both_gender', 'recency_group',
    'visit', 'spend', 'history_segment', 'row_id'
}
assert not (set(FS1_FEATURES) & forbidden_cols), \
    f"FS1 đang chứa biến không được phép: {set(FS1_FEATURES) & forbidden_cols}"
assert set(FS1_FEATURES) == set(CONTINUOUS_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES)

# Tạo X/y riêng cho mục 5.2.1 — tuyệt đối không lấy FS2.
X_train = train_df[FS1_FEATURES].copy()
y_train = train_df[TARGET].astype(int).copy()
X_test = test_df[FS1_FEATURES].copy()
y_test = test_df[TARGET].astype(int).copy()

print(f"Không gian đặc trưng: {FEATURE_SET_NAME} ({len(FS1_FEATURES)} biến)")
print("Danh sách FS1:", FS1_FEATURES)
print(f"Train: {len(X_train):,} dòng | {int(y_train.sum()):,} mẫu dương | tỷ lệ {y_train.mean():.4%}")
print(f"Test : {len(X_test):,} dòng | {int(y_test.sum()):,} mẫu dương | tỷ lệ {y_test.mean():.4%}")

# Thiết lập siêu tham số và đánh giá chéo.
# Lưới tìm kiếm giữ theo cấu hình của notebook 5.2.1 FS1 chuyên biệt đã cung cấp.
CONFIG = {
    'cv_splits': 5,
    'n_jobs': -1,
    'verbose_grid_search': 1,
    'threshold': 0.50,
    'lr_C_values': [0.01, 0.1, 1.0],
    'lr_class_weight_values': [None, 'balanced'],
    'lr_l1_ratio_values': [0.0, 1.0],
    'lr_penalty_values_legacy': ['l1', 'l2'],
    'dt_param_grid': {
        'model__criterion': ['gini', 'entropy'],
        'model__max_depth': [3, 4, 5, 6],
        'model__min_samples_leaf': [50, 100, 200],
        'model__class_weight': [None, 'balanced'],
    }
}


def make_preprocessor(*, scale_continuous: bool) -> ColumnTransformer:
    """Scale hai biến liên tục; giữ cờ nhị phân; One-Hot hai biến phân loại của FS1."""
    continuous_transformer = StandardScaler() if scale_continuous else 'passthrough'
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    return ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, CONTINUOUS_FEATURES),
            ('binary', 'passthrough', BINARY_FEATURES),
            ('categorical', categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder='drop',
    )


def build_lr_pipeline() -> Pipeline:
    model = LogisticRegression(
        solver='liblinear', max_iter=2000, tol=1e-3, random_state=RANDOM_STATE
    )
    return Pipeline([
        ('preprocess', make_preprocessor(scale_continuous=True)),
        ('model', model),
    ])


def build_dt_pipeline() -> Pipeline:
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return Pipeline([
        ('preprocess', make_preprocessor(scale_continuous=False)),
        ('model', model),
    ])


def get_lr_param_grid():
    grid = {
        'model__C': CONFIG['lr_C_values'],
        'model__class_weight': CONFIG['lr_class_weight_values'],
    }
    if Version(sklearn.__version__) >= Version('1.8'):
        grid['model__l1_ratio'] = CONFIG['lr_l1_ratio_values']
    else:
        grid['model__penalty'] = CONFIG['lr_penalty_values_legacy']
    return grid


cv = StratifiedKFold(
    n_splits=CONFIG['cv_splits'], shuffle=True, random_state=RANDOM_STATE
)

model_specs = {
    'LR': (build_lr_pipeline(), get_lr_param_grid()),
    'DT': (build_dt_pipeline(), CONFIG['dt_param_grid']),
}
searches = {}
cv_rows = []

for model_name, (pipeline, param_grid) in model_specs.items():
    print(f"\n{'=' * 20} GridSearchCV: {model_name} trên {FEATURE_SET_NAME} {'=' * 20}")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv,
        refit=True,
        n_jobs=CONFIG['n_jobs'],
        verbose=CONFIG['verbose_grid_search'],
        return_train_score=False,
        error_score='raise',
    )
    # Chỉ fit bằng tập train; không truyền X_test/y_test vào quá trình tuning.
    search.fit(X_train, y_train)
    searches[model_name] = search

    best_idx = search.best_index_
    best_mean = float(search.cv_results_['mean_test_score'][best_idx])
    best_std = float(search.cv_results_['std_test_score'][best_idx])
    cv_rows.append({
        'Feature_set': FEATURE_SET_NAME,
        'Model': model_name,
        'CV_average_precision_mean': best_mean,
        'CV_average_precision_std': best_std,
        'Best_params': json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
    })

    cv_results = pd.DataFrame(search.cv_results_).sort_values('rank_test_score')
    cv_results.insert(0, 'Feature_set', FEATURE_SET_NAME)
    cv_results.to_csv(
        OUTPUT_DIR / f'cv_results_{model_name.lower()}.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"AP CV tốt nhất: {best_mean:.6f} ± {best_std:.6f}")
    print('Siêu tham số tốt nhất:')
    print(json.dumps(search.best_params_, ensure_ascii=False, indent=2, sort_keys=True))

cv_summary = pd.DataFrame(cv_rows)
cv_summary.to_csv(OUTPUT_DIR / 'cv_best_models.csv', index=False, encoding='utf-8-sig')
print("\nTóm tắt kết quả CV trên FS1:")
display(cv_summary)


# --- Notebook Code cell 29 ---
threshold = float(CONFIG["threshold"])
y_true = y_test.to_numpy(dtype=int)
probabilities = {}
predicted_labels = {}
metric_rows = []
predictions = pd.DataFrame({"y_true": y_true}, index=X_test.index)
confusion_rows = []

for model_name, search in searches.items():
    best_model = search.best_estimator_
    prob = best_model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    probabilities[model_name] = prob
    predicted_labels[model_name] = pred
    predictions[f"probability_{model_name}"] = prob
    predictions[f"prediction_{model_name}"] = pred

    best_idx = search.best_index_
    metric_rows.append({
        "Feature_set": FEATURE_SET_NAME,
        "Model": model_name,
        "Threshold": threshold,
        "Test_AUC_ROC": roc_auc_score(y_true, prob),
        "Test_Average_Precision_AUC_PR": average_precision_score(y_true, prob),
        "Test_Precision": precision_score(y_true, pred, zero_division=0),
        "Test_Recall": recall_score(y_true, pred, zero_division=0),
        "Test_F1": f1_score(y_true, pred, zero_division=0),
        "Test_Accuracy": accuracy_score(y_true, pred),
        "Test_positive_rate": float(y_true.mean()),
        "CV_AP_mean": float(search.cv_results_["mean_test_score"][best_idx]),
        "CV_AP_std": float(search.cv_results_["std_test_score"][best_idx]),
        "Best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
    })

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
    for actual_i, actual_label in enumerate([0, 1]):
        for predicted_i, predicted_label in enumerate([0, 1]):
            confusion_rows.append({
                "Model": model_name,
                "Actual_label": actual_label,
                "Predicted_label": predicted_label,
                "Count": int(cm[actual_i, predicted_i]),
                "Percent_within_actual_class": float(row_pct[actual_i, predicted_i]),
            })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(confusion_rows).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False, encoding="utf-8-sig")

comparison_df = cv_summary.merge(
    metrics_df.drop(columns=["CV_AP_mean", "CV_AP_std", "Best_params"]),
    on="Model", how="left"
)
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

print(f"Chỉ số trên test tại ngưỡng {threshold:.2f}:")
display(metrics_df[[
    "Model", "Test_AUC_ROC", "Test_Average_Precision_AUC_PR",
    "Test_Precision", "Test_Recall", "Test_F1", "Test_Accuracy"
]])


# --- Notebook Code cell 30 ---
def plot_roc(probabilities, y_true, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_value = roc_auc_score(y_true, prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ngẫu nhiên (AUC = 0,5000)")
    ax.set_title(f"Đường cong ROC trên tập kiểm tra ({FEATURE_SET_NAME})")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "01_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(probabilities, y_true, output_dir):
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, prob in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP = {ap:.4f})")
    ax.axhline(prevalence, linestyle="--", color="gray",
               label=f"Baseline tỷ lệ dương = {prevalence:.4%}")
    ax.set_title(f"Đường cong Precision–Recall trên tập kiểm tra ({FEATURE_SET_NAME})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25); ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "02_precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def cumulative_gain_and_lift(y_true, probability):
    order = np.argsort(-probability, kind="mergesort")
    sorted_y = np.asarray(y_true)[order]
    cumulative_positives = np.cumsum(sorted_y)
    n = len(sorted_y)
    total_positives = int(np.sum(sorted_y))
    if total_positives == 0:
        raise ValueError("Không có mẫu dương trong tập test; không thể tính Gain/Lift.")
    fraction = np.arange(1, 101, dtype=float) / 100.0
    positions = np.maximum(1, np.ceil(fraction * n).astype(int))
    gain = cumulative_positives[positions - 1] / total_positives
    lift = gain / fraction
    gain_x = np.concatenate(([0.0], fraction))
    gain_y = np.concatenate(([0.0], gain))
    return gain_x, gain_y, fraction, lift


def plot_gain_lift(probabilities, y_true, output_dir):
    fig, ax_gain = plt.subplots(figsize=(10, 6))
    ax_lift = ax_gain.twinx()
    color_map = {"LR": "tab:blue", "DT": "tab:orange"}
    for name, prob in probabilities.items():
        gain_x, gain_y, lift_x, lift_y = cumulative_gain_and_lift(y_true, prob)
        color = color_map.get(name, None)
        ax_gain.plot(gain_x * 100, gain_y * 100, linewidth=2, color=color,
                     label=f"{name} – Cumulative Gain")
        ax_lift.plot(lift_x * 100, lift_y, linewidth=1.8, color=color, linestyle="--",
                     label=f"{name} – Lift")
    baseline_x = np.linspace(0, 100, 101)
    ax_gain.plot(baseline_x, baseline_x, color="gray", linestyle=":", label="Gain ngẫu nhiên")
    ax_lift.axhline(1.0, color="black", linestyle=":", alpha=0.65, label="Lift ngẫu nhiên = 1")
    ax_gain.set_title(f"Cumulative Gain và Lift theo tỷ lệ khách hàng được chọn ({FEATURE_SET_NAME})")
    ax_gain.set_xlabel("Tỷ lệ mẫu được chọn theo xác suất giảm dần (%)")
    ax_gain.set_ylabel("Cumulative Gain: tỷ lệ người mua được thu hồi (%)")
    ax_lift.set_ylabel("Lift (lần so với chọn ngẫu nhiên)")
    ax_gain.set_xlim(0, 100); ax_gain.set_ylim(0, 102)
    ax_gain.grid(alpha=0.25)
    h1, l1 = ax_gain.get_legend_handles_labels()
    h2, l2 = ax_lift.get_legend_handles_labels()
    ax_gain.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "03_gain_lift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(predicted_labels, y_true, output_dir):
    names = list(predicted_labels.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 5.5), squeeze=False)
    last_image = None
    threshold = CONFIG["threshold"]
    for ax, name in zip(axes[0], names):
        cm = confusion_matrix(y_true, predicted_labels[name], labels=[0, 1])
        row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
        last_image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_title(f"Ma trận nhầm lẫn – {name}\nNgưỡng = {threshold:.2f}")
        ax.set_xlabel("Nhãn dự đoán"); ax.set_ylabel("Nhãn thực tế")
        ax.set_xticks([0, 1], labels=["Không mua (0)", "Mua (1)"])
        ax.set_yticks([0, 1], labels=["Không mua (0)", "Mua (1)"])
        ax.tick_params(axis="x", rotation=10)
        for i in range(2):
            for j in range(2):
                pct = row_pct[i, j]
                ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)",
                        ha="center", va="center",
                        color="white" if pct >= 55 else "black",
                        fontsize=11, fontweight="bold")
    if last_image is not None:
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.18, top=0.80, wspace=0.34)
        cax = fig.add_axes([0.91, 0.22, 0.018, 0.52])
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.set_label("Tỷ lệ trong từng lớp thực tế (%)")
    fig.suptitle(f"Ma trận nhầm lẫn trên tập kiểm tra ({FEATURE_SET_NAME}): số lượng và tỷ lệ", y=0.96)
    fig.savefig(output_dir / "04_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_roc(probabilities, y_true, OUTPUT_DIR)
plot_precision_recall(probabilities, y_true, OUTPUT_DIR)
plot_gain_lift(probabilities, y_true, OUTPUT_DIR)
plot_confusion_matrices(predicted_labels, y_true, OUTPUT_DIR)

# Hiển thị trực tiếp các PNG đã lưu để biểu đồ cũng xuất hiện khi notebook được mở.
for image_name in [
    "01_roc_curve.png",
    "02_precision_recall_curve.png",
    "03_gain_lift.png",
    "04_confusion_matrices.png",
]:
    print(image_name)
    display(Image(filename=str(OUTPUT_DIR / image_name)))


# --- Notebook Markdown cell 31 ---
# ### **5.2.2. Random Forest (RF) và Gradient Boosting (GB)**


# --- Notebook Code cell 32 ---
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, precision_score, recall_score,
    fbeta_score, f1_score, ConfusionMatrixDisplay)
warnings.filterwarnings("ignore")
SEED = 42

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"
TARGET = "conversion"

train, test = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
for d in (train, test):
    if "log_history" not in d.columns:          # nếu Chương 4 chưa tạo sẵn
        d["log_history"] = np.log1p(d["history"])

FS1 = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel"]
CAT = ["zip_code", "channel"]
assert not {"visit", "spend", "history_segment", "segment", "history"} & set(FS1)  # chống rò rỉ
assert set(FS1) <= set(train.columns) and set(FS1) <= set(test.columns)

Xtr, ytr = train[FS1], train[TARGET]
Xte, yte = test[FS1],  test[TARGET]
print(f"Train {len(Xtr)} ({ytr.sum()} người mua, {ytr.mean():.4f}) | Test {len(Xte)} ({yte.sum()} người mua, {yte.mean():.4f})")

pre = ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT)], remainder="passthrough")
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


# --- Notebook Code cell 33 ---
rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
rf_grid = {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 6, 10],
           "clf__min_samples_leaf": [20, 50, 100], "clf__max_features": ["sqrt", 0.5],
           "clf__class_weight": [None, "balanced_subsample"]}
gs_rf = RandomizedSearchCV(rf, rf_grid, n_iter=12, scoring="average_precision",
                           cv=cv, random_state=SEED, n_jobs=1)
gs_rf.fit(Xtr, ytr)

gb = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=SEED))])
gb_grid = {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.03, 0.1],
           "clf__max_depth": [2, 3], "clf__subsample": [0.7, 1.0]}
gs_gb = GridSearchCV(gb, gb_grid, scoring="average_precision", cv=cv, n_jobs=-1)
gs_gb.fit(Xtr, ytr)

models = {"Random Forest": gs_rf, "Gradient Boosting": gs_gb}
for n, g in models.items():
    print(n, g.best_params_, f"CV PR-AUC = {g.best_score_:.4f} ± {g.cv_results_['std_test_score'][g.best_index_]:.4f}")


# --- Notebook Code cell 34 ---
def lift_at_k(yt, p, k):
    idx = np.argsort(-p)[:int(len(yt) * k)]; return yt.iloc[idx].mean() / yt.mean()
def gain_at_k(yt, p, k):
    idx = np.argsort(-p)[:int(len(yt) * k)]; return yt.iloc[idx].sum() / yt.sum()
def best_thr_f2(yt, p):
    pr, rc, th = precision_recall_curve(yt, p)
    f2 = (5 * pr * rc / (4 * pr + rc + 1e-12))[:-1]
    return th[np.argmax(f2)]

rng = np.random.RandomState(SEED); yv = yte.values; n = len(yv)
rows, probs, thr = [], {}, {}
for name, g in models.items():
    est = g.best_estimator_
    oof = cross_val_predict(est, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr[name] = best_thr_f2(ytr, oof)
    p = est.predict_proba(Xte)[:, 1]; probs[name] = p
    yp = (p >= thr[name]).astype(int)
    bs_auc, bs_ap = [], []
    for _ in range(1000):
        i = rng.randint(0, n, n)
        if yv[i].sum() == 0: continue
        bs_auc.append(roc_auc_score(yv[i], p[i])); bs_ap.append(average_precision_score(yv[i], p[i]))
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    rows.append({"Mô hình": name,
        "ROC-AUC": round(roc_auc_score(yte, p), 3),
        "ROC-AUC 95% CI": f"{np.percentile(bs_auc,2.5):.3f}-{np.percentile(bs_auc,97.5):.3f}",
        "PR-AUC": round(average_precision_score(yte, p), 4),
        "PR-AUC 95% CI": f"{np.percentile(bs_ap,2.5):.4f}-{np.percentile(bs_ap,97.5):.4f}",
        "Ngưỡng": round(thr[name], 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": round(recall_score(yte, yp), 3), "Precision": round(precision_score(yte, yp), 4),
        "F1": round(f1_score(yte, yp), 3), "F2": round(fbeta_score(yte, yp, beta=2), 3),
        "Lift@5%": round(lift_at_k(yte, p, .05), 2), "Lift@10%": round(lift_at_k(yte, p, .10), 2),
        "Lift@20%": round(lift_at_k(yte, p, .20), 2), "Gain@10%": round(gain_at_k(yte, p, .10), 3),
        "Gain@20%": round(gain_at_k(yte, p, .20), 3), "Gain@30%": round(gain_at_k(yte, p, .30), 3)})
res = pd.DataFrame(rows).set_index("Mô hình").T
display(res)

dm = Pipeline([("pre", pre), ("clf", DummyClassifier(strategy="prior"))]).fit(Xtr, ytr)
pd_ = dm.predict_proba(Xte)[:, 1]
print("Dummy: ROC-AUC", roc_auc_score(yte, pd_), "| PR-AUC", round(average_precision_score(yte, pd_), 4),
      "| Accuracy nếu đoán toàn 0:", round(1 - yte.mean(), 4))


# --- Notebook Code cell 35 ---
col = {"Random Forest": "#1f77b4", "Gradient Boosting": "#d62728"}

fig, ax = plt.subplots(figsize=(6, 5))
for nme, p in probs.items():
    f, t, _ = roc_curve(yte, p)
    ax.plot(f, t, color=col[nme], label=f"{nme} (AUC = {roc_auc_score(yte, p):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Ngẫu nhiên (AUC = 0,500)")
ax.set(xlabel="Tỷ lệ dương tính giả (FPR)", ylabel="Tỷ lệ dương tính thật (TPR)",
       title="Đường ROC trên tập kiểm tra (FS1)")
ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("H5_roc.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 5))
for nme, p in probs.items():
    pr, rc, _ = precision_recall_curve(yte, p)
    ax.plot(rc, pr, color=col[nme], label=f"{nme} (PR-AUC = {average_precision_score(yte, p):.4f})")
ax.axhline(yte.mean(), color="k", ls="--", label=f"Mức nền = {yte.mean():.4f}")
ax.set(xlabel="Recall", ylabel="Precision", title="Đường Precision–Recall trên tập kiểm tra (FS1)")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("H5_pr.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 5))
for nme, p in probs.items():
    o = np.argsort(-p); cum = np.cumsum(yv[o]) / yv.sum(); x = np.arange(1, n + 1) / n
    ax.plot(np.r_[0, x], np.r_[0, cum], color=col[nme], label=nme)
ax.plot([0, 1], [0, 1], "k--", label="Ngẫu nhiên")
ax.set(xlabel="Tỷ lệ khách hàng được liên hệ (top-k%)", ylabel="Tỷ lệ người mua bắt được",
       title="Đường Cumulative Gain trên tập kiểm tra (FS1)")
ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("H5_gain.png", dpi=200)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for a, nme in zip(axs, probs):
    cm = confusion_matrix(yte, (probs[nme] >= thr[nme]).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Không mua", "Mua"]).plot(
        ax=a, cmap="Blues", values_format="d", colorbar=False)
    a.set_title(f"{nme}\nngưỡng = {thr[nme]:.4f}"); a.set_xlabel("Dự đoán"); a.set_ylabel("Thực tế")
fig.suptitle("Ma trận nhầm lẫn trên tập kiểm tra (FS1)"); fig.tight_layout(); fig.savefig("H5_cm.png", dpi=200)
plt.show()


# --- Notebook Code cell 36 ---
for name, g in models.items():
    est = g.best_estimator_
    imp = pd.Series(est.named_steps["clf"].feature_importances_,
                    index=est.named_steps["pre"].get_feature_names_out()).sort_values(ascending=False)
    print(name, "\n", imp.round(3), "\n")


# --- Notebook Markdown cell 40 ---
# ## **5.3. So sánh chiến lược xử lý mất cân bằng (trên FS1)**
#
# Với mỗi mô hình trong {LR, DT, RF, GB}, thử 3 chiến lược:
#
# | Chiến lược | Cách làm |
# |---|---|
# | **Không xử lý** | Giữ nguyên tỷ lệ lớp (≈ 1% người mua) |
# | **class_weight** | LR/DT: `balanced`; RF: `balanced_subsample`; GB: `sample_weight` cân bằng (GB của scikit-learn không có tham số `class_weight`) |
# | **SMOTENC** | Sinh mẫu tổng hợp cho lớp mua, **chỉ trong từng fold huấn luyện** (đặt trong `imblearn.Pipeline`) |
#
# **Nguyên tắc để so sánh công bằng và không rò rỉ**
# - Cùng bộ đặc trưng FS1, cùng 5 fold (`StratifiedKFold`, seed 42), cùng lưới siêu tham số giữa các chiến lược; chỉ khác chiến lược.
# - Mỗi tổ hợp (mô hình × chiến lược) được tinh chỉnh riêng bằng CV với `average_precision`.
# - **Chiến lược tốt nhất được chọn chỉ bằng điểm CV trên tập train.** Điểm trên tập test được tính để báo cáo, không dùng để chọn.
# - SMOTENC coi `zip_code`, `channel`, `mens`, `womens`, `newbie` là biến phân loại (nếu không, biến nhị phân sẽ bị nội suy thành số lẻ như 0,4). Dữ liệu được mã hoá thứ tự -> SMOTENC -> one-hot -> chuẩn hoá, tất cả nằm trong pipeline.
# - Ngưỡng quyết định chọn bằng F2 trên dự đoán out-of-fold của train, riêng cho từng tổ hợp.


# --- Notebook Code cell 41 ---
# 5.3.0 Thiết lập, nạp dữ liệu
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC
except ImportError:                                  # Colab thường đã có sẵn; nếu thiếu thì cài
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"])
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, fbeta_score, brier_score_loss)
warnings.filterwarnings("ignore")

SEED = 42
TRAIN_FILE, TEST_FILE = "data/processed/train.csv", "data/processed/test.csv"
FIG53 = Path("figures");  FIG53.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

# ---- Cấu hình (chỉnh ở đây) ----
SMOTE_RATIO = 0.10   # tỷ lệ lớp mua / lớp không mua sau SMOTENC (0,10 ≈ từ 1% lên ~9%); 'auto' = 1:1
N_ITER_RF   = 12     # số tổ hợp thử của RandomizedSearchCV cho RF (lưới đầy đủ có 36 tổ hợp)
N_ITER_GB   = 8      # GB: 8/16 tổ hợp cho nhanh; đặt 16 để chạy đủ lưới như mục 5.2.2
MODELS = ["LR", "DT", "RF", "GB"]
STRATS = ["Không xử lý", "class_weight", "SMOTENC"]

TARGET = "conversion"
FS1 = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel"]
train53, test53 = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
for d in (train53, test53):
    if "log_history" not in d.columns:
        d["log_history"] = np.log1p(d["history"])
assert not {"visit", "spend", "history_segment", "segment", "history", "promo_match"} & set(FS1)   # chống rò rỉ
Xtr, ytr = train53[FS1], train53[TARGET].astype(int)
Xte, yte = test53[FS1],  test53[TARGET].astype(int)
ytr_a, yte_a = ytr.to_numpy(), yte.to_numpy()
print(f"Train {len(Xtr):,} ({ytr.sum()} người mua, {ytr.mean():.4%}) | Test {len(Xte):,} ({yte.sum()} người mua, {yte.mean():.4%})")

cv53 = StratifiedKFold(5, shuffle=True, random_state=SEED)   # cùng fold với mục 5.2


# --- Notebook Code cell 42 ---
# 5.3.1 Hàm dựng pipeline cho từng (mô hình, chiến lược)
CATLIKE = ["zip_code", "channel", "mens", "womens", "newbie"]   # cột 0-4: coi là phân loại cho SMOTENC
NUMCOLS = ["recency", "log_history"]                              # cột 5-6
CAT_IDX, BIN_IDX, NUM_IDX = [0, 1], [2, 3, 4], [5, 6]
ZIP_LV, CH_LV = ["Rural", "Suburban", "Urban"], ["Multichannel", "Phone", "Web"]

def make_pipeline(model, strat):
    enc = ColumnTransformer(
        [("cat", OrdinalEncoder(categories=[ZIP_LV, CH_LV, [0, 1], [0, 1], [0, 1]]), CATLIKE),
         ("num", "passthrough", NUMCOLS)], remainder="drop")
    scale = StandardScaler() if model == "LR" else "passthrough"      # cây không cần chuẩn hoá
    pre = ColumnTransformer(
        [("oh", OneHotEncoder(categories=[[0., 1., 2.]] * 2, drop="first", sparse_output=False), CAT_IDX),
         ("bin", "passthrough", BIN_IDX),
         ("num", scale, NUM_IDX)], remainder="drop")

    cw = (strat == "class_weight")
    if model == "LR":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced" if cw else None, random_state=SEED)
        grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    elif model == "DT":
        clf = DecisionTreeClassifier(class_weight="balanced" if cw else None, random_state=SEED)
        grid = {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [50, 100, 200, 400],
                "clf__criterion": ["gini", "entropy"]}
    elif model == "RF":
        clf = RandomForestClassifier(class_weight="balanced_subsample" if cw else None,
                                     random_state=SEED, n_jobs=-1)
        grid = {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [20, 50, 100], "clf__max_features": ["sqrt", 0.5]}
    else:  # GB: không có class_weight -> dùng sample_weight khi fit
        clf = GradientBoostingClassifier(random_state=SEED)
        grid = {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3], "clf__subsample": [0.7, 1.0]}

    steps = [("enc", enc)]
    if strat == "SMOTENC":
        steps.append(("smote", SMOTENC(categorical_features=list(range(5)), sampling_strategy=SMOTE_RATIO,
                                       k_neighbors=5, random_state=SEED)))
    steps += [("pre", pre), ("clf", clf)]
    return ImbPipeline(steps), grid

def fit_params(model, strat, y):
    """sample_weight chỉ dùng cho GB + class_weight"""
    if model == "GB" and strat == "class_weight":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}

def make_search(model, pipe, grid):
    scoring = {"ap": "average_precision", "roc": "roc_auc"}
    kw = dict(scoring=scoring, refit="ap", cv=cv53, error_score="raise")
    if model in ("LR", "DT"):
        return GridSearchCV(pipe, grid, n_jobs=-1, **kw)
    n_iter = N_ITER_RF if model == "RF" else N_ITER_GB
    return RandomizedSearchCV(pipe, grid, n_iter=n_iter, random_state=SEED,
                              n_jobs=(1 if model == "RF" else -1), **kw)   # RF đã song song bên trong

# Kiểm tra nhanh: SMOTENC có thực sự tăng số mẫu dương và không tạo số lẻ cho biến nhị phân
_p, _ = make_pipeline("LR", "SMOTENC")
_Z = _p[:-2].fit_resample(Xtr, ytr)
print(f"Sau SMOTENC (trên toàn train, chỉ để kiểm tra): {len(_Z[1]):,} dòng, {int(_Z[1].sum()):,} mẫu dương "
      f"({_Z[1].mean():.2%}); giá trị khác nhau của cột mens: {sorted(set(np.round(_Z[0][:, 2], 6)))}")


# --- Notebook Code cell 43 ---
# 5.3.2 Hàm chỉ số
def lift_at_k(y, p, k):
    n = max(1, int(round(len(y) * k)));  idx = np.argsort(-p, kind="mergesort")[:n]
    return y[idx].mean() / y.mean()

def gain_at_k(y, p, k):
    n = max(1, int(round(len(y) * k)));  idx = np.argsort(-p, kind="mergesort")[:n]
    return y[idx].sum() / y.sum()

def best_thr_f2(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f2 = (5 * pr * rc / (4 * pr + rc + 1e-12))[:-1]
    return float(th[np.argmax(f2)])

def oof_proba(est, model, strat):
    """Dự đoán out-of-fold trên TRAIN (huấn luyện lại từng fold, kể cả SMOTENC / sample_weight)"""
    oof = np.zeros(len(ytr_a))
    for tr, va in cv53.split(Xtr, ytr_a):
        m = clone(est).fit(Xtr.iloc[tr], ytr_a[tr], **fit_params(model, strat, ytr_a[tr]))
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
    return oof


# --- Notebook Code cell 44 ---
for name in ["Xtr", "ytr_a", "cv53", "MODELS", "STRATS"]:
    print(name, ":", name in globals())


# --- Notebook Code cell 45 ---
print("Xtr:", "Xtr" in globals())
print("ytr_a:", "ytr_a" in globals())
print("MODELS:", "MODELS" in globals())
print("STRATS:", "STRATS" in globals())
print("cv53:", "cv53" in globals())

if "rows" in globals():
    print("Số kết quả đã chạy:", len(rows), "/12")
else:
    print("Chưa có kết quả 5.3.3 trong bộ nhớ")


# --- Notebook Code cell 46 ---
# 5.3.3 Chạy thí nghiệm: 4 mô hình x 3 chiến lược (tinh chỉnh bằng CV trên train)
# Thời gian: LR/DT vài giây; RF và GB là phần lâu nhất (có thể vài chục phút trên Colab).
rows, store = [], {}
for model in MODELS:
    for strat in STRATS:
        t0 = time.time()
        pipe, grid = make_pipeline(model, strat)
        search = make_search(model, pipe, grid)
        search.fit(Xtr, ytr_a, **fit_params(model, strat, ytr_a))        # CHỈ dùng train
        i = search.best_index_;  r = search.cv_results_
        est = search.best_estimator_

        oof = oof_proba(est, model, strat)
        thr = best_thr_f2(ytr_a, oof)                                    # ngưỡng chọn trên train
        p = est.predict_proba(Xte)[:, 1]                                 # test: đánh giá, KHÔNG dùng để chọn
        yp = (p >= thr).astype(int)

        rows.append({
            "Mô hình": model, "Chiến lược": strat,
            "CV AP (mean)": r["mean_test_ap"][i], "CV AP (std)": r["std_test_ap"][i],
            "CV ROC-AUC (mean)": r["mean_test_roc"][i],
            "Test ROC-AUC": roc_auc_score(yte_a, p), "Test PR-AUC": average_precision_score(yte_a, p),
            "Lift@5%": lift_at_k(yte_a, p, .05), "Lift@10%": lift_at_k(yte_a, p, .10),
            "Lift@20%": lift_at_k(yte_a, p, .20), "Gain@20%": gain_at_k(yte_a, p, .20),
            "Ngưỡng (F2, OOF)": thr, "Recall": recall_score(yte_a, yp, zero_division=0),
            "Precision": precision_score(yte_a, yp, zero_division=0),
            "F2": fbeta_score(yte_a, yp, beta=2, zero_division=0),
            "Brier": brier_score_loss(yte_a, p),
            "Best_params": json.dumps({k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                                       for k, v in search.best_params_.items()}, ensure_ascii=False),
        })
        store[(model, strat)] = {"est": est, "p": p, "thr": thr}
        print(f"[{model:2s} | {strat:12s}] CV AP = {r['mean_test_ap'][i]:.4f} ± {r['std_test_ap'][i]:.4f} "
              f"| Test PR-AUC = {rows[-1]['Test PR-AUC']:.4f} | {time.time()-t0:.0f}s")

res53 = pd.DataFrame(rows)


# --- Notebook Code cell 47 ---
# 5.3.4 Bảng so sánh và CHỌN chiến lược (chỉ dựa trên điểm CV của train)
base_rate = ytr_a.mean()
print(f"Mức nền PR-AUC (tỷ lệ mua): train {base_rate:.4f} | test {yte_a.mean():.4f}")

best_rows = []
for m in MODELS:
    d = res53[res53["Mô hình"] == m]
    b = d.loc[d["CV AP (mean)"].idxmax()]
    close = d[(b["CV AP (mean)"] - d["CV AP (mean)"]) <= b["CV AP (std)"]]["Chiến lược"].tolist()
    best_rows.append({"Mô hình": m, "Chiến lược được chọn": b["Chiến lược"],
                      "CV AP": b["CV AP (mean)"], "± std": b["CV AP (std)"],
                      "Các chiến lược nằm trong 1 SD của tốt nhất": ", ".join(close),
                      "Test PR-AUC": b["Test PR-AUC"], "Test ROC-AUC": b["Test ROC-AUC"],
                      "Lift@10%": b["Lift@10%"], "Best_params": b["Best_params"]})
best53 = pd.DataFrame(best_rows)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
show = ["Mô hình", "Chiến lược", "CV AP (mean)", "CV AP (std)", "CV ROC-AUC (mean)", "Test ROC-AUC", "Test PR-AUC",
        "Lift@10%", "Lift@20%", "Recall", "Precision", "F2", "Brier"]
print("\n=== Bảng T5.6: so sánh chiến lược mất cân bằng (FS1) ===")
display(res53[show].round(4))
print("\n=== Chiến lược được chọn cho từng mô hình (chọn theo CV AP, không dùng test) ===")
display(best53.drop(columns=["Best_params"]).round(4))

print("\nPivot CV AP (hàng: mô hình, cột: chiến lược):")
display(res53.pivot(index="Mô hình", columns="Chiến lược", values="CV AP (mean)").loc[MODELS, STRATS].round(4))
print("Pivot Test PR-AUC:")
display(res53.pivot(index="Mô hình", columns="Chiến lược", values="Test PR-AUC").loc[MODELS, STRATS].round(4))


# --- Notebook Code cell 48 ---
# 5.3.5 Biểu đồ
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
w, x = 0.26, np.arange(len(MODELS)); colors = ["#7f7f7f", "#1f77b4", "#d62728"]
for j, s in enumerate(STRATS):
    d = res53[res53["Chiến lược"] == s].set_index("Mô hình").loc[MODELS]
    axs[0].bar(x + (j - 1) * w, d["CV AP (mean)"], w, yerr=d["CV AP (std)"], capsize=3, color=colors[j], label=s)
    axs[1].bar(x + (j - 1) * w, d["Test PR-AUC"], w, color=colors[j], label=s)
axs[0].axhline(base_rate, color="k", ls="--", lw=1, label=f"Mức nền = {base_rate:.4f}")
axs[1].axhline(yte_a.mean(), color="k", ls="--", lw=1, label=f"Mức nền = {yte_a.mean():.4f}")
axs[0].set_title("PR-AUC trung bình qua 5 fold CV (train) ± độ lệch chuẩn")
axs[1].set_title("PR-AUC trên tập kiểm tra (chỉ để báo cáo)")
for a in axs:
    a.set_xticks(x); a.set_xticklabels(MODELS); a.set_ylabel("PR-AUC (Average Precision)"); a.grid(axis="y", alpha=.3)
axs[0].legend(fontsize=8)
fig.suptitle("So sánh chiến lược xử lý mất cân bằng trên FS1"); fig.tight_layout()
fig.savefig(FIG53 / "H5_3_imbalance_ap_bars.png", dpi=300, bbox_inches="tight"); plt.show()

fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
for a, m in zip(axs.ravel(), MODELS):
    for j, s in enumerate(STRATS):
        pr, rc, _ = precision_recall_curve(yte_a, store[(m, s)]["p"])
        a.plot(rc, pr, color=colors[j], lw=1.6, label=f"{s} (AP = {average_precision_score(yte_a, store[(m, s)]['p']):.4f})")
    a.axhline(yte_a.mean(), color="k", ls="--", lw=1)
    a.set_title(m); a.grid(alpha=.3); a.legend(fontsize=8)
    a.set_ylim(0, 0.08)    # phóng to vùng Precision thấp; bỏ dòng này nếu muốn xem toàn bộ
for a in axs[1]: a.set_xlabel("Recall")
for a in axs[:, 0]: a.set_ylabel("Precision")
fig.suptitle("Đường Precision–Recall trên tập kiểm tra theo chiến lược (FS1)"); fig.tight_layout()
fig.savefig(FIG53 / "H5_3_imbalance_pr_curves.png", dpi=300, bbox_inches="tight"); plt.show()


# --- Notebook Code cell 49 ---
# 5.3.6 Lưu kết quả cho các mục sau (5.4: FS2, 5.7: mô phỏng lợi nhuận)
res53.to_csv(FIG53 / "T5_6_imbalance_comparison_FS1.csv", index=False, encoding="utf-8-sig")
best53.to_csv(FIG53 / "T5_6_best_strategy_FS1.csv", index=False, encoding="utf-8-sig")

# Dùng ở mục 5.4: mỗi mô hình chạy trên FS2 với đúng chiến lược này
best_strategy = {r["Mô hình"]: {"strategy": r["Chiến lược được chọn"], "cv_ap": float(r["CV AP"]),
                                "best_params_FS1": json.loads(r["Best_params"])} for _, r in best53.iterrows()}
json.dump(best_strategy, open(FIG53 / "best_strategy_FS1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# Điểm dự đoán test của mô hình tốt nhất (nối với outcomes_test.csv bằng row_id ở mục 5.7)
scores = pd.DataFrame({"row_id": test53["row_id"].values, "conversion": yte_a})
for m in MODELS:
    s = best_strategy[m]["strategy"]
    scores[f"p_{m}"] = store[(m, s)]["p"]
    joblib.dump(store[(m, s)]["est"], MODEL_DIR / f"{m}_FS1_{s.replace(' ', '_')}.joblib")
scores.to_csv(FIG53 / "test_scores_FS1_best.csv", index=False)
print(json.dumps({m: v["strategy"] for m, v in best_strategy.items()}, ensure_ascii=False, indent=1))
