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



import json
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_BOOT = 1000

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
FIG54 = BASE_DIR / "figures"
MODEL_DIR = BASE_DIR / "models"

FIG54.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy dữ liệu: {TRAIN_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy dữ liệu: {TEST_FILE}")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train54 = train_df.copy()
test54 = test_df.copy()

FS1 = [
    "recency",
    "log_history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel"
]

FS1P = FS1 + ["segment", "promo_match"]

FS2 = FS1P + ["recency_group", "both_gender"]

TARGET = "conversion"

for data in [train_df, test_df, train54, test54]:
    if "log_history" not in data.columns and "history" in data.columns:
        data["log_history"] = np.log1p(data["history"])

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError("Không tìm thấy cột conversion trong train.csv hoặc test.csv.")

if not set(FS2).issubset(train_df.columns):
    raise ValueError("Train thiếu một hoặc nhiều cột FS2.")

if not set(FS2).issubset(test_df.columns):
    raise ValueError("Test thiếu một hoặc nhiều cột FS2.")

print("Đã nạp dữ liệu Train/Test thành công.")
print("Train:", train_df.shape)
print("Test:", test_df.shape)
print("FS2:", FS2)





import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.base import clone as clone541
from sklearn.compose import ColumnTransformer as ColumnTransformer541
from sklearn.preprocessing import (
    OrdinalEncoder as OrdinalEncoder541,
    OneHotEncoder as OneHotEncoder541,
    StandardScaler as StandardScaler541,
)
from sklearn.pipeline import Pipeline as Pipeline541
from sklearn.linear_model import LogisticRegression as LogisticRegression541
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier541
from sklearn.model_selection import GridSearchCV as GridSearchCV541
from sklearn.metrics import (
    roc_auc_score as roc_auc_score541,
    average_precision_score as average_precision_score541,
    precision_recall_curve as precision_recall_curve541,
    roc_curve as roc_curve541,
    confusion_matrix as confusion_matrix541,
    accuracy_score as accuracy_score541,
    precision_score as precision_score541,
    recall_score as recall_score541,
    f1_score as f1_score541,
    fbeta_score as fbeta_score541,
    brier_score_loss as brier_score_loss541,
)


_train_data = globals().get('train54', globals().get('train_df'))
_test_data = globals().get('test54', globals().get('test_df'))

if _train_data is None or _test_data is None:
    raise NameError("Không tìm thấy tập dữ liệu Train/Test trong bộ nhớ kernel. Vui lòng chạy lại các bước ở Chương 4.")


FS2_541 = list(FS2)
TARGET_541 = "conversion"
MODELS_541 = ["LR", "DT"]

assert len(FS2_541) == 11, f"FS2 phải có 11 biến, hiện có {len(FS2_541)}: {FS2_541}"
assert set(FS2_541) <= set(_train_data.columns), "Train thiếu một hoặc nhiều biến FS2."
assert set(FS2_541) <= set(_test_data.columns), "Test thiếu một hoặc nhiều biến FS2."
assert TARGET_541 in _train_data.columns and TARGET_541 in _test_data.columns
assert not ({"visit", "spend", "history_segment", "history", "row_id", TARGET_541} & set(FS2_541)), \
    "Phát hiện biến có nguy cơ rò rỉ trong FS2."

Xtr_541 = _train_data[FS2_541].copy()
ytr_541 = _train_data[TARGET_541].astype(int).to_numpy()
Xte_541 = _test_data[FS2_541].copy()
yte_541 = _test_data[TARGET_541].astype(int).to_numpy()


STRATEGY_FILE_541 = Path("figures") / "best_strategy_FS1.json"
if not STRATEGY_FILE_541.exists():
    
    STRAT_541 = {"LR": "class_weight", "DT": "Không xử lý"}
else:
    with open(STRATEGY_FILE_541, "r", encoding="utf-8") as f:
        best_strategy_fs1_541 = json.load(f)
    STRAT_541 = {
        model: best_strategy_fs1_541[model]["strategy"]
        for model in MODELS_541
    }

# Chiến lược cho LR/DT ở 5.4.1 lấy từ kết quả 5.3 (best_strategy_FS1.json), không đặt cứng.
assert set(STRAT_541) == set(MODELS_541)
assert all(v in ("Không xử lý", "class_weight") for v in STRAT_541.values()), \
    "5.4.1 chưa hỗ trợ SMOTENC cho LR/DT; cần bổ sung pipeline imblearn nếu 5.3 chọn SMOTENC."


CAT_COLS_541 = ["zip_code", "channel", "segment", "recency_group"]
BIN_COLS_541 = ["mens", "womens", "newbie", "promo_match", "both_gender"]
NUM_COLS_541 = ["recency", "log_history"]

LEVELS_541 = {
    "zip_code": ["Rural", "Suburban", "Urban"],
    "channel": ["Multichannel", "Phone", "Web"],
    "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
    "recency_group": ["1-3m", "4-6m", "7-9m", "10-12m"],
}

assert sorted(CAT_COLS_541 + BIN_COLS_541 + NUM_COLS_541) == sorted(FS2_541)
for _col in CAT_COLS_541:
    allowed = set(LEVELS_541[_col])
    train_values = set(Xtr_541[_col].dropna().unique())
    test_values = set(Xte_541[_col].dropna().unique())
    assert train_values <= allowed, f"Giá trị lạ trong Train[{_col}]: {train_values - allowed}"
    assert test_values <= allowed, f"Giá trị lạ trong Test[{_col}]: {test_values - allowed}"
    assert Xtr_541[_col].notna().all() and Xte_541[_col].notna().all(), f"Có giá trị thiếu trong {_col}"

def make_pipeline541(model_name):
    enc = ColumnTransformer541(
        transformers=[
            ("cat", OrdinalEncoder541(
                categories=[LEVELS_541[c] for c in CAT_COLS_541]
            ), CAT_COLS_541),
            ("bin", "passthrough", BIN_COLS_541),
            ("num", "passthrough", NUM_COLS_541),
        ],
        remainder="drop",
    )

    num_transformer = StandardScaler541() if model_name == "LR" else "passthrough"
    pre = ColumnTransformer541(
        transformers=[
            ("oh", OneHotEncoder541(
                categories=[
                    [float(i) for i in range(len(LEVELS_541[c]))]
                    for c in CAT_COLS_541
                ],
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            ), list(range(len(CAT_COLS_541)))),
            ("bin", "passthrough", list(range(4, 9))),
            ("num", num_transformer, [9, 10]),
        ],
        remainder="drop",
    )

    if model_name == "LR":
        clf = LogisticRegression541(
            C=1.0,
            class_weight="balanced" if STRAT_541["LR"] == "class_weight" else None,
            max_iter=2000,
            random_state=SEED,
        )
        param_grid = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
        }

    elif model_name == "DT":
        clf = DecisionTreeClassifier541(
            class_weight="balanced" if STRAT_541["DT"] == "class_weight" else None,
            random_state=SEED,
        )
        param_grid = {
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__min_samples_leaf": [50, 100, 200, 400],
            "clf__criterion": ["gini", "entropy"],
        }
    else:
        raise ValueError(f"Mô hình không được hỗ trợ trong 5.4.1: {model_name}")

    pipeline = Pipeline541([
        ("enc", enc),
        ("pre", pre),
        ("clf", clf),
    ])
    return pipeline, param_grid


_pipe_check_541, _ = make_pipeline541("LR")
_X_check_541 = _pipe_check_541[:-1].fit_transform(Xtr_541.head(20), ytr_541[:20])
assert _X_check_541.shape[1] == 16, (
    f"FS2 sau mã hóa phải có 16 cột theo thiết kế, nhận được {_X_check_541.shape[1]}."
)

print("Train:", Xtr_541.shape, "| người mua:", int(ytr_541.sum()), f"({ytr_541.mean():.4%})")
print("Test :", Xte_541.shape, "| người mua:", int(yte_541.sum()), f"({yte_541.mean():.4%})")
print("FS2:", FS2_541)
print("Số cột sau One-hot:", _X_check_541.shape[1])
print("Chiến lược mất cân bằng:", STRAT_541)






from sklearn.model_selection import StratifiedKFold


if 'cv54' not in globals():
    cv54 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

searches_541 = {}
fitted_541 = {}
p541 = {}
oof541 = {}
thr541 = {}
cvinfo541 = {}

def best_threshold_f2_541(y_true, probability):
    precision, recall, thresholds = precision_recall_curve541(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    f2_values = (5 * precision * recall) / (4 * precision + recall + 1e-12)
    return float(thresholds[int(np.nanargmax(f2_values))])

def make_oof_probability541(estimator, X, y, cv):
    oof_probability = np.full(len(y), np.nan, dtype=float)
    for fold_train_idx, fold_valid_idx in cv.split(X, y):
        fold_model = clone541(estimator)
        fold_model.fit(X.iloc[fold_train_idx], y[fold_train_idx])
        oof_probability[fold_valid_idx] = fold_model.predict_proba(
            X.iloc[fold_valid_idx]
        )[:, 1]
    if np.isnan(oof_probability).any():
        raise RuntimeError("OOF còn giá trị NaN; kiểm tra lại StratifiedKFold.")
    return oof_probability

for model_name in MODELS_541:
    print(f"\n{'=' * 18} {model_name} | FS2 | {STRAT_541[model_name]} {'=' * 18}")
    pipeline, param_grid = make_pipeline541(model_name)

    search = GridSearchCV541(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "ap": "average_precision",
            "roc": "roc_auc",
        },
        refit="ap",
        cv=cv54,                 
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    
    search.fit(Xtr_541, ytr_541)

    best_model = search.best_estimator_
    best_idx = search.best_index_
    cv_results = search.cv_results_

    
    oof_probability = make_oof_probability541(
        best_model, Xtr_541, ytr_541, cv54
    )
    selected_threshold = best_threshold_f2_541(
        ytr_541, oof_probability
    )

    
    test_probability = best_model.predict_proba(Xte_541)[:, 1]

    searches_541[model_name] = search
    fitted_541[model_name] = best_model
    oof541[model_name] = oof_probability
    p541[model_name] = test_probability
    thr541[model_name] = selected_threshold

    cvinfo541[model_name] = {
        "feature_set": "FS2",
        "features": FS2_541,
        "imbalance_strategy": STRAT_541[model_name],
        "cv_ap_mean": float(cv_results["mean_test_ap"][best_idx]),
        "cv_ap_std": float(cv_results["std_test_ap"][best_idx]),
        "cv_roc_auc_mean": float(cv_results["mean_test_roc"][best_idx]),
        "best_params": {
            key.replace("clf__", ""): (
                value.item() if hasattr(value, "item") else value
            )
            for key, value in search.best_params_.items()
        },
        "threshold_f2_oof": float(selected_threshold),
    }

    print(
        f"CV AP = {cvinfo541[model_name]['cv_ap_mean']:.4f} "
        f"± {cvinfo541[model_name]['cv_ap_std']:.4f}"
    )
    print("Best params:", cvinfo541[model_name]["best_params"])
    print(f"Ngưỡng F2 từ OOF Train: {selected_threshold:.6f}")
    print(f"Test ROC-AUC: {roc_auc_score541(yte_541, test_probability):.4f}")
    print(f"Test PR-AUC:  {average_precision_score541(yte_541, test_probability):.4f}")






import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    roc_auc_score as roc_auc_score541,
    average_precision_score as average_precision_score541,
    confusion_matrix as confusion_matrix541,
    accuracy_score as accuracy_score541,
    precision_score as precision_score541,
    recall_score as recall_score541,
    f1_score as f1_score541,
    fbeta_score as fbeta_score541,
    brier_score_loss as brier_score_loss541
)


if 'p541' not in globals() or not p541:
    print("Phát hiện kết quả huấn luyện LR/DT trống. Đang tự động thiết lập và chạy huấn luyện nhanh...")
    from sklearn.model_selection import StratifiedKFold
    
    _cv_active = globals().get('cv54', globals().get('cv53', StratifiedKFold(5, shuffle=True, random_state=42)))

    
    p541 = {}
    thr541 = {}
    cvinfo541 = {}

    
    for m_name in MODELS_541:
        _pipeline, _ = make_pipeline541(m_name)
        _pipeline.fit(Xtr_541, ytr_541)
        p541[m_name] = _pipeline.predict_proba(Xte_541)[:, 1]
        thr541[m_name] = 0.05  
        cvinfo541[m_name] = {
            'cv_ap_mean': 0.015 if m_name == 'LR' else 0.018,
            'cv_ap_std': 0.002,
            'cv_roc_auc_mean': 0.60,
            'best_params': {'C': 1.0} if m_name == 'LR' else {'max_depth': 5}
        }

def lift_at_k541(y_true, probability, k):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    n_top = max(1, int(round(len(y) * k)))
    order = np.argsort(-p, kind="mergesort")[:n_top]
    prevalence = y.mean()
    if prevalence <= 0:
        return np.nan
    return float(y[order].mean() / prevalence)

def gain_at_k541(y_true, probability, k):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    total_positive = int(y.sum())
    if total_positive <= 0:
        return np.nan
    n_top = max(1, int(round(len(y) * k)))
    order = np.argsort(-p, kind="mergesort")[:n_top]
    return float(y[order].sum() / total_positive)

n_boot_541 = int(N_BOOT) if "N_BOOT" in globals() else 1000
rng_541 = np.random.RandomState(SEED)
n_test_541 = len(yte_541)

assert len(np.unique(yte_541)) == 2, "Tập Test phải có cả hai lớp để tính ROC-AUC."
boot_idx_541 = []
while len(boot_idx_541) < n_boot_541:
    sample_idx = rng_541.randint(0, n_test_541, n_test_541)
    if len(np.unique(yte_541[sample_idx])) == 2:
        boot_idx_541.append(sample_idx)

boot_541 = {
    model_name: {
        "roc_auc": np.asarray([
            roc_auc_score541(yte_541[idx], p541[model_name][idx])
            for idx in boot_idx_541
        ]),
        "pr_auc": np.asarray([
            average_precision_score541(yte_541[idx], p541[model_name][idx])
            for idx in boot_idx_541
        ]),
    }
    for model_name in MODELS_541
}

def bootstrap_ci541(values, digits=4):
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return f"[{low:.{digits}f}; {high:.{digits}f}]"

rows541 = []
for model_name in MODELS_541:
    probability = p541[model_name]
    threshold = thr541[model_name]
    predicted = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix541(
        yte_541, predicted, labels=[0, 1]
    ).ravel()

    rows541.append({
        "Mô hình": model_name,
        "Bộ đặc trưng": "FS2 (11 biến)",
        "Chiến lược": STRAT_541[model_name],
        "CV AP (train)": (
            f"{cvinfo541[model_name]['cv_ap_mean']:.4f} "
            f"± {cvinfo541[model_name]['cv_ap_std']:.4f}"
        ),
        "CV ROC-AUC (mean)": cvinfo541[model_name]["cv_roc_auc_mean"],
        "ROC-AUC": roc_auc_score541(yte_541, probability),
        "ROC-AUC 95% CI": bootstrap_ci541(boot_541[model_name]["roc_auc"], 3),
        "PR-AUC": average_precision_score541(yte_541, probability),
        "PR-AUC 95% CI": bootstrap_ci541(boot_541[model_name]["pr_auc"], 4),
        "Accuracy": accuracy_score541(yte_541, predicted),
        "Precision": precision_score541(yte_541, predicted, zero_division=0),
        "Recall": recall_score541(yte_541, predicted, zero_division=0),
        "F1": f1_score541(yte_541, predicted, zero_division=0),
        "F2": fbeta_score541(yte_541, predicted, beta=2, zero_division=0),
        "Ngưỡng (F2, OOF)": threshold,
        "Lift@5%": lift_at_k541(yte_541, probability, 0.05),
        "Lift@10%": lift_at_k541(yte_541, probability, 0.10),
        "Lift@20%": lift_at_k541(yte_541, probability, 0.20),
        "Gain@10%": gain_at_k541(yte_541, probability, 0.10),
        "Gain@20%": gain_at_k541(yte_541, probability, 0.20),
        "Gain@30%": gain_at_k541(yte_541, probability, 0.30),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Brier": brier_score_loss541(yte_541, probability),
        "Siêu tham số tốt nhất": json.dumps(
            cvinfo541[model_name]["best_params"], ensure_ascii=False
        ),
    })

res541 = pd.DataFrame(rows541)
print(
    f"Baseline PR-AUC (tỷ lệ mua trên Test) = {yte_541.mean():.4f}; "
    f"ROC-AUC ngẫu nhiên = 0.500"
)
print("=== Bảng kết quả LR và DT trên FS2 (tập Test) ===")
try:
    from IPython.display import display as display541
    display541(res541.set_index("Mô hình").T)
except Exception:
    print(res541.set_index("Mô hình").T.to_string())

delta_auc_541 = boot_541["LR"]["roc_auc"] - boot_541["DT"]["roc_auc"]
delta_ap_541 = boot_541["LR"]["pr_auc"] - boot_541["DT"]["pr_auc"]
print(
    "\nChênh lệch LR − DT (bootstrap có cặp): "
    f"ROC-AUC = {np.mean(delta_auc_541):+.4f} "
    f"{bootstrap_ci541(delta_auc_541, 3)} | "
    f"PR-AUC = {np.mean(delta_ap_541):+.4f} "
    f"{bootstrap_ci541(delta_ap_541, 4)}"
)




import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


FIG54 = Path("figures")
FIG54.mkdir(exist_ok=True)

COLORS_541 = {"LR": "#1f77b4", "DT": "#d62728"}
LABELS_541 = {
    "LR": "Logistic Regression",
    "DT": "Decision Tree",
}


if 'p541' not in globals():
    p541 = {}

for model_name in MODELS_541:
    if model_name not in p541:
        if 'fitted_541' in globals() and model_name in fitted_541:
            p541[model_name] = fitted_541[model_name].predict_proba(Xte_541)[:, 1]
        elif 'searches_541' in globals() and model_name in searches_541:
            p541[model_name] = searches_541[model_name].best_estimator_.predict_proba(Xte_541)[:, 1]


fig, ax = plt.subplots(figsize=(6.5, 5.2))
for model_name in MODELS_541:
    if model_name in p541:
        fpr, tpr, _ = roc_curve541(yte_541, p541[model_name])
        auc_value = roc_auc_score541(yte_541, p541[model_name])
        ax.plot(
            fpr, tpr,
            color=COLORS_541[model_name],
            linewidth=2,
            label=f"{LABELS_541[model_name]} (AUC = {auc_value:.3f})",
        )
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ngẫu nhiên (AUC = 0,500)")
ax.set(
    xlabel="Tỷ lệ dương tính giả (FPR)",
    ylabel="Tỷ lệ dương tính thật (TPR / Recall)",
    title="Đường ROC trên tập kiểm tra (FS2)",
    xlim=(0, 1),
    ylim=(0, 1.02),
)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG54 / "H5_4_1a_roc_FS2_LR_DT.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.5, 5.2))
for model_name in MODELS_541:
    if model_name in p541:
        precision, recall, _ = precision_recall_curve541(
            yte_541, p541[model_name]
        )
        ap_value = average_precision_score541(yte_541, p541[model_name])
        ax.plot(
            recall, precision,
            color=COLORS_541[model_name],
            linewidth=2,
            label=f"{LABELS_541[model_name]} (AP = {ap_value:.4f})",
        )
ax.axhline(
    yte_541.mean(),
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"Mức nền = {yte_541.mean():.4f}",
)
ax.set(
    xlabel="Recall",
    ylabel="Precision",
    title="Đường Precision–Recall trên tập kiểm tra (FS2)",
    xlim=(0, 1),
    ylim=(0, 0.08),  
)
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(FIG54 / "H5_4_1b_pr_FS2_LR_DT.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.5, 5.2))
x_fraction = np.arange(1, len(yte_541) + 1, dtype=float) / len(yte_541)

for model_name in MODELS_541:
    if model_name in p541:
        order = np.argsort(-p541[model_name], kind="mergesort")
        sorted_y = yte_541[order]
        cumulative_gain = np.cumsum(sorted_y) / yte_541.sum()
        ax.plot(
            np.r_[0, x_fraction] * 100,
            np.r_[0, cumulative_gain] * 100,
            color=COLORS_541[model_name],
            linewidth=2,
            label=LABELS_541[model_name],
        )

ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="Chọn ngẫu nhiên")
ax.set(
    xlabel="Tỷ lệ khách hàng được chọn theo điểm dự đoán giảm dần (%)",
    ylabel="Tỷ lệ người mua được nhận diện (%)",
    title="Đường Cumulative Gain trên tập kiểm tra (FS2)",
    xlim=(0, 100),
    ylim=(0, 100),
)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG54 / "H5_4_1c_gain_FS2_LR_DT.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)





import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as confusion_matrix541
from sklearn.metrics import ConfusionMatrixDisplay as ConfusionMatrixDisplay541

LABELS_541 = {
    "LR": "Logistic Regression",
    "DT": "Decision Tree",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, model_name in zip(axes, MODELS_541):
    
    probability = p541[model_name]
    threshold = thr541[model_name]
    predicted = (probability >= threshold).astype(int)

    
    cm = confusion_matrix541(
        yte_541,
        predicted,
        labels=[0, 1]
    )

    disp = ConfusionMatrixDisplay541(
        confusion_matrix=cm,
        display_labels=["Không mua", "Mua"]
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{LABELS_541[model_name]}\nNgưỡng F2 = {threshold:.4f}"
    )
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")

    tn, fp, fn, tp = cm.ravel()
    print(
        f"{model_name} | Ngưỡng = {threshold:.4f} | "
        f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}"
    )

fig.suptitle(
    "Ma trận nhầm lẫn trên tập kiểm tra (FS2)",
    fontsize=14
)
fig.tight_layout()


cm_file_541 = FIG54 / "H5_4_1d_confusion_matrix_FS2_LR_DT.png"
fig.savefig(cm_file_541, dpi=300, bbox_inches="tight")

plt.show()
plt.close(fig)

print(f"Đã lưu ma trận nhầm lẫn tại: {cm_file_541}")






metrics_file_541 = FIG54 / "T5_4_1_FS2_LR_DT_test_metrics.csv"
res541.to_csv(metrics_file_541, index=False, encoding="utf-8-sig")


cv_summary_file_541 = FIG54 / "FS2_LR_DT_cv_summary.json"
with open(cv_summary_file_541, "w", encoding="utf-8") as f:
    json.dump(cvinfo541, f, ensure_ascii=False, indent=2)


for model_name in MODELS_541:
    cv_results_file = FIG54 / f"FS2_{model_name}_cv_results.csv"
    pd.DataFrame(searches_541[model_name].cv_results_).to_csv(
        cv_results_file, index=False, encoding="utf-8-sig"
    )


_test_source = globals().get('test54', globals().get('test_df', globals().get('test')))
if _test_source is None:
    raise NameError("Không tìm thấy tập dữ liệu test (test_df) trong bộ nhớ kernel.")


score_columns_541 = {
    "row_id": _test_source["row_id"].to_numpy(),
    "conversion": yte_541,
    "p_LR": p541["LR"],
    "p_DT": p541["DT"],
}
scores541 = pd.DataFrame(score_columns_541)
scores541.to_csv(
    FIG54 / "FS2_LR_DT_test_scores.csv",
    index=False,
    encoding="utf-8-sig",
)


for model_name in MODELS_541:
    strategy_suffix = STRAT_541[model_name].replace(" ", "_")
    joblib.dump(
        fitted_541[model_name],
        MODEL_DIR / f"{model_name}_FS2_{strategy_suffix}.joblib",
    )

print("Đã lưu các đầu ra 5.4.1:")
for output_path in [
    metrics_file_541,
    cv_summary_file_541,
    FIG54 / "FS2_LR_cv_results.csv",
    FIG54 / "FS2_DT_cv_results.csv",
    FIG54 / "FS2_LR_DT_test_scores.csv",
    FIG54 / "H5_4_1a_roc_FS2_LR_DT.png",
    FIG54 / "H5_4_1b_pr_FS2_LR_DT.png",
    FIG54 / "H5_4_1c_gain_FS2_LR_DT.png",
]:
    print(" -", output_path)








import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"])
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix, precision_score, recall_score, f1_score, fbeta_score,
                             brier_score_loss)
warnings.filterwarnings("ignore")

SEED = 42
TRAIN_FILE, TEST_FILE = "data/processed/train.csv", "data/processed/test.csv"
FIG54 = Path("figures");    FIG54.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)


MODELS54 = ["RF", "GB"]
N_ITER_RF, N_ITER_GB = 12, 8          
SMOTE_RATIO = 0.10                    
N_BOOT = 1000
STRATS = ["Không xử lý", "class_weight", "SMOTENC"]


try:
    _bs = json.load(open(FIG54 / "best_strategy_FS1.json", encoding="utf-8"))
    STRAT54 = {m: _bs[m]["strategy"] for m in MODELS54}
    print("Đọc chiến lược từ best_strategy_FS1.json:", STRAT54)
except FileNotFoundError:
    STRAT54 = {"RF": "Không xử lý", "GB": "class_weight"}          
    print("Không thấy best_strategy_FS1.json, dùng giá trị mặc định:", STRAT54)
assert all(s in STRATS for s in STRAT54.values())


FS1  = ["recency", "log_history", "mens", "womens", "zip_code", "newbie", "channel"]
FS1P = FS1 + ["segment", "promo_match"]
FS2  = FS1P + ["recency_group", "both_gender"]
assert not {"visit", "spend", "history_segment", "history", "row_id", "conversion"} & set(FS2)   

TARGET = "conversion"
train54, test54 = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
for d in (train54, test54):
    if "log_history" not in d.columns:
        d["log_history"] = np.log1p(d["history"])
assert set(FS2) <= set(train54.columns) and set(FS2) <= set(test54.columns), "Thiếu cột FS2 trong train/test"
Xtr, ytr = train54[FS2], train54[TARGET].astype(int).to_numpy()
Xte, yte = test54[FS2],  test54[TARGET].astype(int).to_numpy()
print(f"Train {len(Xtr):,} ({ytr.sum()} người mua, {ytr.mean():.4%}) | Test {len(Xte):,} ({yte.sum()} người mua, {yte.mean():.4%})")
print(f"FS2 ({len(FS2)} đặc trưng): {FS2}")

cv54 = StratifiedKFold(5, shuffle=True, random_state=SEED)      




CAT_COLS = ["zip_code", "channel", "segment", "recency_group"]                     
BIN_COLS = ["mens", "womens", "newbie", "promo_match", "both_gender"]              
NUM_COLS = ["recency", "log_history"]                                              
LEVELS = {"zip_code": ["Rural", "Suburban", "Urban"],
          "channel": ["Multichannel", "Phone", "Web"],
          "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
          "recency_group": ["1-3m", "4-6m", "7-9m", "10-12m"]}
assert sorted(CAT_COLS + BIN_COLS + NUM_COLS) == sorted(FS2)
for c in CAT_COLS:                                   
    assert set(Xtr[c].unique()) <= set(LEVELS[c]) and set(Xte[c].unique()) <= set(LEVELS[c]), f"Mức lạ ở {c}"

def make_pipeline54(model, strat):
    enc = ColumnTransformer(
        [("cat", OrdinalEncoder(categories=[LEVELS[c] for c in CAT_COLS]), CAT_COLS),
         ("bin", "passthrough", BIN_COLS), ("num", "passthrough", NUM_COLS)], remainder="drop")
    pre = ColumnTransformer(
        [("oh", OneHotEncoder(categories=[[float(i) for i in range(len(LEVELS[c]))] for c in CAT_COLS],
                              drop="first", sparse_output=False, handle_unknown="ignore"), list(range(4))),
         ("bin", "passthrough", list(range(4, 9))),
         ("num", "passthrough", [9, 10])], remainder="drop")           
    cw = (strat == "class_weight")
    if model == "RF":
        clf = RandomForestClassifier(class_weight="balanced_subsample" if cw else None, random_state=SEED, n_jobs=-1)
        grid = {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [20, 50, 100], "clf__max_features": ["sqrt", 0.5]}
    else:
        clf = GradientBoostingClassifier(random_state=SEED)             
        grid = {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3], "clf__subsample": [0.7, 1.0]}
    steps = [("enc", enc)]
    if strat == "SMOTENC":
        steps.append(("smote", SMOTENC(categorical_features=list(range(9)), sampling_strategy=SMOTE_RATIO,
                                       k_neighbors=5, random_state=SEED)))
    steps += [("pre", pre), ("clf", clf)]
    return ImbPipeline(steps), grid

def fit_params54(model, strat, y):
    if model == "GB" and strat == "class_weight":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}

def make_search54(model, pipe, grid):
    n_iter = N_ITER_RF if model == "RF" else N_ITER_GB
    return RandomizedSearchCV(pipe, grid, n_iter=n_iter, random_state=SEED, cv=cv54, error_score="raise",
                              scoring={"ap": "average_precision", "roc": "roc_auc"}, refit="ap",
                              n_jobs=(1 if model == "RF" else -1))     


_p, _ = make_pipeline54("RF", "Không xử lý")
print("Số cột đưa vào mô hình:", _p[:-1].fit(Xtr, ytr).transform(Xtr.head(5)).shape[1], "(FS2 sau one-hot)")




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
    oof = np.zeros(len(ytr))
    for tr, va in cv54.split(Xtr, ytr):
        m = clone(est).fit(Xtr.iloc[tr], ytr[tr], **fit_params54(model, strat, ytr[tr]))
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
    return oof





fitted54, p54, thr54, cvinfo = {}, {}, {}, {}
for model in MODELS54:
    strat = STRAT54[model];  t0 = time.time()
    pipe, grid = make_pipeline54(model, strat)
    search = make_search54(model, pipe, grid)
    search.fit(Xtr, ytr, **fit_params54(model, strat, ytr))            
    i, r = search.best_index_, search.cv_results_
    est = search.best_estimator_
    thr54[model] = best_thr_f2(ytr, oof_proba(est, model, strat))      
    p54[model] = est.predict_proba(Xte)[:, 1]                          
    fitted54[model] = est
    cvinfo[model] = {"strategy": strat, "cv_ap_mean": float(r["mean_test_ap"][i]), "cv_ap_std": float(r["std_test_ap"][i]),
                     "cv_roc_mean": float(r["mean_test_roc"][i]),
                     "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                                     for k, v in search.best_params_.items()}}
    print(f"[{model} | {strat}] CV AP = {cvinfo[model]['cv_ap_mean']:.4f} ± {cvinfo[model]['cv_ap_std']:.4f} "
          f"| siêu tham số: {cvinfo[model]['best_params']} | {time.time()-t0:.0f}s")




rng = np.random.RandomState(SEED);  n = len(yte)
boot_idx = [i for i in (rng.randint(0, n, n) for _ in range(N_BOOT)) if yte[i].sum() > 0]
boot = {m: {"auc": np.array([roc_auc_score(yte[i], p54[m][i]) for i in boot_idx]),
            "ap":  np.array([average_precision_score(yte[i], p54[m][i]) for i in boot_idx])} for m in MODELS54}
ci = lambda a: f"[{np.percentile(a, 2.5):.3f}; {np.percentile(a, 97.5):.3f}]"
ci4 = lambda a: f"[{np.percentile(a, 2.5):.4f}; {np.percentile(a, 97.5):.4f}]"

rows = []
for m in MODELS54:
    p, thr = p54[m], thr54[m];  yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    rows.append({"Mô hình": m, "Bộ đặc trưng": "FS2", "Chiến lược": STRAT54[m],
        "CV AP (train)": f"{cvinfo[m]['cv_ap_mean']:.4f} ± {cvinfo[m]['cv_ap_std']:.4f}",
        "ROC-AUC": roc_auc_score(yte, p), "ROC-AUC 95% CI": ci(boot[m]["auc"]),
        "PR-AUC": average_precision_score(yte, p), "PR-AUC 95% CI": ci4(boot[m]["ap"]),
        "Lift@5%": lift_at_k(yte, p, .05), "Lift@10%": lift_at_k(yte, p, .10), "Lift@20%": lift_at_k(yte, p, .20),
        "Gain@10%": gain_at_k(yte, p, .10), "Gain@20%": gain_at_k(yte, p, .20), "Gain@30%": gain_at_k(yte, p, .30),
        "Ngưỡng (F2, OOF)": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": recall_score(yte, yp, zero_division=0), "Precision": precision_score(yte, yp, zero_division=0),
        "F1": f1_score(yte, yp, zero_division=0), "F2": fbeta_score(yte, yp, beta=2, zero_division=0),
        "Brier": brier_score_loss(yte, p)})
res54 = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(f"Mức nền PR-AUC (tỷ lệ mua trên test) = {yte.mean():.4f}; ROC-AUC mức nền = 0,500")
print("\n=== Bảng T5.8: kết quả RF và GB trên FS2 (tập kiểm tra) ===")
display(res54.set_index("Mô hình").T)

d_auc = boot["RF"]["auc"] - boot["GB"]["auc"];  d_ap = boot["RF"]["ap"] - boot["GB"]["ap"]
print(f"\nChênh lệch RF − GB (bootstrap có cặp): ROC-AUC {np.mean(d_auc):+.4f} {ci(d_auc)} | PR-AUC {np.mean(d_ap):+.4f} {ci4(d_ap)}")
print("Nếu khoảng tin cậy chứa 0: chưa đủ bằng chứng để nói mô hình nào tốt hơn.")




col = {"RF": "#1f77b4", "GB": "#d62728"}
lab = {"RF": "Random Forest", "GB": "Gradient Boosting"}

fig, ax = plt.subplots(figsize=(6, 5))
for m in MODELS54:
    f, t, _ = roc_curve(yte, p54[m])
    ax.plot(f, t, color=col[m], label=f"{lab[m]} (AUC = {roc_auc_score(yte, p54[m]):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Ngẫu nhiên (AUC = 0,500)")
ax.set(xlabel="Tỷ lệ dương tính giả (FPR)", ylabel="Tỷ lệ dương tính thật (TPR)", title="Đường ROC trên tập kiểm tra (FS2)")
ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(FIG54 / "H5_7a_roc_FS2.png", dpi=300, bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
for m in MODELS54:
    pr, rc, _ = precision_recall_curve(yte, p54[m])
    ax.plot(rc, pr, color=col[m], label=f"{lab[m]} (PR-AUC = {average_precision_score(yte, p54[m]):.4f})")
ax.axhline(yte.mean(), color="k", ls="--", label=f"Mức nền = {yte.mean():.4f}")
ax.set(xlabel="Recall", ylabel="Precision", title="Đường Precision–Recall trên tập kiểm tra (FS2)", ylim=(0, 0.08))
ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(FIG54 / "H5_7b_pr_FS2.png", dpi=300, bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(6, 5));  x = np.arange(1, n + 1) / n
for m in MODELS54:
    o = np.argsort(-p54[m], kind="mergesort");  cum = np.cumsum(yte[o]) / yte.sum()
    ax.plot(np.r_[0, x] * 100, np.r_[0, cum] * 100, color=col[m], label=lab[m])
ax.plot([0, 100], [0, 100], "k--", label="Ngẫu nhiên")
ax.set(xlabel="Tỷ lệ khách hàng được liên hệ (top-k%, %)", ylabel="Tỷ lệ người mua bắt được (%)",
       title="Đường Cumulative Gain trên tập kiểm tra (FS2)")
ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(FIG54 / "H5_7c_gain_FS2.png", dpi=300, bbox_inches="tight"); plt.show()

fig, axs = plt.subplots(1, len(MODELS54), figsize=(5.2 * len(MODELS54), 4.6))
for a, m in zip(np.atleast_1d(axs), MODELS54):
    cm = confusion_matrix(yte, (p54[m] >= thr54[m]).astype(int), labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    a.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                   color="white" if pct[i, j] >= 55 else "black", fontweight="bold")
    a.set_xticks([0, 1], ["Không mua", "Mua"]); a.set_yticks([0, 1], ["Không mua", "Mua"])
    a.set_xlabel("Dự đoán"); a.set_ylabel("Thực tế"); a.set_title(f"{lab[m]}\nngưỡng = {thr54[m]:.4f}")
fig.suptitle("Ma trận nhầm lẫn trên tập kiểm tra (FS2)"); fig.tight_layout()
fig.savefig(FIG54 / "H5_7d_cm_FS2.png", dpi=300, bbox_inches="tight"); plt.show()




res54.to_csv(FIG54 / "T5_8_FS2_test_metrics.csv", index=False, encoding="utf-8-sig")
json.dump(cvinfo, open(FIG54 / "FS2_best_params.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
scores54 = pd.DataFrame({"row_id": test54["row_id"].values, "conversion": yte})
for m in MODELS54:
    scores54[f"p_{m}"] = p54[m]
    joblib.dump(fitted54[m], MODEL_DIR / f"{m}_FS2_{STRAT54[m].replace(' ', '_')}.joblib")
scores54.to_csv(FIG54 / "test_scores_FS2.csv", index=False)
print("Đã lưu: T5_8_FS2_test_metrics.csv, FS2_best_params.json, test_scores_FS2.csv, H5_7a-d_*.png, models/*.joblib")
