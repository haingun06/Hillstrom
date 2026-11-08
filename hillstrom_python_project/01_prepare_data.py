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
# 01 - PREPARE DATA
# ==============================================================================


# --- Notebook Code cell 1 ---
# 0. Thư viện và cấu hình
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, fbeta_score,
                             precision_recall_curve)

try:
    from IPython.display import display
except ImportError:
    display = print

RANDOM_STATE = 42
FIG_DIR = Path('figures');            FIG_DIR.mkdir(exist_ok=True)
PROC_DIR = Path('data') / 'processed'; PROC_DIR.mkdir(parents=True, exist_ok=True)

# Cột bị cấm làm đặc trưng
OUTCOME_COLS = ['visit', 'spend']                       # xảy ra SAU email
LEAKAGE_COLS = OUTCOME_COLS + ['history_segment']       # + trùng thông tin với history

print('scikit-learn', sklearn.__version__, '| pandas', pd.__version__, '| numpy', np.__version__)


# --- Notebook Code cell 2 ---
from pathlib import Path
import shutil

print("Thư mục làm việc:", Path.cwd())

if Path("hillstrom.csv").exists():
    print("Dữ liệu đã nằm đúng vị trí.")
elif Path("/hillstrom.csv").exists():
    shutil.copy2("/hillstrom.csv", "hillstrom.csv")
    print("Đã sao chép hillstrom.csv vào /content.")
else:
    print("Chưa tìm thấy hillstrom.csv tại hai vị trí đã kiểm tra.")


# --- Notebook Markdown cell 3 ---
# ## 1. Nạp dữ liệu, kiểm tra chất lượng và làm sạch
# Thứ tự quan trọng: **khử trùng lặp trên đủ 12 cột gốc trước**, rồi mới xét việc loại cột. Mỗi dòng được gán `row_id` (vị trí trong file gốc) để về sau nối lại `visit`/`spend` mà không cần ID khách hàng.


# --- Notebook Code cell 4 ---
def find_hillstrom_csv():
    candidates = ['hillstrom.csv', 'data/hillstrom.csv', 'data/raw/hillstrom.csv',
                  '../hillstrom.csv', '/content/hillstrom.csv']
    for p in candidates:
        if os.path.exists(p):
            return p
    print('Không tìm thấy hillstrom.csv cục bộ. Đang tải từ MineThatData...')
    import urllib.request
    url = 'http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp, open('hillstrom.csv', 'wb') as f:
        f.write(resp.read())
    return 'hillstrom.csv'

raw_file = find_hillstrom_csv()
df_raw = pd.read_csv(raw_file)

RAW_COLS = ['recency', 'history_segment', 'history', 'mens', 'womens', 'zip_code',
            'newbie', 'channel', 'segment', 'visit', 'conversion', 'spend']
assert set(RAW_COLS) <= set(df_raw.columns), 'Thiếu cột so với dữ liệu Hillstrom gốc'
df_raw = df_raw[RAW_COLS].copy()
df_raw.insert(0, 'row_id', np.arange(len(df_raw)))     # khoá nối, KHÔNG phải đặc trưng

print(f'Nạp từ: {raw_file} | {len(df_raw):,} dòng x {len(RAW_COLS)} cột | giá trị thiếu: {int(df_raw[RAW_COLS].isna().sum().sum())}')

# --- Khử trùng lặp trên đủ 12 cột gốc (không tính row_id) ---
df_clean = df_raw.drop_duplicates(subset=RAW_COLS).copy()
n_dup = len(df_raw) - len(df_clean)
print(f'Sau khử trùng lặp: {len(df_clean):,} dòng (loại {n_dup:,} dòng trùng)')

# --- Sửa chính tả ---
df_clean['zip_code'] = df_clean['zip_code'].replace({'Surburban': 'Suburban'})
assert set(df_clean['zip_code'].unique()) == {'Rural', 'Suburban', 'Urban'}

# --- Kiểm tra các quy tắc dữ liệu đã nêu trong kế hoạch ---
assert ((df_clean['conversion'] == 1) == (df_clean['spend'] > 0)).all(), 'Vi phạm: conversion=1 <=> spend>0'
assert (df_clean.loc[df_clean['conversion'] == 1, 'visit'] == 1).all(), 'Vi phạm: conversion=1 => visit=1'
assert df_clean['recency'].between(1, 12).all()
assert ((df_clean['mens'] + df_clean['womens']) >= 1).all()

# --- Bảng T4.1 trước/sau làm sạch ---
t41 = pd.DataFrame({
    'Chỉ tiêu': ['Số dòng', 'Số người mua (conversion=1)', 'Tỷ lệ conversion (%)'],
    'Trước': [len(df_raw), int(df_raw['conversion'].sum()), round(df_raw['conversion'].mean() * 100, 3)],
    'Sau':   [len(df_clean), int(df_clean['conversion'].sum()), round(df_clean['conversion'].mean() * 100, 3)],
})
display(t41)
t41.to_csv(FIG_DIR / 'T4_1_cleaning_summary.csv', index=False, encoding='utf-8-sig')

print('\nSố khách mỗi nhóm segment sau làm sạch:')
display(df_clean['segment'].value_counts().to_frame('Số khách'))

# Lưu dữ liệu sạch ĐẦY ĐỦ (có visit/spend/history_segment) để tra cứu; KHÔNG dùng trực tiếp làm X
df_clean.to_csv(PROC_DIR / 'hillstrom_cleaned.csv', index=False)
print(f"\nĐã lưu: {PROC_DIR / 'hillstrom_cleaned.csv'}")


# --- Notebook Markdown cell 5 ---
# ## 2. Xây dựng đặc trưng
# `recency_group` chia `recency` (1–12) thành 4 nhóm. Lưu ý đây là phiên bản chia nhóm của `recency`, nên hai biến trùng một phần thông tin; điều này cần nêu ở phần hạn chế khi diễn giải hệ số Logistic Regression.


# --- Notebook Code cell 6 ---
def engineer_features(df):
    out = df.copy()
    out['log_history'] = np.log1p(out['history'])
    out['promo_match'] = (((out['segment'] == 'Mens E-Mail')   & (out['mens'] == 1)) |
                          ((out['segment'] == 'Womens E-Mail') & (out['womens'] == 1))).astype(int)
    out['both_gender'] = ((out['mens'] == 1) & (out['womens'] == 1)).astype(int)
    out['recency_group'] = pd.cut(out['recency'], bins=[0, 3, 6, 9, 12],
                                  labels=['0-3m', '4-6m', '7-9m', '10-12m']).astype(str)
    assert not (out['recency_group'] == 'nan').any(), 'recency nằm ngoài khoảng 1-12'
    return out

df_feat = engineer_features(df_clean)

# --- Minh chứng định lượng cho both_gender (kèm kiểm định chi-square) ---
bg1, bg0 = df_feat[df_feat['both_gender'] == 1], df_feat[df_feat['both_gender'] == 0]
tab = pd.crosstab(df_feat['both_gender'], df_feat['conversion'])
chi2, p_val, _, _ = chi2_contingency(tab)

print('=== both_gender: khách đã mua cả hàng nam và nữ ===')
print(f'Số khách: {len(bg1):,} ({len(bg1)/len(df_feat):.2%})')
print(f"history trung bình: {bg1['history'].mean():.2f} USD (median {bg1['history'].median():.2f}) "
      f"vs {bg0['history'].mean():.2f} USD (median {bg0['history'].median():.2f}) "
      f"-> gấp {bg1['history'].mean()/bg0['history'].mean():.2f} lần")
print(f"conversion: {bg1['conversion'].mean():.4%} ({bg1['conversion'].sum()}/{len(bg1):,}) "
      f"vs {bg0['conversion'].mean():.4%} ({bg0['conversion'].sum()}/{len(bg0):,}) "
      f"-> gấp {bg1['conversion'].mean()/bg0['conversion'].mean():.2f} lần | chi-square p = {p_val:.2e}")
diff_rec = bg0['recency'].mean() - bg1['recency'].mean()
print(f"Số tháng kể từ lần mua gần nhất (recency): {bg1['recency'].mean():.2f} vs {bg0['recency'].mean():.2f} "
      f"-> chênh lệch (đơn ngành - both_gender) = {diff_rec:+.2f} tháng")
print('Lưu ý: đây là so sánh đơn biến, chưa kiểm soát nhiễu (both_gender có thể đi cùng history cao).')

# --- Hình H4.2: history trước/sau log ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df_feat['history'], bins=50, color='tab:blue', alpha=.8)
ax[0].set_title('history (gốc, lệch phải)'); ax[0].set_xlabel('USD'); ax[0].set_ylabel('Số khách')
ax[1].hist(df_feat['log_history'], bins=50, color='tab:green', alpha=.8)
ax[1].set_title('log_history = log(1 + history)'); ax[1].set_xlabel('log(1 + USD)')
fig.tight_layout(); fig.savefig(FIG_DIR / 'H4_2_history_log.png', dpi=300, bbox_inches='tight'); plt.show()


# --- Notebook Code cell 7 ---
from pathlib import Path
import pandas as pd

p = Path("hillstrom.csv")

print("Đã tìm thấy file:", p.exists())

if p.exists():
    df = pd.read_csv(p)
    print("Kích thước dữ liệu:", df.shape)


# --- Notebook Markdown cell 8 ---
# ## 3. Các bộ đặc trưng FS1, FS1P, FS2
# - **FS1** (7): hồ sơ khách hàng.
# - **FS1P** (9, tuỳ chọn): FS1 + `segment` + `promo_match`. Dùng để tách riêng tác động của khuyến mãi khi trả lời SQ1; nếu chỉ so FS1 với FS2 thì mức tăng chung của 4 biến mới không quy được riêng cho khuyến mãi.
# - **FS2** (11): FS1P + `both_gender` + `recency_group`.


# --- Notebook Code cell 9 ---
from pathlib import Path

hillstrom_path = Path("hillstrom.csv")
if hillstrom_path.exists():
    print(f"Tệp đã tồn tại sẵn tại: {hillstrom_path} | Kích thước: {hillstrom_path.stat().st_size:,} bytes")
else:
    print("Không tìm thấy tệp /content/hillstrom.csv. Vui lòng chạy lại các bước tải dữ liệu ở trên.")


# --- Notebook Code cell 10 ---
from pathlib import Path
import pandas as pd

base = Path(".")

for name in [
    "hillstrom.csv",
    "data/processed/train.csv",
    "data/processed/test.csv"
]:
    path = base / name
    print(f"{name}: {'ĐÃ CÓ' if path.exists() else 'CHƯA CÓ'}")


# --- Notebook Code cell 11 ---
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "hillstrom.csv"

df = df_clean.copy()
print("Kích thước dữ liệu:", df.shape)


# --- Notebook Code cell 12 ---
FS1  = ['recency', 'log_history', 'mens', 'womens', 'zip_code', 'newbie', 'channel']
FS1P = FS1 + ['segment', 'promo_match']
FS2  = FS1P + ['both_gender', 'recency_group']
FEATURE_SETS = {'FS1': FS1, 'FS1P': FS1P, 'FS2': FS2}

# Kiểm tra chống rò rỉ TRÊN DANH SÁCH ĐẶC TRƯNG (không phải trên dataframe đã drop)
for name, cols in FEATURE_SETS.items():
    assert not set(cols) & set(LEAKAGE_COLS), f'Rò rỉ trong {name}: {set(cols) & set(LEAKAGE_COLS)}'
    assert 'row_id' not in cols and 'conversion' not in cols
    assert len(cols) == len(set(cols)) and set(cols) <= set(df_feat.columns)
    print(f'{name} ({len(cols)} đặc trưng): {cols}')
print('\nĐã kiểm tra: không có visit, spend, history_segment, row_id, conversion trong X.')

t42 = pd.DataFrame({'Đặc trưng': FS2,
                    'Có trong FS1': [c in FS1 for c in FS2],
                    'Có trong FS1P': [c in FS1P for c in FS2],
                    'Có trong FS2': True})
display(t42); t42.to_csv(FIG_DIR / 'T4_2_feature_sets.csv', index=False, encoding='utf-8-sig')


# --- Notebook Markdown cell 13 ---
# ## 4. Chia train/test 80/20 phân tầng và lưu dữ liệu không rò rỉ
# - `train.csv`, `test.csv`: chỉ gồm `row_id`, đặc trưng của FS2 và `conversion`.
# - `outcomes_train.csv`, `outcomes_test.csv`: `row_id`, `segment`, `conversion`, `visit`, `spend`. Dùng ở Chương 5 để mô phỏng lợi nhuận bằng doanh thu thực, **không bao giờ đưa vào X**.


# --- Notebook Code cell 14 ---
train_df, test_df = train_test_split(df_feat, test_size=0.20, random_state=RANDOM_STATE,
                                     stratify=df_feat['conversion'])

MODEL_COLS   = ['row_id'] + FS2 + ['conversion']
OUTCOME_FILE = ['row_id', 'segment', 'conversion'] + OUTCOME_COLS

train_df[MODEL_COLS].to_csv(PROC_DIR / 'train.csv', index=False)
test_df[MODEL_COLS].to_csv(PROC_DIR / 'test.csv', index=False)
train_df[OUTCOME_FILE].to_csv(PROC_DIR / 'outcomes_train.csv', index=False)
test_df[OUTCOME_FILE].to_csv(PROC_DIR / 'outcomes_test.csv', index=False)

# Kiểm tra trên CHÍNH tệp đã lưu
for f in ['train.csv', 'test.csv']:
    cols = pd.read_csv(PROC_DIR / f, nrows=1).columns
    assert not set(cols) & set(LEAKAGE_COLS), f'Rò rỉ trong {f}'
assert set(train_df['row_id']).isdisjoint(set(test_df['row_id']))

y_train, y_test = train_df['conversion'], test_df['conversion']
t44 = pd.DataFrame({
    'Tập': ['Train', 'Test', 'Tổng'],
    'Số dòng': [len(train_df), len(test_df), len(df_feat)],
    'Người mua': [int(y_train.sum()), int(y_test.sum()), int(df_feat['conversion'].sum())],
    'Tỷ lệ mua (%)': [round(y_train.mean()*100, 3), round(y_test.mean()*100, 3), round(df_feat['conversion'].mean()*100, 3)],
})
display(t44); t44.to_csv(FIG_DIR / 'T4_4_split_summary.csv', index=False, encoding='utf-8-sig')

seg_split = pd.concat([train_df['segment'].value_counts().rename('Train'),
                       test_df['segment'].value_counts().rename('Test')], axis=1)
display(seg_split)


# --- Notebook Markdown cell 15 ---
# ## 5. Mã hoá, chuẩn hoá và pipeline dùng chung
# Một hàm duy nhất tạo `ColumnTransformer` cho mọi bộ đặc trưng: biến số → `StandardScaler`, biến nhị phân giữ nguyên, biến phân loại → `OneHotEncoder(drop='first')` với thứ tự mức cố định (mức tham chiếu: Rural, Multichannel, No E-Mail, 0-3m). Mọi bước học từ dữ liệu nằm trong Pipeline nên chỉ fit trên train.


# --- Notebook Code cell 16 ---
NUM_COLS = ['recency', 'log_history']
CAT_LEVELS = {
    'zip_code':      ['Rural', 'Suburban', 'Urban'],
    'channel':       ['Multichannel', 'Phone', 'Web'],
    'segment':       ['No E-Mail', 'Mens E-Mail', 'Womens E-Mail'],
    'recency_group': ['0-3m', '4-6m', '7-9m', '10-12m'],
}

def make_preprocessor(features, scale=True):
    num = [c for c in features if c in NUM_COLS]
    cat = [c for c in features if c in CAT_LEVELS]
    binary = [c for c in features if c not in num + cat]
    parts = []
    if num:    parts.append(('num', StandardScaler() if scale else 'passthrough', num))
    if binary: parts.append(('bin', 'passthrough', binary))
    if cat:    parts.append(('cat', OneHotEncoder(categories=[CAT_LEVELS[c] for c in cat],
                                                  drop='first', sparse_output=False), cat))
    return ColumnTransformer(parts, remainder='drop')

def make_lr(features, class_weight='balanced', C=1.0):
    return Pipeline([('pre', make_preprocessor(features)),
                     ('clf', LogisticRegression(C=C, class_weight=class_weight,
                                                max_iter=2000, random_state=RANDOM_STATE))])

# Ví dụ dữ liệu sau biến đổi (Bảng T4.3)
pre_demo = make_preprocessor(FS2).fit(train_df[FS2])
names = [n.split('__', 1)[1] for n in pre_demo.get_feature_names_out()]
demo = pd.DataFrame(pre_demo.transform(train_df[FS2].head(5)), columns=names).round(3)
print(f'FS2 sau biến đổi: {len(names)} cột')
display(demo)
demo.to_csv(FIG_DIR / 'T4_3_onehot_example.csv', index=False, encoding='utf-8-sig')


# --- Notebook Markdown cell 17 ---
# ## 6. Mô hình cơ sở: Dummy và Logistic Regression
# Cách đánh giá đã sửa so với bản cũ:
# - **Ngưỡng không cố định 0,5.** Xác suất dự đoán chỉ cỡ 1–5% nên ngưỡng 0,5 không bắt được ca dương nào. Ngưỡng được chọn bằng cách tối đa F2 trên dự đoán out-of-fold của **tập train** (5-fold), rồi áp lên test đúng một lần.
# - Chỉ số chính không phụ thuộc ngưỡng: ROC-AUC, PR-AUC, Lift@10%/20%, có khoảng tin cậy bootstrap 95%.
# - Accuracy chỉ để minh hoạ cái bẫy của dữ liệu mất cân bằng.


# --- Notebook Code cell 18 ---
def lift_at_k(y_true, p, k=0.10):
    y = np.asarray(y_true)
    if np.ptp(p) == 0:            # điểm hằng số (Dummy): không phân biệt được ai
        return 1.0
    n = max(1, int(round(len(y) * k)))
    idx = np.argsort(-np.asarray(p), kind='mergesort')[:n]
    return y[idx].mean() / y.mean()

def bootstrap_ci(y_true, p, n_boot=1000, seed=RANDOM_STATE):
    y, p = np.asarray(y_true), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        aps.append(average_precision_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])

def choose_threshold_f2(pipe, X_tr, y_tr, cv):
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f2 = 5 * prec * rec / (4 * prec + rec + 1e-12)
    return float(thr[np.argmax(f2[:-1])])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted, test_scores = [], {}, {}

def add_row(name, fs, y_true, prob, pred, thr):
    ci_auc, ci_ap = bootstrap_ci(y_true, prob)
    rows.append({
        'Mô hình': name, 'Bộ đặc trưng': fs,
        'ROC-AUC': roc_auc_score(y_true, prob), 'ROC-AUC 95% CI': f'[{ci_auc[0]:.3f}; {ci_auc[1]:.3f}]',
        'PR-AUC': average_precision_score(y_true, prob), 'PR-AUC 95% CI': f'[{ci_ap[0]:.4f}; {ci_ap[1]:.4f}]',
        'Lift@10%': lift_at_k(y_true, prob, 0.10), 'Lift@20%': lift_at_k(y_true, prob, 0.20),
        'Ngưỡng': thr,
        'Precision': precision_score(y_true, pred, zero_division=0),
        'Recall': recall_score(y_true, pred, zero_division=0),
        'F1': f1_score(y_true, pred, zero_division=0),
        'F2': fbeta_score(y_true, pred, beta=2, zero_division=0),
        'Accuracy (chỉ minh hoạ)': accuracy_score(y_true, pred),
    })

# --- Dummy (prior) ---
dummy = DummyClassifier(strategy='prior', random_state=RANDOM_STATE).fit(train_df[FS1], y_train)
p_d = dummy.predict_proba(test_df[FS1])[:, 1]
add_row('Dummy (prior)', 'FS1', y_test, p_d, dummy.predict(test_df[FS1]), np.nan)

# --- Logistic Regression trên 3 bộ đặc trưng ---
for fs_name, cols in FEATURE_SETS.items():
    pipe = make_lr(cols)
    thr = choose_threshold_f2(pipe, train_df[cols], y_train, cv)   # chỉ dùng train
    pipe.fit(train_df[cols], y_train)
    prob = pipe.predict_proba(test_df[cols])[:, 1]                 # test dùng một lần
    add_row('Logistic Regression (balanced)', fs_name, y_test, prob, (prob >= thr).astype(int), thr)
    fitted[fs_name], test_scores[fs_name] = pipe, prob

res = pd.DataFrame(rows)
res.to_csv(FIG_DIR / 'T4_5_baseline_results.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
display(res.round(4))
print(f'PR-AUC mức nền (đoán ngẫu nhiên) = tỷ lệ lớp dương test = {y_test.mean():.4f}')


# --- Notebook Code cell 19 ---
# Minh chứng: Logistic Regression mặc định (không class_weight) cho xác suất rất thấp -> ngưỡng 0,5 vô dụng
for fs_name in ['FS1', 'FS2']:
    cols = FEATURE_SETS[fs_name]
    m = make_lr(cols, class_weight=None).fit(train_df[cols], y_train)
    p = m.predict_proba(test_df[cols])[:, 1]
    print(f'{fs_name} mặc định: p_min = {p.min():.5f}, p_max = {p.max():.5f}, số ca dự báo dương tại 0,5 = {(p >= 0.5).sum()}')


# --- Notebook Markdown cell 20 ---
# ## 7. Minh hoạ rò rỉ dữ liệu
# Chạy **một lần** để đưa vào báo cáo. Thêm `visit` vào X sẽ làm AUC tăng giả tạo. Kết quả này **không** dùng cho bất kỳ kết luận nào.


# --- Notebook Code cell 21 ---
RUN_LEAKAGE_DEMO = True
if RUN_LEAKAGE_DEMO:
    visit_tr = pd.read_csv(PROC_DIR / 'outcomes_train.csv').set_index('row_id')['visit']
    visit_te = pd.read_csv(PROC_DIR / 'outcomes_test.csv').set_index('row_id')['visit']
    Xtr_leak = train_df[FS2].assign(visit=visit_tr.loc[train_df['row_id']].values)
    Xte_leak = test_df[FS2].assign(visit=visit_te.loc[test_df['row_id']].values)
    m_leak = make_lr(FS2 + ['visit']).fit(Xtr_leak, y_train)
    p_leak = m_leak.predict_proba(Xte_leak)[:, 1]
    print(f'[CHỈ MINH HOẠ] FS2 + visit: ROC-AUC = {roc_auc_score(y_test, p_leak):.4f}, PR-AUC = {average_precision_score(y_test, p_leak):.4f}')
    print(f"FS2 hợp lệ (không rò rỉ): ROC-AUC = {roc_auc_score(y_test, test_scores['FS2']):.4f}")


# --- Notebook Markdown cell 22 ---
# ## 8. Tổng kết đầu ra của Chương 4
# - `data/processed/hillstrom_cleaned.csv`: dữ liệu sạch đầy đủ (để tra cứu).
# - `data/processed/train.csv`, `test.csv`: đặc trưng + `conversion` (không rò rỉ).
# - `data/processed/outcomes_train.csv`, `outcomes_test.csv`: `visit`, `spend` nối được bằng `row_id` (dùng cho mô phỏng lợi nhuận ở Chương 5).
# - `figures/T4_1`–`T4_5`, `H4_2`: bảng và hình cho báo cáo.


# --- Notebook Code cell 23 ---
print('Các tệp đã tạo:')
for p in sorted(list(PROC_DIR.glob('*.csv')) + list(FIG_DIR.glob('*'))):
    print(' -', p)
