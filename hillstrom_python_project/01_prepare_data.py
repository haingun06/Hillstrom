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


OUTCOME_COLS = ['visit', 'spend']                       
LEAKAGE_COLS = OUTCOME_COLS + ['history_segment']       

print('scikit-learn', sklearn.__version__, '| pandas', pd.__version__, '| numpy', np.__version__)



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
df_raw.insert(0, 'row_id', np.arange(len(df_raw)))     

print(f'Nạp từ: {raw_file} | {len(df_raw):,} dòng x {len(RAW_COLS)} cột | giá trị thiếu: {int(df_raw[RAW_COLS].isna().sum().sum())}')


df_deduplicated = df_raw.drop_duplicates(subset=RAW_COLS).copy()
df_deduplicated['zip_code'] = df_deduplicated['zip_code'].replace({'Surburban': 'Suburban'})   # cùng quy tắc sửa chính tả với mẫu chính
n_dup = len(df_raw) - len(df_deduplicated)
df_clean = df_raw.copy()
print(f'Dữ liệu chính được giữ nguyên: {len(df_clean):,} dòng.')
print(f'Có {n_dup:,} dòng giống hệt trên 12 cột; chưa loại khỏi mẫu chính vì không có customer_id để xác nhận danh tính.')
deduplicated_file = PROC_DIR / 'hillstrom_deduplicated_sensitivity.csv'
df_deduplicated.to_csv(deduplicated_file, index=False)
print('Đã lưu bộ dữ liệu khử trùng lặp riêng để phân tích độ nhạy:', deduplicated_file)


df_clean['zip_code'] = df_clean['zip_code'].replace({'Surburban': 'Suburban'})
assert set(df_clean['zip_code'].unique()) == {'Rural', 'Suburban', 'Urban'}


assert ((df_clean['conversion'] == 1) == (df_clean['spend'] > 0)).all(), 'Vi phạm: conversion=1 <=> spend>0'
assert (df_clean.loc[df_clean['conversion'] == 1, 'visit'] == 1).all(), 'Vi phạm: conversion=1 => visit=1'
assert df_clean['recency'].between(1, 12).all()
assert ((df_clean['mens'] + df_clean['womens']) >= 1).all()


t41 = pd.DataFrame({
    'Chỉ tiêu': ['Số dòng', 'Số người mua (conversion=1)', 'Tỷ lệ conversion (%)'],
    'Trước': [len(df_raw), int(df_raw['conversion'].sum()), round(df_raw['conversion'].mean() * 100, 3)],
    'Sau':   [len(df_clean), int(df_clean['conversion'].sum()), round(df_clean['conversion'].mean() * 100, 3)],
})
display(t41)
t41.to_csv(FIG_DIR / 'T4_1_cleaning_summary.csv', index=False, encoding='utf-8-sig')

print('\nSố khách mỗi nhóm segment sau làm sạch:')
display(df_clean['segment'].value_counts().to_frame('Số khách'))


df_clean.to_csv(PROC_DIR / 'hillstrom_cleaned.csv', index=False)
print(f"\nĐã lưu: {PROC_DIR / 'hillstrom_cleaned.csv'}")


def engineer_features(df):
    out = df.copy()
    out['log_history'] = np.log1p(out['history'])
    out['promo_match'] = (((out['segment'] == 'Mens E-Mail')   & (out['mens'] == 1)) |
                          ((out['segment'] == 'Womens E-Mail') & (out['womens'] == 1))).astype(int)
    out['both_gender'] = ((out['mens'] == 1) & (out['womens'] == 1)).astype(int)
    out['recency_group'] = pd.cut(out['recency'], bins=[0, 3, 6, 9, 12],
                                  labels=['1-3m', '4-6m', '7-9m', '10-12m']).astype(str)
    assert not (out['recency_group'] == 'nan').any(), 'recency nằm ngoài khoảng 1-12'
    return out

df_feat = engineer_features(df_clean)



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


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df_feat['history'], bins=50, color='tab:blue', alpha=.8)
ax[0].set_title('history (gốc, lệch phải)'); ax[0].set_xlabel('USD'); ax[0].set_ylabel('Số khách')
ax[1].hist(df_feat['log_history'], bins=50, color='tab:green', alpha=.8)
ax[1].set_title('log_history = log(1 + history)'); ax[1].set_xlabel('log(1 + USD)')
fig.tight_layout(); fig.savefig(FIG_DIR / 'H4_2_history_log.png', dpi=300, bbox_inches='tight'); plt.show()



from pathlib import Path
import pandas as pd

p = Path("hillstrom.csv")

print("Đã tìm thấy file:", p.exists())

if p.exists():
    df = pd.read_csv(p)
    print("Kích thước dữ liệu:", df.shape)










from pathlib import Path

hillstrom_path = Path("hillstrom.csv")
if hillstrom_path.exists():
    print(f"Tệp đã tồn tại sẵn tại: {hillstrom_path} | Kích thước: {hillstrom_path.stat().st_size:,} bytes")
else:
    print("Không tìm thấy tệp /content/hillstrom.csv. Vui lòng chạy lại các bước tải dữ liệu ở trên.")



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



import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "hillstrom.csv"

df = df_clean.copy()
print("Kích thước dữ liệu:", df.shape)



FS1  = ['recency', 'log_history', 'mens', 'womens', 'zip_code', 'newbie', 'channel']
FS1P = FS1 + ['segment', 'promo_match']
FS2  = FS1P + ['recency_group', 'both_gender']
FEATURE_SETS = {'FS1': FS1, 'FS1P': FS1P, 'FS2': FS2}


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









train_df, test_df = train_test_split(df_feat, test_size=0.20, random_state=RANDOM_STATE,
                                     stratify=df_feat['conversion'])

df_dedup_feat = engineer_features(df_deduplicated)
train_dedup, test_dedup = train_test_split(
    df_dedup_feat, test_size=0.20, random_state=RANDOM_STATE,
    stratify=df_dedup_feat['conversion']
)
train_dedup[['row_id'] + FS2 + ['conversion']].to_csv(
    PROC_DIR / 'train_deduplicated_sensitivity.csv', index=False
)
test_dedup[['row_id'] + FS2 + ['conversion']].to_csv(
    PROC_DIR / 'test_deduplicated_sensitivity.csv', index=False
)

MODEL_COLS   = ['row_id'] + FS2 + ['conversion']
OUTCOME_FILE = ['row_id', 'segment', 'conversion'] + OUTCOME_COLS

train_df[MODEL_COLS].to_csv(PROC_DIR / 'train.csv', index=False)
test_df[MODEL_COLS].to_csv(PROC_DIR / 'test.csv', index=False)
train_df[OUTCOME_FILE].to_csv(PROC_DIR / 'outcomes_train.csv', index=False)
test_df[OUTCOME_FILE].to_csv(PROC_DIR / 'outcomes_test.csv', index=False)


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








NUM_COLS = ['recency', 'log_history']
CAT_LEVELS = {
    'zip_code':      ['Rural', 'Suburban', 'Urban'],
    'channel':       ['Multichannel', 'Phone', 'Web'],
    'segment':       ['No E-Mail', 'Mens E-Mail', 'Womens E-Mail'],
    'recency_group': ['1-3m', '4-6m', '7-9m', '10-12m'],
}

def make_preprocessor(features, scale=True):
    num = [c for c in features if c in NUM_COLS]
    cat = [c for c in features if c in CAT_LEVELS]
    binary = [c for c in features if c not in num + cat]
    parts = []
    if num:    parts.append(('num', StandardScaler() if scale else 'passthrough', num))
    if binary: parts.append(('bin', 'passthrough', binary))
    if cat:    parts.append(('cat', OneHotEncoder(categories=[CAT_LEVELS[c] for c in cat],
                                                  drop='first', handle_unknown='ignore', sparse_output=False), cat))
    return ColumnTransformer(parts, remainder='drop')

def make_lr(features, class_weight='balanced', C=1.0):
    return Pipeline([('pre', make_preprocessor(features)),
                     ('clf', LogisticRegression(C=C, class_weight=class_weight,
                                                max_iter=2000, random_state=RANDOM_STATE))])


pre_demo = make_preprocessor(FS2).fit(train_df[FS2])
names = [n.split('__', 1)[1] for n in pre_demo.get_feature_names_out()]
demo = pd.DataFrame(pre_demo.transform(train_df[FS2].head(5)), columns=names).round(3)
print(f'FS2 sau biến đổi: {len(names)} cột')
display(demo)
demo.to_csv(FIG_DIR / 'T4_3_onehot_example.csv', index=False, encoding='utf-8-sig')











def lift_at_k(y_true, p, k=0.10):
    y = np.asarray(y_true)
    if np.ptp(p) == 0:            
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


dummy = DummyClassifier(strategy='prior', random_state=RANDOM_STATE).fit(train_df[FS1], y_train)
p_d = dummy.predict_proba(test_df[FS1])[:, 1]
add_row('Dummy (prior)', 'FS1', y_test, p_d, dummy.predict(test_df[FS1]), np.nan)


for fs_name, cols in FEATURE_SETS.items():
    pipe = make_lr(cols)
    thr = choose_threshold_f2(pipe, train_df[cols], y_train, cv)   
    pipe.fit(train_df[cols], y_train)
    prob = pipe.predict_proba(test_df[cols])[:, 1]                 
    add_row('Logistic Regression (balanced)', fs_name, y_test, prob, (prob >= thr).astype(int), thr)
    fitted[fs_name], test_scores[fs_name] = pipe, prob

res = pd.DataFrame(rows)
res.to_csv(FIG_DIR / 'T4_5_baseline_results.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
display(res.round(4))
print(f'PR-AUC mức nền (đoán ngẫu nhiên) = tỷ lệ lớp dương test = {y_test.mean():.4f}')




for fs_name in ['FS1', 'FS2']:
    cols = FEATURE_SETS[fs_name]
    m = make_lr(cols, class_weight=None).fit(train_df[cols], y_train)
    p = m.predict_proba(test_df[cols])[:, 1]
    print(f'{fs_name} mặc định: p_min = {p.min():.5f}, p_max = {p.max():.5f}, số ca dự báo dương tại 0,5 = {(p >= 0.5).sum()}')








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











print('Các tệp đã tạo:')
for p in sorted(list(PROC_DIR.glob('*.csv')) + list(FIG_DIR.glob('*'))):
    print(' -', p)
