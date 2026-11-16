from __future__ import annotations
import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..paths import DATA_CSV as DATA_PATH, MODEL_FILE as MODEL_PATH, MODEL_META as META_PATH
# Thứ tự 11 biến FS2 đúng như common.py của pipeline (FS1P + recency_group + both_gender).
FS2 = ['recency', 'log_history', 'mens', 'womens', 'zip_code', 'newbie', 'channel',
       'segment', 'promo_match', 'recency_group', 'both_gender']
RECENCY_LABELS = ['1-3m', '4-6m', '7-9m', '10-12m']


def _model_name():
    try:
        meta = json.loads(META_PATH.read_text(encoding='utf-8'))
        return f"{meta.get('model', 'Mô hình cuối')} (FS2, {meta.get('strategy', '')})"
    except Exception:
        return 'Mô hình cuối (FS2)'


MODEL_NAME = _model_name()
SEGMENTS = ('Mens E-Mail', 'Womens E-Mail')

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Thiếu mô hình: {MODEL_PATH}. Chép final_model_FS2.joblib từ pipeline vào thư mục models/.')
    # Only load the trusted model exported by the user's own training workflow.
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, 'predict_proba'):
        raise TypeError('Mô hình cuối không hỗ trợ predict_proba.')
    expected = list(getattr(model, 'feature_names_in_', []))
    if expected and set(expected) != set(FS2):
        raise ValueError(f'11 biến FS2 không khớp mô hình: {expected}')
    classes = list(getattr(model, 'classes_', []))
    if classes and classes != [0, 1]:
        raise ValueError(f'Nhãn mô hình phải là [0, 1], nhận được {classes}')
    return model


def make_features(form: dict, segment: str) -> pd.DataFrame:
    if segment not in SEGMENTS:
        raise ValueError('Loại email không hợp lệ.')
    try:
        recency = int(form.get('recency', 6))
        history = float(form.get('history', 242.09))
        mens = int(form.get('mens', 1))
        womens = int(form.get('womens', 0))
        newbie = int(form.get('newbie', 0))
    except (ValueError, TypeError) as exc:
        raise ValueError('Dữ liệu khách hàng phải là số hợp lệ.') from exc
    channel = str(form.get('channel', 'Multichannel'))
    zip_code = str(form.get('zip_code', 'Urban'))
    if not 1 <= recency <= 12 or not math.isfinite(history) or not 0 <= history <= 100000:
        raise ValueError('Recency phải từ 1–12 và History từ 0–100.000 USD.')
    if channel not in ('Multichannel', 'Web', 'Phone') or zip_code not in ('Urban', 'Suburban', 'Rural'):
        raise ValueError('Kênh mua hàng hoặc khu vực không hợp lệ.')
    if any(v not in (0, 1) for v in (mens, womens, newbie)):
        raise ValueError('Các biến Mens, Womens, Newbie chỉ nhận 0 hoặc 1.')
    if not (mens or womens):
        raise ValueError('Khách hàng phải từng mua hàng nam hoặc hàng nữ. Chọn ít nhất một mục Có.')
    group = '1-3m' if recency <= 3 else '4-6m' if recency <= 6 else '7-9m' if recency <= 9 else '10-12m'
    record = dict(recency=recency, log_history=math.log1p(history), mens=mens, womens=womens,
                  zip_code=zip_code, newbie=newbie, channel=channel, segment=segment,
                  promo_match=int((segment == SEGMENTS[0] and mens == 1) or
                                  (segment == SEGMENTS[1] and womens == 1)),
                  both_gender=int(mens == 1 and womens == 1), recency_group=group)
    return pd.DataFrame([record], columns=FS2)


def predict_customer(form: dict) -> dict:
    model = get_model()
    pm = float(model.predict_proba(make_features(form, SEGMENTS[0]))[0, 1])
    pw = float(model.predict_proba(make_features(form, SEGMENTS[1]))[0, 1])
    recommendation = ('Hai phương án có xác suất tương đương' if abs(pm-pw) < 1e-9
                      else 'Ưu tiên Mens E-Mail' if pm > pw else 'Ưu tiên Womens E-Mail')
    return {'mens': pm, 'womens': pw, 'recommendation': recommendation,
            'difference_pp': abs(pm-pw)*100, 'model': MODEL_NAME}


@lru_cache(maxsize=1)
def simulation_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f'Thiếu dữ liệu: {DATA_PATH}. Chép hillstrom.csv vào thư mục data/.')
    df = pd.read_csv(DATA_PATH)
    cols = ['recency', 'history_segment', 'history', 'mens', 'womens', 'zip_code',
            'newbie', 'channel', 'segment', 'visit', 'conversion', 'spend']
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f'CSV thiếu các cột: {sorted(missing)}')
    df = df[cols].copy()
    if df[cols].isna().any().any():
        raise ValueError('CSV có dữ liệu thiếu.')
    if not df['recency'].between(1, 12).all() or (df['history'] < 0).any():
        raise ValueError('Recency/history không hợp lệ trong CSV.')
    if not ((df['mens'] + df['womens']) >= 1).all():
        raise ValueError('Có khách hàng không mua cả hàng nam và nữ.')
    # Preserve all rows: no unique customer ID to justify deduplication.
    df['zip_code'] = df['zip_code'].replace({'Surburban': 'Suburban'})
    df['log_history'] = np.log1p(df['history'])
    df['both_gender'] = ((df['mens'] == 1) & (df['womens'] == 1)).astype(int)
    df['recency_group'] = pd.cut(df['recency'], bins=[0,3,6,9,12],
                                 labels=RECENCY_LABELS).astype(str)
    return df


@lru_cache(maxsize=1)
def campaign_predictions():
    df = simulation_data()
    model = get_model()
    base_features = [c for c in FS2 if c not in ('segment','promo_match')]
    x = df[base_features].copy()
    probabilities = []
    for segment in SEGMENTS:
        x['segment'] = segment
        x['promo_match'] = (df['mens'] if segment == SEGMENTS[0] else df['womens']).astype(int)
        p = np.asarray(model.predict_proba(x[FS2])[:, 1], dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Mô hình trả về xác suất không hợp lệ.')
        probabilities.append(p)
    best = np.maximum(*probabilities)
    order = np.argsort(-best, kind='stable')
    return {'sorted_best': best[order], 'n': len(best),
            'mens_count': int((probabilities[0] >= probabilities[1]).sum())}


def simulate(form: dict) -> dict:
    try:
        top_k = int(form.get('top_k', 10))
        cost_per_email = float(form.get('cost', 0.05))
        aov = float(form.get('aov', 116.36))
        margin = float(form.get('margin', 20))
    except (TypeError, ValueError) as exc:
        raise ValueError('Thông số mô phỏng không hợp lệ.') from exc
    if not (1 <= top_k <= 100 and 0 <= cost_per_email <= 10000 and
            0 <= aov <= 100000 and 0 <= margin <= 100 and
            all(math.isfinite(v) for v in (cost_per_email, aov, margin))):
        raise ValueError('Thông số Top-K, chi phí, giá trị đơn hoặc biên lợi nhuận ngoài phạm vi.')
    result = campaign_predictions()
    n = result['n']
    if n == 0:
        raise ValueError('Không có hồ sơ để mô phỏng.')
    count = max(1, math.ceil(n * top_k / 100))
    expected_orders = float(result['sorted_best'][:count].sum())
    total_cost = cost_per_email * count
    expected_profit = expected_orders * aov * margin / 100 - total_cost
    return {'n': n, 'top_k': top_k, 'selected': count,
            'expected_orders': expected_orders, 'email_cost': total_cost,
            'expected_profit': expected_profit, 'model': MODEL_NAME}
