"""Data/EDA logic for the Hillstrom dashboard. Independent of PyQt6 for testing."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

SEGMENTS = ['Mens E-Mail', 'Womens E-Mail', 'No E-Mail']
TREATED = ['Mens E-Mail', 'Womens E-Mail']
BASELINE = pd.DataFrame([
    {'segment':'Mens E-Mail', 'n':21307, 'visit_pct':18.28, 'conversion_pct':1.25, 'avg_spend':1.42},
    {'segment':'Womens E-Mail', 'n':21387, 'visit_pct':15.14, 'conversion_pct':0.88, 'avg_spend':1.08},
    {'segment':'No E-Mail', 'n':21306, 'visit_pct':10.62, 'conversion_pct':0.57, 'avg_spend':0.65},
])
LABELS = {'Mens E-Mail':'Email nam', 'Womens E-Mail':'Email nữ', 'No E-Mail':'Đối chứng'}
FILTERS = {
    'Tất cả khách hàng': None,
    'Hai nhóm nhận email': TREATED,
    'Mens E-Mail': ['Mens E-Mail'],
    'Womens E-Mail': ['Womens E-Mail'],
    'No E-Mail': ['No E-Mail'],
}
EXPECTED = ['recency', 'history_segment', 'history', 'mens', 'womens',
            'zip_code', 'newbie', 'channel', 'segment', 'visit', 'conversion', 'spend']

@dataclass
class Overview:
    total: int
    emailed: int
    control: int
    email_buys: int | None
    email_rate: float
    source: str


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy CSV: {path}')
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='cp1252', low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    must = {'segment', 'conversion', 'visit'}
    missing = must.difference(df.columns)
    if missing:
        raise ValueError('Thiếu cột bắt buộc: ' + ', '.join(sorted(missing)))
    df['segment'] = df['segment'].astype('string').str.strip()
    df['segment'] = df['segment'].replace({'Mens E-mail':'Mens E-Mail', 'Womens E-mail':'Womens E-Mail',
                                            'No E-mail':'No E-Mail'})
    for c in ('conversion','visit','spend','recency','history'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype('string').str.strip().replace({'Surburban':'Suburban'})
    if 'channel' in df.columns:
        df['channel'] = df['channel'].astype('string').str.strip()
    return df


def filtered(df: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    allowed = FILTERS.get(filter_name)
    return df if allowed is None else df.loc[df['segment'].isin(allowed)]


def overview(df: pd.DataFrame | None) -> Overview:
    if df is None:
        # Grounded summary: 64,000 total; 42,694 treated; 456 treated converters.
        return Overview(64000,42694,21306,456,456 / 42694 * 100,'Số liệu tổng hợp từ Structure(1).pdf')
    treated = df['segment'].isin(TREATED)
    controls = df['segment'].eq('No E-Mail')
    y = df.loc[treated, 'conversion']
    buyers = int((y == 1).sum())
    rate = float(y.mean() * 100) if y.notna().any() else 0.0
    return Overview(len(df), int(treated.sum()), int(controls.sum()), buyers, rate,'Đang tính trực tiếp từ CSV')


def group_stats(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return BASELINE.copy()
    rows=[]
    for segment in SEGMENTS:
        d = df.loc[df['segment'].eq(segment)]
        rows.append(dict(segment=segment, n=len(d), visit_pct=float(d['visit'].mean()*100) if len(d) else 0,
                         conversion_pct=float(d['conversion'].mean()*100) if len(d) else 0,
                         avg_spend=float(d['spend'].mean()) if len(d) and 'spend' in d else 0))
    return pd.DataFrame(rows)


def conversion_by(df: pd.DataFrame, col: str, order=None, max_items=10):
    if col not in df:
        return [], [], []
    data = df[[col,'conversion']].dropna(subset=[col, 'conversion'])
    if data.empty:
        return [], [], []
    result = data.groupby(col, observed=True)['conversion'].agg(['mean','size'])
    if order is not None:
        result = result.reindex(order).dropna(subset=['mean'])
    else:
        result = result.sort_values('size', ascending=False).head(max_items)
    return list(map(str,result.index)), (result['mean']*100).tolist(), result['size'].astype(int).tolist()


def data_quality(df: pd.DataFrame):
    rows=[]
    for c in df.columns:
        miss = int(df[c].isna().sum())
        rows.append((c, str(df[c].dtype), len(df), miss, 100*miss/max(1,len(df)), int(df[c].nunique(dropna=True))))
    return rows, int(df.duplicated().sum())
