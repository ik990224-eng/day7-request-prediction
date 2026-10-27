"""제공 코드: Day5/6 데이터·전처리 재사용 및 출력 보조. 학습된 모델은 포함하지 않습니다.
실습의 목표 함수는 각 main.py의 TODO에서 구현합니다. expected_outputs를 읽지 않습니다.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import math
import numbers
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
NUM = ['input_chars', 'allowed_tools']
CAT = ['task_type', 'priority']
FEATURES = NUM + CAT
FORBIDDEN = {'request_id', 'failed', 'elapsed_ms', 'retry_count', 'status'}
K_CANDIDATES = [1, 3, 5, 9, 15]

@dataclass
class DevData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    yc_train: pd.Series
    yc_valid: pd.Series
    yr_train: pd.Series
    yr_valid: pd.Series
    test_ids: list
    manifest: pd.DataFrame
    source_hashes: dict

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_table(path):
    """빈칸은 NaN. 범주형을 object/np.nan으로 맞춰 버전별 결측 표현 차이를 줄입니다."""
    frame = pd.read_csv(Path(path), encoding='utf-8-sig', dtype={'request_id':str})
    for c in CAT:
        if c in frame:
            s=frame[c].astype(object)
            frame[c]=s.where(s.notna(),np.nan)
    return frame

def check_features(X, *, training=False):
    """입력 계약: 정확한 4개 열, 고유 ID, 유한한 비음수 수치. 결측·미지 범주는 허용."""
    if not isinstance(X,pd.DataFrame):
        raise TypeError('입력 X는 열 이름과 ID가 있는 DataFrame이어야 합니다.')
    missing=[c for c in FEATURES if c not in X]
    extra=[c for c in X if c not in FEATURES]
    if missing or extra:
        raise ValueError(f'입력 열 불일치: 누락={missing}, 추가/금지={extra}')
    if len(X)==0 or X.index.has_duplicates or X.index.isna().any():
        raise ValueError('입력은 비어 있지 않아야 하며 request_id는 고유해야 합니다.')
    result=X.loc[:,FEATURES].copy()
    for c in NUM:
        result[c]=pd.to_numeric(result[c],errors='raise').astype(float)
        v=result[c].dropna().to_numpy()
        if not np.isfinite(v).all() or (v<0).any() or not np.equal(v,np.floor(v)).all():
            raise ValueError(f'{c}: 결측 또는 유한한 비음수 정수형 측정값만 허용합니다.')
    for c in CAT:
        s=result[c].astype(object)
        if any(not isinstance(v,str) or not v.strip() for v in s.dropna()):
            raise ValueError(f'{c}: 비어 있지 않은 문자열 또는 결측이어야 합니다.')
        result[c]=s.where(s.notna(),np.nan)
    if training and result.isna().all().any():
        empty=result.columns[result.isna().all()].tolist()
        raise ValueError(f'train 전체 결측 열: {empty}; 학습 전에 별도 정책을 정하세요.')
    return result

def validate_k(k,n_train):
    if isinstance(k,bool) or not isinstance(k,numbers.Integral) or not 1 <= k <= n_train:
        raise ValueError(f'K는 1~{n_train}의 정수여야 합니다: {k!r}')
    return int(k)

def load_data(data_dir=None):
    """원본·분할표·분할 내 순서를 재사용합니다. test의 X/y는 반환하지 않습니다.
    split_manifest는 Day5의 원본 그대로, split_order는 같은 ID의 원래 순서를 보존합니다.
    회귀와 분류 정답을 같은 ID로 선택합니다. 새 train_test_split을 호출하지 않습니다.
    """
    d=Path(data_dir) if data_dir is not None else DATA
    raw=read_table(d/'requests.csv').set_index('request_id')
    manifest=read_table(d/'split_manifest.csv')
    order=read_table(d/'split_order.csv')
    if raw.index.has_duplicates or raw.index.isna().any():
        raise ValueError('원본 request_id 중복 또는 결측')
    if manifest['request_id'].duplicated().any() or order['request_id'].duplicated().any():
        raise ValueError('분할표 ID 중복: 두 분할에 같은 요청을 넣을 수 없습니다.')
    if set(manifest['request_id']) != set(raw.index) or set(order['request_id']) != set(raw.index):
        raise ValueError('원본·분할표·순서표의 ID 합집합이 다릅니다.')
    if set(manifest['split']) != {'train','valid','test'}:
        raise ValueError('분할 이름은 train/valid/test이어야 합니다.')
    m=manifest.set_index('request_id')['split']
    if not order.set_index('request_id')['split'].sort_index().equals(m.sort_index()):
        raise ValueError('split_order와 split_manifest의 소속이 다릅니다.')
    parts={}
    for name in ['train','valid','test']:
        p=order.loc[order['split']==name].sort_values('position',kind='stable')
        if p.empty or p['position'].tolist() != list(range(len(p))):
            raise ValueError(f'{name}: position은 0부터 시작하는 연속 정수여야 합니다.')
        parts[name]=p['request_id'].tolist()
    # 고정 수업 자료의 입력·레이블 검사도 train/valid에만 적용합니다.
    Xs={}; ycs={}; yrs={}
    for name in ['train','valid']:
        part=raw.loc[parts[name]]
        Xs[name]=check_features(part.loc[:,FEATURES],training=(name=='train'))
        yc=pd.to_numeric(part['failed'],errors='raise')
        yr=pd.to_numeric(part['elapsed_ms'],errors='raise')
        if yc.isna().any() or not yc.isin([0,1]).all():
            raise ValueError(f'{name}: failed는 결측 없는 0/1이어야 합니다.')
        if yr.isna().any() or not np.isfinite(yr).all() or (yr<0).any():
            raise ValueError(f'{name}: elapsed_ms는 결측 없는 유한한 비음수여야 합니다.')
        ycs[name]=yc.astype(int); yrs[name]=yr.astype(float)
        if not (Xs[name].index.equals(ycs[name].index) and Xs[name].index.equals(yrs[name].index)):
            raise ValueError('X와 두 y의 ID/순서 불일치')
    hashes={n:sha256(d/n) for n in ['requests.csv','split_manifest.csv','split_order.csv']}
    return DevData(Xs['train'],Xs['valid'],ycs['train'],ycs['valid'],
                   yrs['train'],yrs['valid'],parts['test'],manifest,hashes)

def load_checkpoint(data_dir=None):
    """앞 실습과 독립적으로 학습할 수 있는 미전처리 train/valid 입력입니다."""
    d=Path(data_dir) if data_dir is not None else DATA
    tr=read_table(d/'checkpoints'/'train.csv').set_index('request_id')
    va=read_table(d/'checkpoints'/'valid.csv').set_index('request_id')
    return (tr[FEATURES].copy(),va[FEATURES].copy(),tr['failed'].copy(),
            va['failed'].copy(),tr['elapsed_ms'].copy(),va['elapsed_ms'].copy())

def make_pre(scale='none'):
    """Day5 복습용 제공 함수: 수치 경로만 스케일링하고 범주는 원핫 처리합니다."""
    steps=[('imputer',SimpleImputer(strategy='median'))]
    if scale=='standard': steps.append(('scaler',StandardScaler()))
    elif scale=='minmax': steps.append(('scaler',MinMaxScaler()))
    elif scale!='none': raise ValueError('scale은 none/standard/minmax 중 하나')
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                  ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('num',Pipeline(steps),NUM),('cat',cat,CAT)],remainder='drop')

def build(estimator,scale='none'):
    """각 후보는 독립적인 전처리기·모델을 갖습니다. clone은 학습 상태를 복사하지 않습니다."""
    return Pipeline([('pre',make_pre(scale)),('model',clone(estimator))])

def reg_metrics(y,pred):
    """이전 실습 재사용용. 회귀 단위는 ms이며 R²는 단위가 없습니다."""
    return {'MAE_ms':float(mean_absolute_error(y,pred)),
            'RMSE_ms':float(root_mean_squared_error(y,pred)),
            'R2':float(r2_score(y,pred))}

def cls_metrics(y,pred):
    """실패(1)를 양성으로 고정. 행=실제, 열=예측, labels=[0,1]."""
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'Accuracy':float(accuracy_score(y,pred)),
            'Precision':float(precision_score(y,pred,zero_division=0)),
            'Recall':float(recall_score(y,pred,zero_division=0)),
            'F1':float(f1_score(y,pred,zero_division=0)),
            'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def snapshot(pre):
    """학습한 중앙값·범주·스케일 통계를 값으로 복사해 변환 전후 대조합니다."""
    num=pre.named_transformers_['num']; cat=pre.named_transformers_['cat']
    values={'numeric_imputer':num['imputer'].statistics_.copy(),
            'categorical_imputer':cat['imputer'].statistics_.copy(),
            'categories':[x.copy() for x in cat['encoder'].categories_]}
    if 'scaler' in num.named_steps:
        for a in ['mean_','scale_','var_','min_','data_min_','data_max_']:
            if hasattr(num['scaler'],a): values[a]=getattr(num['scaler'],a).copy()
    return json_ready(values)

def json_ready(value):
    if isinstance(value,dict): return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,pd.DataFrame): return json_ready(value.to_dict(orient='records'))
    if isinstance(value,(list,tuple,np.ndarray,pd.Index,pd.Series)):
        return [json_ready(v) for v in value]
    if isinstance(value,np.generic): return json_ready(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    if value is pd.NA or value is pd.NaT: return None
    return value

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_ready(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def emit(result,output):
    """계산 결과를 저장합니다. expected_outputs와 교사용 답안을 읽지 않습니다."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/'result.json',result)
    for key,value in result.items():
        if isinstance(value,pd.DataFrame): value.to_csv(output/(key+'.csv'),index=False,encoding='utf-8-sig')
    print(json.dumps(json_ready(result),ensure_ascii=False,indent=2,allow_nan=False))

def environment():
    names=['numpy','pandas','scikit-learn','scipy','joblib','threadpoolctl','xgboost']
    return {'python':platform.python_version(),
            **{x:importlib.metadata.version(x) for x in names}}
