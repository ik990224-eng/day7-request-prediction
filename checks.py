"""제공 코드: 입력·분할 계약 및 실행 기록. test의 X/y를 만들지 않습니다."""
import numpy as np
from course_support import FEATURES, NUM, check_features, snapshot

def verify_development_data(d):
    if len(d.X_train)!=60 or len(d.X_valid)!=20 or len(d.test_ids)!=20:
        raise ValueError('공통 분할은 60/20/20이어야 합니다.')
    tr=set(d.X_train.index); va=set(d.X_valid.index); te=set(d.test_ids)
    if tr&va or tr&te or va&te or len(tr|va|te)!=100:
        raise ValueError('분할 ID 중복/누락')
    for X,yc,yr in [(d.X_train,d.yc_train,d.yr_train),(d.X_valid,d.yc_valid,d.yr_valid)]:
        if not X.index.equals(yc.index) or not X.index.equals(yr.index): raise ValueError('X/y 정렬 오류')
        check_features(X)
    return {'train_n':60,'valid_n':20,'heldout_test_id_n':20,'disjoint':True}

def boundary_checks(fitted, d):
    """응용·심화 제공 안전 검사: 미지 범주와 결측을 허용, 필수 열 누락은 차단."""
    new=d.X_valid.iloc[:2].copy();new.index=['NEW_A','NEW_B']
    new['task_type']=['translation','translation'];new['priority']=['urgent','urgent']
    new.loc['NEW_A','input_chars']=np.nan
    before=snapshot(fitted['pre'])
    z=fitted['pre'].transform(check_features(new))
    fitted.predict(check_features(new))
    if z.shape!=(2,8) or not np.isfinite(z).all(): raise AssertionError('새 범주/결측 변환 실패')
    if before!=snapshot(fitted['pre']): raise AssertionError('통계 변경')
    try: check_features(new.drop(columns=['priority']))
    except ValueError: blocked=True
    else: blocked=False
    if not blocked: raise AssertionError('필수 열 누락 미차단')
    return {'unknown_category_shape':[2,8],'statistics_unchanged':True,'missing_column_blocked':True}
