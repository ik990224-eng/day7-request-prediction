"""CV·valid·해석. train 내부 CV와 외부 valid의 역할을 분리합니다."""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from course_support import cls_metrics, reg_metrics, snapshot, json_ready

def run_candidate(name, pipe, d, folds, task='classification'):
    """행 dict, fold DataFrame, 검증예측 DataFrame, train60으로 fit한 Pipeline 반환.
    CV는 clone으로 fold 안에서 fit; valid는 train60으로 별도 fit한 모델로만 예측.
    보관 test의 입력·정답은 받지 않습니다.
    """
    if task not in ('classification', 'regression'):
        raise ValueError('task는 classification 또는 regression이어야 합니다.')
    is_cls=(task=='classification')
    yt=d.yc_train if is_cls else d.yr_train
    yv=d.yc_valid if is_cls else d.yr_valid
    scorer='f1' if is_cls else 'neg_root_mean_squared_error'
    # TODO D7-07-E1 (구현 완료): fold 내부 평가와 train 전체 학습을 분리합니다.
    # [풀이 설명] name은 후보 이름, pipe는 전처리+모델, d는 고정된 개발 자료, folds는 CV 위치 목록입니다.
    # task='classification'이면 실패 여부 yc를 F1으로, 'regression'이면 실행 시간 yr을 RMSE로 평가합니다.
    # 위 yt/yv는 각각 train/valid 정답이며, 입력과 같은 request_id 순서를 유지해야 합니다.
    # cross_validate에 clone(pipe), d.X_train, yt, cv=folds, scoring=scorer를 전달합니다.
    # return_train_score=True는 각 fold의 학습 점수를, return_estimator=True는 학습된 Pipeline을 남깁니다.
    # n_jobs=1로 실행하고 error_score='raise'를 지정하면 학습 오류를 NaN 점수로 숨기지 않습니다.
    # 반환 딕셔너리를 cv에 저장해야 아래 코드가 cv['estimator']로 fold 중앙값을 감사할 수 있습니다.
    # cv['test_score']는 fold evaluate 12건의 점수이며 별도 보관 test 20건의 점수가 아닙니다.
    # 분류에서는 test_score/train_score를 각각 scores/train_scores로 사용합니다.
    # 회귀 scorer는 큰 값이 좋다는 API 규칙 때문에 RMSE의 음수를 반환합니다.
    # 따라서 회귀의 scores와 train_scores 모두 부호를 뒤집어 양의 RMSE_ms로 복원합니다.
    # scores는 다섯 fold 평가값, train_scores는 대응하는 다섯 fold 학습값인 1차원 배열입니다.
    # CV용 학습과 별도로 fitted = clone(pipe).fit(d.X_train, yt)로 train 60건 전체 모델을 만듭니다.
    # pred = fitted.predict(d.X_valid)는 이 전체 학습 모델의 valid 20건 예측이며 재학습은 하지 않습니다.
    # 여기까지 cv, scores, train_scores, fitted, pred를 준비하면 아래 공통 코드가 표와 예측을 만듭니다.
    # row의 CV_mean/CV_std는 fold 점수의 평균/표준편차이며 ddof=0은 5로 나누는 표준편차 설정입니다.
    # 분류 예측은 0/1이고 failure_probability는 classes_에서 클래스 1의 위치를 찾아 추출합니다.
    # 회귀 예측은 ms이며 residual_ms = 실제값 - 예측값이므로 양수이면 실행 시간을 낮게 예측한 것입니다.
    # 반환 순서는 비교 행 dict, fold별 DataFrame, valid 예측 DataFrame, fitted Pipeline입니다.
    # 각 fold의 전처리를 fold fit 48건에서만 학습하고, valid/test로 전처리나 모델을 fit하지 않습니다.
    cv = cross_validate(
        clone(pipe), d.X_train, yt, cv=folds, scoring=scorer,
        return_train_score=True, return_estimator=True,
        n_jobs=1, error_score='raise',
    )
    scores = cv['test_score']
    train_scores = cv['train_score']
    if not is_cls:
        scores = -scores
        train_scores = -train_scores
    fitted = clone(pipe).fit(d.X_train, yt)
    pred = fitted.predict(d.X_valid)
    metric='F1' if is_cls else 'RMSE_ms'
    valid=cls_metrics(yv,pred) if is_cls else reg_metrics(yv,pred)
    train=cls_metrics(yt,fitted.predict(d.X_train)) if is_cls else reg_metrics(yt,fitted.predict(d.X_train))
    row={'model':name,'task':task,'scaling':'none' if name in ['Tree','Forest','XGB'] else 'standard',
         'CV_mean':float(scores.mean()),'CV_std':float(scores.std(ddof=0)),
         'CV_metric':metric,'train_primary':train[metric],
         **{'valid_'+k:v for k,v in valid.items()},
         'params':json.dumps(json_ready(fitted['model'].get_params()),ensure_ascii=False,sort_keys=True)}
    fold_rows=[]
    for i,((a,b),score,tr,fold_model) in enumerate(zip(folds,scores,train_scores,cv['estimator']),1):
        expected=d.X_train.iloc[a][['input_chars','allowed_tools']].median().to_numpy()
        actual=fold_model['pre'].named_transformers_['num']['imputer'].statistics_
        # 이 검사는 모든 단계에 제공합니다. 06의 통계 감사에서도 같은 원리를 구현합니다.
        if not np.allclose(expected,actual): raise AssertionError('fold 학습 범위 밖의 중앙값을 사용했습니다.')
        fold_rows.append({'task':task,'model':name,'fold':i,'fit_n':len(a),'eval_n':len(b),
            'metric':metric,'score':float(score),'train_score':float(tr),'median_audit_passed':True})
    predictions=pd.DataFrame({'request_id':d.X_valid.index,'task':task,'model':name,
                              'actual':yv.to_numpy(),'predicted':pred})
    if is_cls:
        pindex=list(fitted.classes_).index(1)
        predictions['failure_probability']=fitted.predict_proba(d.X_valid)[:,pindex]
    else: predictions['residual_ms']=yv.to_numpy()-pred
    return row,pd.DataFrame(fold_rows),predictions,fitted

def importance_table(forest, d, include_permutation=False):
    """train MDI 8행. 응용이면 valid 원본 4열 순열 진단 4행 추가. 인과/특성선택 아님."""
    # TODO D7-07-E2 (구현 완료): 학습 중요도와 검증 순열 진단을 구분합니다.
    # [풀이 설명] forest는 train 60건으로 이미 학습된 분류 RandomForest Pipeline입니다.
    # forest['pre'].get_feature_names_out()은 수치/원핫 변환 뒤의 실제 입력 열 이름을 반환합니다.
    # 현재 자료에서는 num__로 시작하는 수치 2열과 cat__로 시작하는 원핫 6열, 총 8열입니다.
    # 이 이름 순서와 forest['model'].feature_importances_ 순서를 짝지어 MDI 행을 만듭니다.
    # MDI는 학습 트리의 불순도 감소에 기반하므로 evaluation_data='train60', metric='impurity_decrease'입니다.
    # 이 행의 method는 'MDI', importance는 계산된 값, repeat_std는 반복값이 없으므로 NaN으로 둡니다.
    # include_permutation=False이면 여기까지 8행만 반환합니다. True이면 별도 진단 4행을 추가합니다.
    # 순열 진단에는 forest 전체와 d.X_valid, d.yc_valid, scoring='f1'을 전달합니다.
    # 그러면 원본 4열을 하나씩 섞은 뒤 Pipeline이 변환하므로 범주 열도 원래 열 단위로 평가됩니다.
    # 같은 실험을 재현하도록 random_state=42와 n_jobs=1을 두고 n_repeats=10을 실험 조건으로 고정합니다.
    # 원본 열마다 10번 섞어 F1 감소량의 평균과 표준편차를 구하며, 기준 결과의 수치는 계산에 사용하지 않습니다.
    # 순열 행의 feature는 d.X_valid.columns의 순서, importance는 importances_mean입니다.
    # repeat_std에는 importances_std, method에는 'permutation'을 기록합니다.
    # 순열 행은 evaluation_data='valid20 (diagnostic)', metric='F1_drop'으로 계산 자료와 지표를 구분합니다.
    # 반환 열은 feature, method, importance, repeat_std, evaluation_data, metric인 DataFrame입니다.
    # 중요도 평균이 음수여도 0으로 잘라내지 않으며 repeat_std를 신뢰구간으로 해석하지 않습니다.
    # 중요도는 인과관계가 아니고, 이 진단을 근거로 특성을 자동 삭제하거나 숲을 다시 학습하지 않습니다.
    mdi = pd.DataFrame({
        'feature': forest['pre'].get_feature_names_out(),
        'method': 'MDI',
        'importance': forest['model'].feature_importances_,
        'repeat_std': np.nan,
        'evaluation_data': 'train60',
        'metric': 'impurity_decrease',
    })
    if not include_permutation:
        return mdi

    result = permutation_importance(
        forest, d.X_valid, d.yc_valid, scoring='f1',
        n_repeats=10, random_state=42, n_jobs=1,
    )
    permutation = pd.DataFrame({
        'feature': d.X_valid.columns,
        'method': 'permutation',
        'importance': result.importances_mean,
        'repeat_std': result.importances_std,
        'evaluation_data': 'valid20 (diagnostic)',
        'metric': 'F1_drop',
    })
    return pd.concat([mdi, permutation], ignore_index=True)

def choose_by_cv(table, task='classification'):
    """F1 평균 최대 / RMSE 평균 최소. 정확히 동점이면 사전 후보 순서 우선.
    valid 점수는 함수에서 참조하지 않습니다. 선택점수는 최종 성능이 아닙니다.
    """
    # TODO D7-07-E3 (구현 완료): valid를 참조하지 않고 CV 평균으로 선택합니다.
    # [풀이 설명] table은 사전에 선언한 후보 순서로 쌓은 비교표이며 model과 CV_mean 열을 사용합니다.
    # 분류의 CV_mean은 다섯 fold의 F1 평균이므로 가장 큰 값을 가진 모델을 고릅니다.
    # 회귀의 CV_mean은 양수로 복원한 RMSE 평균이므로 가장 작은 값을 가진 모델을 고릅니다.
    # np.argmax()/np.argmin()은 같은 최적값이 여러 개면 처음 나온 행 위치를 반환하므로 선언 순서를 유지합니다.
    # iloc로 해당 위치의 model 값을 읽고 모델 이름 문자열을 반환합니다. 행 전체나 Pipeline이 아닙니다.
    # 정확한 동점만 순서로 해결하므로 반올림하거나 np.isclose로 비슷한 점수까지 동점 처리하지 않습니다.
    # model 이름이나 valid 점수로 표를 정렬하면 동점 규칙이 달라지므로 후보 순서를 먼저 보존합니다.
    # valid_F1/valid_RMSE_ms는 진단용으로 남기고 이 함수의 선택 기준에는 사용하지 않습니다.
    # 여러 후보 중 최대/최소를 골랐으므로 선택된 CV 점수를 독립적인 최종 성능으로 주장하지 않습니다.
    if task not in ('classification', 'regression'):
        raise ValueError('task는 classification 또는 regression이어야 합니다.')
    if table.empty:
        raise ValueError('선택할 후보가 없습니다.')
    scores = table['CV_mean'].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('CV_mean은 결측 없는 유한한 점수여야 합니다.')
    position = int(np.argmax(scores) if task == 'classification' else np.argmin(scores))
    return table.iloc[position]['model']
