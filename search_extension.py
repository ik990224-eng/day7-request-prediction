"""선택 심화: 기본 비교와 분리된 TreeGrid 실험. 결과는 선택 편향을 포함합니다."""
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from preprocessing import make_preprocessor
from course_support import write_json

def run_search(d,folds,output):
    """4후보×5fold + train60 refit. 반환: fit된 grid. test/valid로 탐색하지 않는다."""
    # TODO D7-07-H1 (구현 완료): train 내부 탐색 후 최적 조건으로 다시 학습하고 별도 결과를 저장합니다.
    # [풀이 설명] d는 개발 자료, folds는 train 내부의 같은 5분할, output은 결과를 쓸 Path입니다.
    # Pipeline의 'pre' 단계는 make_preprocessor('none'), 'model' 단계는 새 DecisionTreeClassifier입니다.
    # random_state=42를 고정하고 param_grid는 model__max_depth=[2, 4], model__min_samples_leaf=[1, 3]만 둡니다.
    # 깊이는 질문의 최대 단계 수, min_samples_leaf는 최종 잎에 남길 최소 학습 표본 수입니다.
    # 이중 밑줄은 Pipeline 안의 'model' 단계로 설정을 전달하는 문법입니다.
    # GridSearchCV(..., scoring='f1', cv=folds, refit=True, n_jobs=1, error_score='raise')로 구성합니다.
    # 네 조합을 다섯 fold로 평가하므로 20번의 fold 학습과 최적 조건의 train 60건 재학습이 생깁니다.
    # grid 변수에 탐색 객체를 저장하고 grid.fit(d.X_train, d.yc_train)으로 학습을 완료합니다.
    # 전처리를 탐색 밖에서 미리 fit하지 않아야 각 fold의 중앙값/범주를 그 fold 학습 부분에서만 배웁니다.
    # 아래 저장 코드가 cv_results_의 평균·표준편차·순위와 best_params_/best_score_를 기록합니다.
    # mean_test_score의 test는 fold 평가 부분을 뜻하며 보관 test 20건과 다릅니다.
    # 반환값은 학습된 grid 객체이고, 출력은 separate_tree_grid.csv/json으로 기본 비교와 분리됩니다.
    # valid/test로 조합을 고르거나 기본 Tree의 설정·선택 결과를 이 탐색 결과로 자동 덮어쓰지 않습니다.
    pipeline = Pipeline([
        ('pre', make_preprocessor('none')),
        ('model', DecisionTreeClassifier(random_state=42)),
    ])
    grid = GridSearchCV(
        pipeline,
        param_grid={
            'model__max_depth': [2, 4],
            'model__min_samples_leaf': [1, 3],
        },
        scoring='f1',
        cv=folds,
        refit=True,
        n_jobs=1,
        error_score='raise',
    )
    grid.fit(d.X_train, d.yc_train)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cols=['param_model__max_depth','param_model__min_samples_leaf',
          'mean_test_score','std_test_score','rank_test_score']
    pd.DataFrame(grid.cv_results_)[cols].to_csv(output/'separate_tree_grid.csv',index=False,encoding='utf-8-sig')
    write_json(output/'separate_tree_grid.json',{'name':'TreeGrid','best_params':grid.best_params_,
        'best_CV_F1':float(grid.best_score_),'basic_table_replaced':False,'test_evaluated':False})
    return grid
