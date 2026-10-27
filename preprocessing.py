"""특성 선택·전처리·모델 생성. 아직 학습하지 않은 Pipeline을 만듭니다."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from course_support import NUM, CAT, FEATURES
from model_config import new_classifier, new_regressor

def make_preprocessor(scale='none'):
    """num: median (+standard 선택), cat: 최빈값 -> dense OneHot(ignore)."""
    # TODO D7-07-P1 (구현 완료): 수치·범주 전처리를 연결한 새 객체를 반환합니다.
    # [풀이 설명] 학습은 아직 하지 않고, 원본 4개 열을 수치 배열로 바꿀 전처리 객체를 반환합니다.
    # NUM=['input_chars', 'allowed_tools']는 글자 수와 도구 수, CAT=['task_type', 'priority']는 범주 열입니다.
    # 수치 경로는 ('imputer', SimpleImputer(strategy='median'))으로 결측값을 열별 중앙값으로 채웁니다.
    # scale='standard'일 때만 그 뒤에 ('scaler', StandardScaler())를 연결하여 수치 단위를 맞춥니다.
    # scale='none'이면 scaler를 'passthrough'로 두어 표준화하지 않습니다. 다른 값은 ValueError로 거절합니다.
    # 중앙값과 표준화의 평균·분산은 이후 Pipeline.fit을 호출할 때 전달된 학습 부분에서 계산됩니다.
    # 범주 경로는 SimpleImputer(strategy='most_frequent')로 빈칸을 가장 흔한 값으로 채웁니다.
    # 다음은 OneHotEncoder(handle_unknown='ignore', sparse_output=False)로 범주별 0/1 열을 만듭니다.
    # ignore는 학습에 없던 범주를 만나도 해당 범주의 출력들을 0으로 두며 열을 새로 늘리지 않습니다.
    # sparse_output=False는 희소 행렬 대신 일반적인 밀집 배열을 반환하도록 하는 설정입니다.
    # 범주 Pipeline의 단계 이름은 'imputer', 'encoder', 수치 Pipeline의 단계 이름은 'imputer', 'scaler'입니다.
    # ColumnTransformer에 ('num', 수치 Pipeline, NUM), ('cat', 범주 Pipeline, CAT)를 이 순서로 넣습니다.
    # remainder='drop'은 지정하지 않은 열을 변환 결과에 포함하지 않는다는 뜻입니다.
    # 단계 이름은 중앙값 감사와 중요도 열 이름 생성에 사용되므로 'num'/'cat'/'imputer'를 유지합니다.
    # 반환값은 미학습 ColumnTransformer이고, 호출할 때마다 새 객체를 만들어 후보 간 학습 상태를 분리합니다.
    # 전처리 생성 과정에서 전체 데이터나 valid를 읽어 fit하지 않습니다. CV에서는 fold fit 부분에서만 배웁니다.
    if scale not in ('none', 'standard'):
        raise ValueError("scale은 'none' 또는 'standard'여야 합니다.")
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler() if scale == 'standard' else 'passthrough'),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric, NUM),
        ('cat', categorical, CAT),
    ], remainder='drop')

def candidates(include_previous=False, task='classification'):
    """기본 분류 Dummy/Tree/Forest/XGB. 응용에 Logistic/KNN3, 회귀는 6후보.
    Dummy와 전날 모델만 scale=standard. 각 후보는 서로 다른 새 객체여야 합니다.
    """
    # TODO D7-07-P2 (구현 완료): 선언 순서와 스케일링 계약에 맞게 독립적인 후보들을 만듭니다.
    # [풀이 설명] 반환값은 모델 이름을 키, 미학습 Pipeline을 값으로 갖는 순서가 유지되는 사전입니다.
    # 기본 분류 후보 순서는 Dummy, Tree, Forest, XGB입니다. 이름과 대소문자를 그대로 사용합니다.
    # 분류에서 include_previous=True이면 Dummy, Logistic, KNN3, Tree, Forest, XGB 순서로 구성합니다.
    # 회귀 후보는 Dummy, Linear, KNN15, Tree, Forest, XGB의 6개이며 분류의 KNN3와 구분합니다.
    # 분류는 new_classifier(name), 회귀는 new_regressor(name)로 각 후보의 새 모델을 받습니다.
    # Dummy 분류는 최빈 클래스를, Dummy 회귀는 학습 정답 평균을 예측하는 기준선입니다.
    # Logistic은 C=1, max_iter=1000이며, KNN3/KNN15의 숫자는 참고할 최근접 이웃 수입니다.
    # Tree는 깊이 3/잎 최소 3, Forest는 트리 100개/깊이 4/잎 최소 2라는 공통 설정을 사용합니다.
    # Forest의 max_features는 분류에서 'sqrt', 회귀에서 1.0이므로 모델 종류별 설정을 유지합니다.
    # XGB는 80회차/깊이 2/학습률 0.05이며 분류와 회귀의 목적함수는 model_config가 구분합니다.
    # Dummy와 이전 모델(Logistic/KNN3 또는 Linear/KNN15)에는 scale='standard'를 사용합니다.
    # Tree/Forest/XGB에는 scale='none'을 사용하며 범주 열에는 어느 경우에도 StandardScaler를 적용하지 않습니다.
    # 각 Pipeline은 [('pre', make_preprocessor(scale)), ('model', 새 모델)] 구조로 만듭니다.
    # pre/model 이름은 평가 코드와 model__max_depth 같은 탐색 설정이 참조하므로 그대로 유지합니다.
    # make_preprocessor를 후보마다 새로 호출해야 한 모델의 fit이 다른 모델의 전처리 상태를 바꾸지 않습니다.
    # 이 함수에서는 fit하지 않습니다. run_candidate가 CV용 복제와 train 전체용 복제를 각각 학습합니다.
    # 후보의 삽입 순서는 CV 평균이 정확히 같을 때의 우선순위이므로 임의 정렬하지 않습니다.
    if task == 'classification':
        names = ['Dummy', 'Logistic', 'KNN3', 'Tree', 'Forest', 'XGB'] if include_previous else [
            'Dummy', 'Tree', 'Forest', 'XGB',
        ]
        new_model = new_classifier
    elif task == 'regression':
        names = ['Dummy', 'Linear', 'KNN15', 'Tree', 'Forest', 'XGB']
        new_model = new_regressor
    else:
        raise ValueError("task는 'classification' 또는 'regression'이어야 합니다.")

    models = {}
    for name in names:
        scale = 'none' if name in ('Tree', 'Forest', 'XGB') else 'standard'
        models[name] = Pipeline([
            ('pre', make_preprocessor(scale)),
            ('model', new_model(name)),
        ])
    return models
