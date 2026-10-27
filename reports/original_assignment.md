# 실습 07 · 앙상블 비교 프로젝트 통합 · 심화

PPT 68~75 / E14~18 | 교시 실습 25분 | 기본필수 후 선택 확장

## 목표
분류 비교·CV·중요도·예측·보고서를 동일 실행에서 생성한다.

## 구현 위치와 계약
### `evaluate.py` → `run_candidate` · D7-07-E1
행 dict, fold DataFrame, 검증예측 DataFrame, train60으로 fit한 Pipeline 반환.
CV는 clone으로 fold 안에서 fit; valid는 train60으로 별도 fit한 모델로만 예측.
보관 test의 입력·정답은 받지 않습니다.

### `evaluate.py` → `importance_table` · D7-07-E2
train MDI 8행. 응용이면 valid 원본 4열 순열 진단 4행 추가. 인과/특성선택 아님.

### `evaluate.py` → `choose_by_cv` · D7-07-E3
F1 평균 최대 / RMSE 평균 최소. 정확히 동점이면 사전 후보 순서 우선.
valid 점수는 함수에서 참조하지 않습니다. 선택점수는 최종 성능이 아닙니다.

### `main.py` → `run_project` · D7-07-M1


### `preprocessing.py` → `make_preprocessor` · D7-07-P1
num: median (+standard 선택), cat: 최빈값 -> dense OneHot(ignore).

### `preprocessing.py` → `candidates` · D7-07-P2
기본 분류 Dummy/Tree/Forest/XGB. 응용에 Logistic/KNN3, 회귀는 6후보.
Dummy와 전날 모델만 scale=standard. 각 후보는 서로 다른 새 객체여야 합니다.

### `search_extension.py` → `run_search` · D7-07-H1
4후보×5fold + train60 refit. 반환: fit된 grid. test/valid로 탐색하지 않는다.

## 실행
역할 폴더 루트에서 실행하거나 이 폴더로 이동해 `python main.py`를 실행합니다.
```powershell
python "ex07_project/03_심화/day7_ensemble/main.py"
```

## 입력과 파일 위치
이 폴더의 data/를 사용합니다. 프로젝트 전체를 복사하면 단독 실행할 수 있습니다.
앞 단원 완료 파일을 필수 입력으로 요구하지 않습니다. 공통 로더/전처리는 Day5~6 재사용 코드로 제공합니다.

## 결과와 완료 기준
실제 계산 결과는 `outputs/`, 기준 출력은 `expected_output.txt`와 `expected_outputs/`에 있습니다.
`classification_comparison.csv`, `cv_scores.csv`, `feature_importance.csv`, `fold_manifest.csv`, `fresh_process_restore.json`, `persistence_check.json`, `regression_comparison.csv`, `run_summary.json`, `separate_tree_grid.csv`, `separate_tree_grid.json`, `validation_predictions.csv`
숫자는 결과 파일에서 계산해야 합니다. 기준 파일을 읽어 정답을 반환하거나 숫자를 하드코딩하지 않습니다.
수치 허용오차는 일반적으로0.001(ms 포함), 비율은 상대오차1e-6도 함께 사용합니다. 다른 라이브러리/운영체제에서 순위가 다르면 설정과 버전을 먼저 기록합니다.

## 구현 힌트
실습06에서 완성한 CV 호출을 evaluate.run_candidate에 재사용합니다. 이름과 열 스펠링을 유지합니다. 기본필수의 6개 TODO는 전처리·후보구성·CV학습·중요도·선택·흐름연결입니다. 이미 작성한 함수를 옮겨 쓰는 시간이지 25분 만에 처음부터 구현하는 시간이 아닙니다.

## 3단계 선택 원칙
8개 실습마다 기본필수1개를 먼저 수행합니다. 응용·심화를 모두 당일 완료할 필요는 없습니다.
실습07~08은 같은 프로젝트를 이어 씁니다. 프로젝트를 두 번 새로 만들지 않습니다.

## 학습기록
`학습기록.md`에 실제 결과와 판단 근거를 적습니다. 자동검사는 계산을 확인할 뿐 해석을 대신하지 않습니다.

`NotImplementedError`는 미완성 TODO의 표시입니다. 함수를 완성한 후 다시 실행하세요.
