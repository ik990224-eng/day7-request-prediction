# 정상·경계·실패 테스트 기록

검증일: **2026-10-27** / 환경: **Windows, Python 3.13.16** / 범위: `main.py --level 3`의 분류 6후보·회귀 6후보와 별도 TreeGrid.

실제 실행 로그와 CSV·JSON을 대조해 작성한 기록이다. 단위 테스트 **17개가 모두 통과**했고, 출력 감사의 상태는 `passed`였다. 저장 모델의 새 프로세스 복원과 새 요청 2건의 예측도 정상 종료했다. 패키지 버전은 [run_summary.json](../outputs/run_summary.json)의 `versions`에 남겼다.

## 1. 검증 결과와 근거

아래 기대값은 수업의 데이터·함수 계약이다. 사용자가 실행 전에 직접 작성한 예상이나 이해도 평가로 표시하지 않는다.

| 상황 | 계약상의 기대 | 실제 확인 | 증거 | 판정 |
|---|---|---|---|---|
| 단위 테스트 | 입력·분할·선택·전처리·회귀 평가 계약을 만족 | `Ran 17 tests`, `OK`; 테스트별 결과 모두 `ok` | [test_run.txt](test_run.txt) | 통과 |
| valid 예측 | 각 task·모델에 valid 20건, ID·정답·순서 일치 | 12개 task·모델 조합에 각 20행, 합계 240행. 저장 예측으로 분류 F1·FP·FN과 회귀 MAE·RMSE를 재계산해 비교표와 대조 | [output_audit.json](../outputs/output_audit.json), [validation_predictions.csv](../outputs/validation_predictions.csv) | 통과 |
| CV 점수와 통계 | 후보마다 5개 fold, 각 학습 48건·평가 12건, 평균·표준편차 일치 | 총 60행의 fold 점수와 비교표의 CV 평균·`ddof=0` 표준편차가 일치 | [output_audit.json](../outputs/output_audit.json), [cv_scores.csv](../outputs/cv_scores.csv) | 통과 |
| 각 fold 중앙값 | 해당 fold 학습 48건에서만 중앙값 계산 | 분류·회귀 60개 후보·fold 기록의 `median_audit_passed=True`. 학습된 대체 중앙값을 해당 fit 입력의 중앙값과 대조 | [cv_scores.csv](../outputs/cv_scores.csv), [evaluate.py](../evaluate.py)의 `run_candidate` | 통과 |
| fold ID 소속 | fit/evaluate 중복 없음, train ID만 사용 | task별 5개 fold가 선언한 분할과 일치. 각 train ID는 평가에 1번·학습에 4번 등장 | [output_audit.json](../outputs/output_audit.json), [fold_manifest.csv](../outputs/fold_manifest.csv) | 통과 |
| 회귀 scorer와 재학습 | 음수 scorer를 양수 RMSE로 복원하고 최종 모델은 train 60건으로 학습 | 학습·평가 RMSE가 유한한 비음수. 첫 회귀 fold를 별도 학습해 RMSE 정의로 직접 계산한 값과 일치. 반환 모델의 학습 표본 수 60, 원본 후보는 미학습 상태 유지 | [test_run.txt](test_run.txt), [test_project.py](../test_project.py)의 `test_regression_candidate_cv_matches_independent_fold` | 통과 |
| CV 선택 규칙 | 분류 최대·회귀 최소, 정확한 동점만 선언 순서, valid 무시 | 재계산 선택은 XGB/KNN15로 실행 요약과 일치. 분류·회귀 정확한 동점, 아주 가까우나 다른 점수, 회귀 valid 변경을 포함한 테스트 통과 | [output_audit.json](../outputs/output_audit.json), [test_run.txt](test_run.txt) | 통과 |
| 후보 객체의 독립성 | 선언 순서 유지, 후보 간 학습 상태 공유 없음 | 분류·회귀 후보 순서 일치. Pipeline·전처리 분기·대체기·인코더·모델 객체를 서로 공유하지 않음 | [test_run.txt](test_run.txt), [test_project.py](../test_project.py)의 `test_full_candidate_order_and_independent_preprocessing` | 통과 |
| 새 범주·수치 결측 | 8열 유지, 유한한 변환값, 기존 학습 통계 불변 | 미지 범주와 수치 결측을 포함한 2건이 2×8로 변환되고 전처리 snapshot이 유지됨 | [run_summary.json](../outputs/run_summary.json)의 `boundary_checks`, [test_run.txt](test_run.txt) | 통과 |
| 입력 누락·추가·빈 입력 | 잘못된 입력을 기본값으로 숨기지 않고 명확히 거절 | `priority` 누락, `status` 추가, 빈 입력에 `check_features`가 `ValueError` 발생. 학습 전체 결측 열도 `training=True` 검사에서 거절 | [test_run.txt](test_run.txt), [checks.py](../checks.py), [test_project.py](../test_project.py) | 통과 |
| 분할 ID 중복 | 같은 요청의 중복 소속을 로더가 거절 | 임시 복사본의 중복 `split_manifest`를 `load_data`가 거절 | [output_audit.json](../outputs/output_audit.json)의 `loader_boundary_checks` | 통과 |
| test의 평가 배제 | test의 입력·정답을 모델 평가에 사용하지 않음 | 저장 예측은 valid ID만 포함. 임시 복사본의 test 20행 입력·두 정답을 오염시켜도 반환 train/valid 입력·두 정답은 동일하며 test X/y는 반환하지 않음 | [output_audit.json](../outputs/output_audit.json), [run_summary.json](../outputs/run_summary.json)의 `test_evaluated=false` | 통과 |
| 원본 데이터 보존 | 실패 입력 실험이 원본 CSV를 바꾸지 않음 | 분할 중복·test 오염은 임시 복사본에서만 수행. 원본 자료·분할표·순서표 SHA-256 유지 | [output_audit.json](../outputs/output_audit.json)의 `original_data_hashes_unchanged` | 통과 |
| 중요도와 별도 탐색 | Forest MDI 8행·순열 4행, TreeGrid 4조건을 기본 비교와 구분 | 중요도 이름·메타데이터·범위 확인. Grid 후보와 최적 조건 일치, `basic_table_replaced=false` | [output_audit.json](../outputs/output_audit.json), [separate_tree_grid.json](../outputs/separate_tree_grid.json) | 통과 |
| 같은 프로세스 저장·복원 | 직접 만든 모델의 저장 전후 예측 일치 | `same_predictions=true`, `trusted_local_files_only=true` | [persistence_check.json](../outputs/persistence_check.json) | 통과 |
| 새 프로세스 저장·복원 | 새 프로세스에서 저장 결과 재현 | 분류 클래스·분류 실패 확률·회귀 예측 세 항목 모두 `true` | [fresh_process_restore.json](../outputs/fresh_process_restore.json) | 통과 |
| 새 요청 N001·N002 | 입력 검사를 거쳐 두 요청에 유한한 예측 출력 | 명령 종료 코드 0, 두 요청의 실패 여부·확률·예측 시간 생성. 정답은 없으므로 성능 지표는 계산하지 않음 | [new_request_predictions.csv](../outputs/new_request_predictions.csv), [predict_new.py](../predict_new.py) | 실행 통과 |

출력 감사가 확인한 행 수는 아래와 같다. `fold_manifest.csv`는 두 task × 5 fold × train ID 60개이므로 600행이며, 모델마다 같은 fold를 중복 기록한 표가 아니다.

| 파일 | 행 수 |
|---|---:|
| classification_comparison.csv | 6 |
| regression_comparison.csv | 6 |
| cv_scores.csv | 60 |
| validation_predictions.csv | 240 |
| fold_manifest.csv | 600 |
| feature_importance.csv | 12 |
| separate_tree_grid.csv | 4 |

새 요청의 실제 출력은 N001: 실패 예측 0·실패 확률 약 0.086801·1177.000ms, N002: 실패 예측 0·실패 확률 약 0.435104·1469.667ms였다. N002에는 글자 수 결측과 `translation`·`urgent`라는 미지 범주가 있다. 이 실행은 입력 처리와 예측 흐름의 확인이며, 새 요청의 실제 정답이나 보정된 운영 위험도를 확인한 실험은 아니다.

## 2. PowerShell 재현 명령

[README](../README.md)에 따라 가상환경과 지정 패키지를 준비한 뒤, 프로젝트 루트에서 아래 순서로 실행한다. 모델 파일은 먼저 직접 생성해야 하며 GitHub 배포물에는 포함하지 않는다.

```powershell
# 학습용 데이터 개요와 level 3 결과를 생성
.\.venv\Scripts\python.exe -X utf8 inspect_data.py
.\.venv\Scripts\python.exe -X utf8 main.py --level 3

# 17개 테스트를 실행하고 로그를 보관
.\.venv\Scripts\python.exe -X utf8 -m unittest -v test_project.py 2>&1 | Tee-Object -FilePath .\reports\test_run.txt

# 저장된 결과의 지표·ID·fold·데이터 경계를 감사
.\.venv\Scripts\python.exe -X utf8 verify_outputs.py

# 방금 직접 생성한 모델의 새 프로세스 복원과 새 요청 예측
.\.venv\Scripts\python.exe -X utf8 restore_check.py --trust-local-models
.\.venv\Scripts\python.exe -X utf8 predict_new.py --trust-local-models
```

`verify_outputs.py`는 저장 결과를 재검산하며 모델을 추가 학습하거나 `expected_outputs/`를 읽어 정답을 대신 만들지 않는다. test 오염 검사는 로더의 반환 자료가 변하지 않는지를 확인한 것이며, 오염된 test로 성능을 계산한 검사가 아니다.

## 3. 검증 범위와 한계

입력 차단은 `check_features`를 명시적으로 호출하는 경로의 계약이다. `load_data`와 `predict_new.py`는 이 검사를 거치며, 경계 테스트도 같은 함수를 사용한다. 저장된 Pipeline 자체가 모든 추가 열을 거절한다고 보장하지 않는다. `Pipeline.predict`를 직접 호출하는 별도 프로그램에서도 예측 전에 입력 검사를 적용해야 한다.

이 기록은 코드의 입력·분할·선택·산출물·저장 복원 동작을 확인한다. **운영 예측 성능, 오류 비용 절감, 모델 우위나 사용자의 이해도를 보증하지 않는다.** 합성 소표본의 성능과 한계는 [실험 보고서](ensemble_report.md)에서 별도로 해석했다. 다른 운영체제에서의 재현과 사람인 짝의 독립 검토는 이번 기록에서 확인하지 않았다.
