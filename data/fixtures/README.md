# 정상·경계·실패 입력

모두 수업용 합성 데이터입니다. 원본과 분할표를 덮어쓰지 마세요.

| 파일 | 기대 결과 |
|---|---|
| ../new_requests.csv | 결측·미지 범주도 8열로 변환, 학습 통계 불변 |
| missing_priority.csv | 예측 전 ValueError: priority 누락 |
| forbidden_status.csv | 예측 전 ValueError: status 추가/금지 |
| infinite_numeric.csv | 유한값 계약 위반으로 ValueError |
| negative_numeric.csv | 허용 도구 수 음수로 ValueError |
| all_numeric_missing_new.csv | 새 입력의 수치 결측 대체, 8열 유지 |
| duplicate_id.csv | ID 중복으로 ValueError |
| empty_input.csv | 빈 입력으로 ValueError |
| overlap_manifest.csv | 중복 ID 분할표로 ValueError |

K 검사는 코드로 0·1·60·61·실수·bool을 전달해 수행합니다.
K=60은 현재 train60에 허용되지만 좋은 성능이라는 의미는 아닙니다.
