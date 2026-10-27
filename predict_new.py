"""7단계: 직접 학습·저장한 모델로 새 요청을 예측하는 사용 예시."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from course_support import check_features, read_table

ROOT = Path(__file__).resolve().parent


def predict_new(input_path, output):
    import joblib
    # 금지 열을 조용히 버리지 않도록 Pipeline.predict보다 먼저 입력 계약을 검사합니다.
    X = check_features(read_table(input_path).set_index('request_id'))
    classification = joblib.load(output / 'classification_pipeline.joblib')
    regression = joblib.load(output / 'regression_pipeline.joblib')
    positive_column = list(classification.classes_).index(1)
    results = pd.DataFrame({
        'request_id': X.index,
        'predicted_failed': classification.predict(X),
        'failure_probability': classification.predict_proba(X)[:, positive_column],
        'predicted_elapsed_ms': regression.predict(X),
    })
    if not np.isfinite(results.select_dtypes(include='number')).all().all():
        raise AssertionError('예측에 비유한 값이 있습니다.')
    results.to_csv(output / 'new_request_predictions.csv', index=False, encoding='utf-8-sig')
    print(results.to_string(index=False))
    print('새 요청에는 정답이 없으므로 이 파일로 정확도를 계산하지 않습니다.')
    print('실패 확률은 보정된 운영 위험도가 아닙니다. 실제 실패율로 단정하지 않습니다.')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data' / 'new_requests.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--trust-local-models', action='store_true')
    args = parser.parse_args()
    if not args.trust_local_models:
        parser.error('직접 생성한 모델일 때만 --trust-local-models를 지정하세요.')
    predict_new(args.input, args.output)
