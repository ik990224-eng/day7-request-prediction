"""1단계: 모델을 만들기 전에 train/valid의 의미와 상태를 확인합니다."""
from pathlib import Path
import argparse
import json
from course_support import load_data, write_json

ROOT = Path(__file__).resolve().parent


def inspect_data(data_dir=ROOT / 'data', output=ROOT / 'outputs'):
    d = load_data(data_dir)
    summary = {
        'purpose': '요청 실행 전 실패 여부와 실행시간을 예측하는 교육용 실험',
        'source': '교육용 합성 100건, 실제 서비스 기록 아님',
        'features': list(d.X_train.columns),
        'targets': {'classification': 'failed: 0 성공 / 1 실패', 'regression': 'elapsed_ms: 실행시간(ms)'},
        'excluded_features': ['request_id', 'failed', 'elapsed_ms', 'status', 'retry_count'],
        'train': {
            'rows': len(d.X_train),
            'class_counts': d.yc_train.value_counts().sort_index().to_dict(),
            'missing_by_feature': d.X_train.isna().sum().to_dict(),
            'runtime_ms': d.yr_train.describe().to_dict(),
        },
        'valid': {
            'rows': len(d.X_valid),
            'class_counts': d.yc_valid.value_counts().sort_index().to_dict(),
            'missing_by_feature': d.X_valid.isna().sum().to_dict(),
        },
        'heldout_test_id_count': len(d.test_ids),
        'test_evaluated': False,
        # R001은 valid에 속한 설명용 사례입니다. 이것을 보고 모델 설정을 바꾸지는 않습니다.
        'example_R001': {
            'split': 'valid',
            'inputs': d.X_valid.loc['R001'].to_dict(),
            'actual_failed': int(d.yc_valid.loc['R001']),
            'actual_elapsed_ms': float(d.yr_valid.loc['R001']),
        },
        'data_sha256': d.source_hashes,
    }
    write_json(Path(output) / 'data_overview.json', summary)
    print('입력 X: 실행 전에 알고 있는 요청 정보 4개')
    print(d.X_train.head().to_string())
    print('\n정답 y: 실행 후에 관측한 실패 여부와 실행시간')
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    inspect_data(args.data_dir, args.output)
