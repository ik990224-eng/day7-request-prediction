"""level3 저장 결과를 재학습 없이 검증합니다.

실행: python verify_outputs.py --output outputs --data-dir data
expected_outputs는 읽지 않습니다. 지표는 저장된 실제 예측과 CV 원점수로 다시 계산합니다.
잘못된 입력 실험은 임시 복사본에서만 수행하며, 성공하면 output_audit.json을 저장합니다.
"""
from pathlib import Path
import argparse
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, root_mean_squared_error

from course_support import FEATURES, load_data, sha256, write_json
from model_config import make_folds


ROOT = Path(__file__).resolve().parent
CANDIDATE_ORDER = {
    'classification': ['Dummy', 'Logistic', 'KNN3', 'Tree', 'Forest', 'XGB'],
    'regression': ['Dummy', 'Linear', 'KNN15', 'Tree', 'Forest', 'XGB'],
}
EXPECTED_ROWS = {
    'classification_comparison.csv': 6,
    'regression_comparison.csv': 6,
    'cv_scores.csv': 60,
    'validation_predictions.csv': 240,
    'fold_manifest.csv': 600,
    'feature_importance.csv': 12,
    'separate_tree_grid.csv': 4,
}
DATA_FILES = ('requests.csv', 'split_manifest.csv', 'split_order.csv')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_close(actual, expected, label):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f'{label}: 비유한 수치')
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8, err_msg=label)


def audit_loader_boundaries(data_dir, baseline):
    """보관 test의 값이 개발 자료에 영향을 주지 않는지 복사본으로 검사합니다."""
    with tempfile.TemporaryDirectory(prefix='day7_output_audit_') as temporary:
        temporary = Path(temporary)
        duplicate_dir = temporary / 'duplicate_manifest'
        poisoned_dir = temporary / 'poisoned_test'
        for destination in (duplicate_dir, poisoned_dir):
            destination.mkdir()
            for filename in DATA_FILES:
                shutil.copy2(data_dir / filename, destination / filename)

        manifest_path = duplicate_dir / 'split_manifest.csv'
        manifest = pd.read_csv(manifest_path, dtype={'request_id': str}, encoding='utf-8-sig')
        pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True).to_csv(
            manifest_path, index=False, encoding='utf-8-sig',
        )
        try:
            load_data(duplicate_dir)
        except ValueError as exc:
            duplicate_error = str(exc)
            require('중복' in duplicate_error, '중복 manifest가 다른 이유로 실패했습니다.')
        else:
            raise AssertionError('중복 manifest를 로더가 허용했습니다.')

        # test ID만 이용해 값들을 오염시킵니다. test 정답으로 성능을 계산하지 않습니다.
        requests_path = poisoned_dir / 'requests.csv'
        raw = pd.read_csv(requests_path, dtype={'request_id': str}, encoding='utf-8-sig')
        mask = raw['request_id'].isin(baseline.test_ids)
        require(int(mask.sum()) == 20, '오염 실험의 test ID 수가 20이 아닙니다.')
        raw.loc[mask, ['input_chars', 'allowed_tools', 'failed', 'elapsed_ms']] = -999
        raw.loc[mask, ['task_type', 'priority']] = ' '
        raw.to_csv(requests_path, index=False, encoding='utf-8-sig')
        poisoned = load_data(poisoned_dir)
        for attribute in ('X_train', 'X_valid'):
            pd.testing.assert_frame_equal(getattr(baseline, attribute), getattr(poisoned, attribute))
        for attribute in ('yc_train', 'yc_valid', 'yr_train', 'yr_valid'):
            pd.testing.assert_series_equal(getattr(baseline, attribute), getattr(poisoned, attribute))
        require(poisoned.test_ids == baseline.test_ids, '보관 test ID가 변경되었습니다.')
        require(not hasattr(poisoned, 'X_test') and not hasattr(poisoned, 'y_test'),
                '로더가 보관 test의 입력 또는 정답을 반환했습니다.')

    original_hashes = {filename: sha256(data_dir / filename) for filename in DATA_FILES}
    require(original_hashes == baseline.source_hashes, '원본 데이터가 변경되었습니다.')
    return {
        'duplicate_split_manifest_rejected': True,
        'duplicate_error': duplicate_error,
        'poisoned_test_rows': 20,
        'poisoned_columns': FEATURES + ['failed', 'elapsed_ms'],
        'train_valid_inputs_and_both_targets_unchanged': True,
        'test_inputs_and_targets_not_returned': True,
        'original_data_hashes_unchanged': True,
        'temporary_copies_only': True,
    }


def audit_fold_manifest(manifest, d):
    """저장 ID/위치를 다시 만든 고정 fold와 대조하고 평가 행의 전체 커버리지를 확인합니다."""
    require(set(manifest['task']) == set(CANDIDATE_ORDER), 'fold task 구성 오류')
    require(set(manifest['role']) == {'fit', 'evaluate'}, 'fold role 구성 오류')
    require(not manifest.duplicated(['task', 'fold', 'role', 'request_id']).any(), 'fold 행 중복')
    train_ids = set(d.X_train.index)
    require(set(manifest['request_id']) == train_ids, 'fold에 train 외 ID 또는 누락 ID가 있습니다.')
    require(not set(manifest['request_id']) & (set(d.X_valid.index) | set(d.test_ids)),
            'fold에 valid/test ID가 포함되었습니다.')
    records = []
    for task in CANDIDATE_ORDER:
        target = d.yc_train if task == 'classification' else d.yr_train
        folds = make_folds(d.X_train, target, task=task)
        task_rows = manifest.loc[manifest['task'] == task]
        require(set(task_rows['fold']) == set(range(1, 6)), f'{task}: fold 번호 오류')
        require(len(task_rows) == 300, f'{task}: fold 기록은 300행이어야 합니다.')
        for number, (fit_positions, eval_positions) in enumerate(folds, 1):
            selected = task_rows.loc[task_rows['fold'] == number]
            ids_by_role = {}
            for role, positions, size in [('fit', fit_positions, 48), ('evaluate', eval_positions, 12)]:
                rows = selected.loc[selected['role'] == role]
                require(len(rows) == size, f'{task}/fold{number}/{role}: 행 수 오류')
                np.testing.assert_array_equal(rows['position_in_train'].to_numpy(), positions)
                expected_ids = d.X_train.iloc[positions].index.tolist()
                require(rows['request_id'].tolist() == expected_ids,
                        f'{task}/fold{number}/{role}: 고정 fold의 ID 또는 순서와 다릅니다.')
                ids_by_role[role] = set(expected_ids)
            require(not ids_by_role['fit'] & ids_by_role['evaluate'], f'{task}/fold{number}: fit/evaluate 중복')
            require(ids_by_role['fit'] | ids_by_role['evaluate'] == train_ids,
                    f'{task}/fold{number}: train60 커버리지 오류')
            records.append({'task': task, 'fold': number, 'fit_n': 48, 'eval_n': 12,
                            'disjoint': True, 'train_only': True, 'matches_declared_fold': True})
        for role, repetitions in [('evaluate', 1), ('fit', 4)]:
            counts = task_rows.loc[task_rows['role'] == role, 'request_id'].value_counts()
            require(len(counts) == 60 and (counts == repetitions).all(),
                    f'{task}: 각 train 행의 {role} 등장 횟수 오류')
    return records


def audit_outputs(output=ROOT / 'outputs', data_dir=ROOT / 'data'):
    output, data_dir = Path(output).resolve(), Path(data_dir).resolve()
    d = load_data(data_dir)
    require((len(d.X_train), len(d.X_valid), len(d.test_ids)) == (60, 20, 20), '60/20/20 분할 오류')
    tables = {}
    for filename, count in EXPECTED_ROWS.items():
        table = pd.read_csv(output / filename, encoding='utf-8-sig')
        require(len(table) == count, f'{filename}: {len(table)}행, 필요한 행 수 {count}')
        tables[filename] = table
    summary = json.loads((output / 'run_summary.json').read_text(encoding='utf-8'))
    require(summary['level'] == 3, 'level3 실행 결과가 아닙니다.')
    require(summary['test_evaluated'] is False, 'summary의 test 미평가 기록 오류')
    require(summary['data_sha256'] == d.source_hashes, '실행 데이터와 현재 데이터의 해시가 다릅니다.')

    cv = tables['cv_scores.csv']
    predictions = tables['validation_predictions.csv']
    expected_pairs = {(task, name) for task, names in CANDIDATE_ORDER.items() for name in names}
    for label, table in [('CV', cv), ('예측', predictions)]:
        actual_pairs = set(table[['task', 'model']].itertuples(index=False, name=None))
        require(actual_pairs == expected_pairs, f'{label}: 후보/과제 구성 오류')
    require(cv['median_audit_passed'].astype(str).eq('True').all(), 'fold 중앙값 감사 실패 기록')
    prediction_checks, cv_checks, choices = [], [], {}

    for task, names in CANDIDATE_ORDER.items():
        comparison = tables[f'{task}_comparison.csv']
        require(comparison['model'].tolist() == names, f'{task}: 후보 선언 순서 오류')
        require(comparison['task'].eq(task).all(), f'{task}: 비교표 task 오류')
        is_classification = task == 'classification'
        metric = 'F1' if is_classification else 'RMSE_ms'
        require(comparison['CV_metric'].eq(metric).all(), f'{task}: CV 지표 이름 오류')
        target = d.yc_valid if is_classification else d.yr_valid

        for name in names:
            label = f'{task}/{name}'
            row = comparison.loc[comparison['model'] == name].iloc[0]
            selected = predictions.loc[(predictions['task'] == task) & (predictions['model'] == name)]
            require(len(selected) == 20, f'{label}: valid 예측 행 수 오류')
            require(selected['request_id'].tolist() == d.X_valid.index.tolist(), f'{label}: valid ID/순서 오류')
            assert_close(selected['actual'], target.to_numpy(), f'{label}: 실제 정답 정렬')
            predicted = selected['predicted'].to_numpy(dtype=float)
            require(np.isfinite(predicted).all(), f'{label}: 비유한 예측')
            actual = target.to_numpy()
            if is_classification:
                require(np.isin(predicted, [0, 1]).all(), f'{label}: 분류 예측 라벨 오류')
                recalculated = {
                    'F1': float(f1_score(actual, predicted, zero_division=0)),
                    'FP': int(((actual == 0) & (predicted == 1)).sum()),
                    'FN': int(((actual == 1) & (predicted == 0)).sum()),
                }
                probability = selected['failure_probability'].to_numpy(dtype=float)
                require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
                        f'{label}: 실패 확률 범위 오류')
            else:
                recalculated = {
                    'MAE_ms': float(mean_absolute_error(actual, predicted)),
                    'RMSE_ms': float(root_mean_squared_error(actual, predicted)),
                }
                assert_close(selected['residual_ms'], actual - predicted, f'{label}: 회귀 잔차')
            for key, value in recalculated.items():
                assert_close(row['valid_' + key], value, f'{label}: valid {key} 재계산')
            prediction_checks.append({'task': task, 'model': name, 'rows': 20,
                                      'valid_ids_and_targets_in_order': True, 'recalculated_metrics': recalculated})

            fold_scores = cv.loc[(cv['task'] == task) & (cv['model'] == name)]
            require(len(fold_scores) == 5 and fold_scores['fold'].tolist() == list(range(1, 6)),
                    f'{label}: CV fold 구성/순서 오류')
            require(fold_scores['fit_n'].eq(48).all() and fold_scores['eval_n'].eq(12).all(),
                    f'{label}: CV 48/12 행 수 오류')
            require(fold_scores['metric'].eq(metric).all(), f'{label}: fold 지표 이름 오류')
            values = fold_scores[['score', 'train_score']].to_numpy(dtype=float)
            require(np.isfinite(values).all() and (values >= 0).all(), f'{label}: CV 점수 부호/유한성 오류')
            if is_classification:
                require((values <= 1).all(), f'{label}: F1 범위 오류')
            mean, std = float(values[:, 0].mean()), float(values[:, 0].std(ddof=0))
            assert_close(row['CV_mean'], mean, f'{label}: CV 평균')
            assert_close(row['CV_std'], std, f'{label}: CV 표준편차(ddof=0)')
            cv_checks.append({'task': task, 'model': name, 'folds': 5, 'mean': mean,
                              'std_ddof0': std, 'fit_n': 48, 'eval_n': 12, 'median_audit_passed': True})

        # Python의 max/min도 정확한 동점에서는 첫 위치를 선택합니다. valid 열은 참조하지 않습니다.
        scores = comparison['CV_mean'].to_numpy(dtype=float)
        choose = max if is_classification else min
        winner = choose(range(len(comparison)), key=lambda position: scores[position])
        choices[task] = comparison.iloc[winner]['model']
    require(summary['choices_by_CV'] == choices, 'summary의 선택 모델이 CV 규칙과 다릅니다.')

    fold_checks = audit_fold_manifest(tables['fold_manifest.csv'], d)
    importance = tables['feature_importance.csv']
    require(importance.columns.tolist() == ['feature', 'method', 'importance', 'repeat_std', 'evaluation_data', 'metric'],
            '중요도 열 스키마 오류')
    mdi = importance.loc[importance['method'] == 'MDI']
    permutation = importance.loc[importance['method'] == 'permutation']
    require(len(mdi) == 8 and len(permutation) == 4, 'MDI8/순열4 구성 오류')
    require(mdi['feature'].is_unique and permutation['feature'].tolist() == FEATURES, '중요도 특성 구성 오류')
    require(mdi['evaluation_data'].eq('train60').all() and mdi['metric'].eq('impurity_decrease').all(),
            'MDI 계산 자료/지표 오류')
    require(permutation['evaluation_data'].eq('valid20 (diagnostic)').all() and permutation['metric'].eq('F1_drop').all(),
            '순열 계산 자료/지표 오류')
    require(np.isfinite(importance['importance']).all(), '비유한 중요도')
    require((mdi['importance'] >= 0).all() and mdi['repeat_std'].isna().all(), 'MDI 값/반복편차 오류')
    assert_close(mdi['importance'].sum(), 1.0, 'MDI 중요도 합')
    require(np.isfinite(permutation['repeat_std']).all() and (permutation['repeat_std'] >= 0).all(),
            '순열 반복 표준편차 오류')

    grid = tables['separate_tree_grid.csv']
    grid_parameters = set(grid[['param_model__max_depth', 'param_model__min_samples_leaf']].itertuples(index=False, name=None))
    require(grid_parameters == {(2, 1), (2, 3), (4, 1), (4, 3)}, '별도 TreeGrid 사전4조합 오류')
    require(np.isfinite(grid[['mean_test_score', 'std_test_score']].to_numpy()).all(), 'TreeGrid 비유한 점수')
    require(grid['mean_test_score'].between(0, 1).all() and grid['std_test_score'].ge(0).all(), 'TreeGrid F1/편차 범위 오류')
    grid_summary = json.loads((output / 'separate_tree_grid.json').read_text(encoding='utf-8'))
    best_grid = grid.iloc[int(np.argmax(grid['mean_test_score'].to_numpy()))]
    expected_grid_parameters = {'model__max_depth': int(best_grid['param_model__max_depth']),
                                'model__min_samples_leaf': int(best_grid['param_model__min_samples_leaf'])}
    require(grid_summary['best_params'] == expected_grid_parameters, 'TreeGrid 선택 조건 오류')
    assert_close(grid_summary['best_CV_F1'], best_grid['mean_test_score'], 'TreeGrid 선택 점수')
    require(grid_summary['basic_table_replaced'] is False and grid_summary['test_evaluated'] is False,
            'TreeGrid 기본 비교 분리/test 미평가 기록 오류')

    loader_checks = audit_loader_boundaries(data_dir, d)
    audit = {
        'status': 'passed',
        'audit_version': 1,
        'output': output.name,
        'data_dir': data_dir.name,
        'training_performed': False,
        'expected_outputs_used': False,
        'row_counts': {filename: len(table) for filename, table in tables.items()},
        'prediction_checks': prediction_checks,
        'cv_checks': cv_checks,
        'choices_by_recalculated_CV': choices,
        'summary_choices_match': True,
        'fold_checks': fold_checks,
        'fold_evaluate_once_fit_four_times_per_train_id': True,
        'importance_checks': {'MDI_rows': 8, 'permutation_rows': 4, 'metadata_and_ranges_valid': True},
        'separate_tree_grid_checks': {'rows': 4, 'predeclared_parameters_match': True,
                                     'best_params_match': True, 'basic_table_replaced': False},
        'loader_boundary_checks': loader_checks,
        'data_sha256': d.source_hashes,
        'output_sha256': {filename: sha256(output / filename) for filename in [
            *EXPECTED_ROWS, 'run_summary.json', 'separate_tree_grid.json',
        ]},
    }
    write_json(output / 'output_audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    audit = audit_outputs(args.output, args.data_dir)
    print(json.dumps({'status': audit['status'], 'audit': str(args.output / 'output_audit.json'),
                      'row_counts': audit['row_counts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
