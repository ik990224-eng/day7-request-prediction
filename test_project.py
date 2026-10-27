"""제공 회귀방지 테스트. 완료 후 python -m unittest -v test_project.py"""
import unittest
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from course_support import load_data, check_features, snapshot, FEATURES
from preprocessing import candidates
from evaluate import choose_by_cv, run_candidate
from checks import verify_development_data, boundary_checks
from model_config import make_folds
import pandas as pd

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=load_data()
        cls.m=candidates()['Tree'].fit(cls.d.X_train,cls.d.yc_train)
    def test_split(self):
        self.assertTrue(verify_development_data(self.d)['disjoint'])
    def test_feature_count(self):
        self.assertEqual(self.m['pre'].transform(self.d.X_valid).shape,(20,8))
    def test_no_test_returned(self):
        self.assertFalse(hasattr(self.d,'X_test'));self.assertFalse(hasattr(self.d,'y_test'))
    def test_medians(self):
        np.testing.assert_allclose(self.m['pre'].named_transformers_['num']['imputer'].statistics_,[1315,3])
    def test_boundary(self):
        self.assertTrue(boundary_checks(self.m,self.d)['missing_column_blocked'])
    def test_extra_column(self):
        with self.assertRaises(ValueError): check_features(self.d.X_valid.assign(status='ok'))
    def test_empty(self):
        with self.assertRaises(ValueError): check_features(self.d.X_valid.iloc[:0])
    def test_statistics(self):
        before=snapshot(self.m['pre']);self.m.predict(self.d.X_valid)
        self.assertEqual(before,snapshot(self.m['pre']))
    def test_cv(self):
        folds=make_folds(self.d.X_train,self.d.yc_train)
        self.assertEqual(len(folds),5)
        self.assertTrue(all(len(a)==48 and len(b)==12 and not set(a)&set(b) for a,b in folds))
    def test_choice_ignores_valid(self):
        t=pd.DataFrame({'model':['A','B'],'CV_mean':[0.6,0.5],'valid_F1':[0.1,1.0]})
        self.assertEqual(choose_by_cv(t),'A')
    def test_candidate_freshness(self):
        a=candidates();self.assertIsNot(a['Tree']['pre'],a['Forest']['pre'])
    def test_training_missing_column(self):
        bad=self.d.X_train.copy();bad['input_chars']=np.nan
        with self.assertRaises(ValueError): check_features(bad,training=True)

    def test_regression_choice_minimizes_cv_and_ignores_valid(self):
        table = pd.DataFrame({
            'model': ['A', 'B', 'C'],
            'CV_mean': [650.0, 600.0, 620.0],
            'valid_RMSE_ms': [1.0, 900.0, 2.0],
        })
        self.assertEqual(choose_by_cv(table, task='regression'), 'B')
        table['valid_RMSE_ms'] = [900.0, 2.0, 1.0]
        self.assertEqual(choose_by_cv(table, task='regression'), 'B')

    def test_exact_cv_ties_keep_declared_order(self):
        # 이름순 정렬이나 중복 index의 label 선택으로 후보 순서가 바뀌면 실패합니다.
        for task, scores in [('classification', [0.5, 0.5, 0.4]),
                             ('regression', [600.0, 600.0, 700.0])]:
            with self.subTest(task=task):
                table = pd.DataFrame({
                    'model': ['Z_first', 'A_second', 'C_third'],
                    'CV_mean': scores,
                    'valid_F1': [0.0, 1.0, 0.9],
                    'valid_RMSE_ms': [900.0, 1.0, 2.0],
                }, index=[7, 7, 2])
                self.assertEqual(choose_by_cv(table, task=task), 'Z_first')

    def test_near_cv_scores_are_not_ties(self):
        # 한 단계 차이의 부동소수점도 정확한 동점은 아니며 반올림하지 않고 비교해야 합니다.
        for task, first, better in [
            ('classification', 0.5, np.nextafter(0.5, np.inf)),
            ('regression', 600.0, np.nextafter(600.0, -np.inf)),
        ]:
            with self.subTest(task=task):
                table = pd.DataFrame({'model': ['first', 'better'],
                                      'CV_mean': [first, better]})
                self.assertEqual(choose_by_cv(table, task=task), 'better')

    def test_regression_candidate_cv_matches_independent_fold(self):
        # 가벼운 회귀 트리 하나만 실제 학습하여 scorer 부호와 fold 경계를 검증합니다.
        pipe = candidates(task='regression')['Tree']
        folds = make_folds(self.d.X_train, self.d.yr_train, task='regression')
        row, scores, predictions, fitted = run_candidate(
            'Tree', pipe, self.d, folds, task='regression',
        )
        self.assertEqual(len(folds), 5)
        self.assertEqual(len(scores), 5)
        self.assertEqual(scores['fold'].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue((scores['fit_n'] == 48).all())
        self.assertTrue((scores['eval_n'] == 12).all())
        self.assertTrue((scores['metric'] == 'RMSE_ms').all())
        self.assertTrue(scores['median_audit_passed'].all())
        self.assertTrue(all(len(a) == 48 and len(b) == 12 and not set(a) & set(b)
                            for a, b in folds))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([b for _, b in folds])), np.arange(60),
        )
        values = scores[['score', 'train_score']].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())
        self.assertTrue((scores['score'] > 0).all())
        self.assertEqual(row['CV_metric'], 'RMSE_ms')
        self.assertAlmostEqual(row['CV_mean'], scores['score'].mean(), places=10)
        self.assertAlmostEqual(row['CV_std'], scores['score'].to_numpy().std(ddof=0), places=10)

        # CV API를 다시 호출하지 않고 첫 fold를 직접 fit한 뒤 RMSE 정의로 독립 계산합니다.
        fit_positions, eval_positions = folds[0]
        independent = clone(pipe).fit(
            self.d.X_train.iloc[fit_positions], self.d.yr_train.iloc[fit_positions],
        )
        for positions, column in [(eval_positions, 'score'), (fit_positions, 'train_score')]:
            actual = self.d.yr_train.iloc[positions].to_numpy()
            predicted = independent.predict(self.d.X_train.iloc[positions])
            expected_rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
            self.assertAlmostEqual(scores.iloc[0][column], expected_rmse, places=9)

        # 반환된 최종 모델은 fold 모델이 아닌 train60 재학습본이며 원본 후보는 미학습 상태입니다.
        self.assertIsNot(fitted, pipe)
        self.assertFalse(hasattr(pipe['pre'], 'transformers_'))
        self.assertEqual(fitted['model'].tree_.n_node_samples[0], 60)
        self.assertEqual(predictions['request_id'].tolist(), self.d.X_valid.index.tolist())
        np.testing.assert_allclose(predictions['predicted'], fitted.predict(self.d.X_valid))
        np.testing.assert_allclose(
            predictions['residual_ms'], self.d.yr_valid.to_numpy() - predictions['predicted'],
        )

    def test_full_candidate_order_and_independent_preprocessing(self):
        classification = candidates(include_previous=True, task='classification')
        regression = candidates(include_previous=True, task='regression')
        self.assertEqual(list(classification), ['Dummy', 'Logistic', 'KNN3', 'Tree', 'Forest', 'XGB'])
        self.assertEqual(list(regression), ['Dummy', 'Linear', 'KNN15', 'Tree', 'Forest', 'XGB'])
        pipes = list(classification.values()) + list(regression.values())
        # 바깥 Pipeline뿐 아니라 내부 결측 대체기/인코더도 후보 간 공유되지 않아야 합니다.
        objects = []
        for pipe in pipes:
            objects.extend([pipe, pipe['pre'], pipe['model']])
            for _, branch, _ in pipe['pre'].transformers:
                objects.append(branch)
                objects.extend(step for _, step in branch.steps if not isinstance(step, str))
        self.assertEqual(len({id(obj) for obj in objects}), len(objects))

if __name__=='__main__': unittest.main()
