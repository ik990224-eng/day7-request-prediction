"""Day7 앙상블 비교 프로젝트. 모든 상대경로는 이 파일 위치를 기준으로 합니다."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from course_support import load_data, write_json, environment
from model_config import make_folds, fold_table
from preprocessing import candidates
from evaluate import run_candidate, importance_table, choose_by_cv
from checks import verify_development_data, boundary_checks
ROOT=Path(__file__).resolve().parent
DEFAULT_LEVEL=3

def run_project(data_dir=ROOT/'data', output=ROOT/'outputs', level=DEFAULT_LEVEL):
    if level not in (1,2,3): raise ValueError('level은 1~3')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    d=load_data(data_dir); integrity=verify_development_data(d)
    cls_folds=make_folds(d.X_train,d.yc_train)
    tasks=['classification'] if level==1 else ['classification','regression']
    comparisons={}; models={}; all_scores=[]; all_pred=[]; all_folds=[]; choices={}
    # TODO D7-07-M1 (구현 완료): 같은 조건으로 모델을 비교하고 CV로 선택합니다.
    # [풀이 설명] 이 부분은 전처리·후보 구성·평가·CV 선택 함수를 연결해 아래 저장 코드의 입력을 준비합니다.
    # data_dir는 원본/고정 분할/순서표 위치이고 output은 결과 저장 폴더입니다. 기본값은 이 파일 기준 경로입니다.
    # level=1은 기본 분류 4후보, level>=2는 이전 모델과 회귀를 포함하며 level=3은 별도 탐색도 실행합니다.
    # tasks를 순회하면서 candidates(include_previous=level>=2, task=task)로 미학습 후보 사전을 만듭니다.
    # 후보 사전의 순서는 동점 규칙의 일부이므로 items() 순서를 유지하여 비교 행을 쌓습니다.
    # 분류에는 이미 만든 cls_folds를 재사용하고, 회귀에는 make_folds(d.X_train, d.yr_train, task='regression')을 씁니다.
    # 분류는 클래스 비율을 유지하는 StratifiedKFold, 회귀는 KFold이며 모두 5분할·shuffle·seed 42입니다.
    # 같은 task의 모든 후보는 같은 folds로 평가해야 후보별 데이터 조건이 달라지지 않습니다.
    # fold_table(d.X_train, folds, task)를 all_folds에 task당 한 번 추가하여 ID 소속을 기록합니다.
    # 각 name/pipe에 run_candidate(name, pipe, d, folds, task)를 호출합니다.
    # 반환된 row는 현재 task의 행 목록에, fold 점수 표는 all_scores에, valid 예측 표는 all_pred에 추가합니다.
    # 반환된 fitted Pipeline은 models[(task, name)]에 보관하여 중요도·경계 검사·저장 복원에 재사용합니다.
    # 한 task의 모든 후보가 끝나면 행 목록으로 DataFrame을 만들어 comparisons[task]에 넣습니다.
    # choose_by_cv(비교표, task)로 선택한 모델 이름은 choices[task]에 기록합니다.
    # comparisons는 task별 표, models는 (task, 모델명)별 학습 객체, choices는 task별 선택 이름인 사전입니다.
    # all_scores/all_pred/all_folds는 DataFrame 목록이므로 아래 pd.concat이 한 CSV로 합칠 수 있어야 합니다.
    # 분류 실패 확률과 회귀 잔차는 run_candidate가 계산하므로 이 연결부에서 다시 예측할 필요는 없습니다.
    # level>=2의 저장·복원과 level>=3의 TreeGrid 실행은 아래 코드가 담당합니다.
    # TreeGrid 결과로 기본 비교표를 바꾸지 않고, valid 순위로 choices를 바꾸거나 보관 test를 평가하지 않습니다.
    for task in tasks:
        folds = cls_folds if task == 'classification' else make_folds(
            d.X_train, d.yr_train, task='regression'
        )
        all_folds.append(fold_table(d.X_train, folds, task))
        rows = []
        for name, pipe in candidates(include_previous=level >= 2, task=task).items():
            print(f'[{task}] {name}: train 내부 5-fold 평가 후 train 전체 학습')
            row, scores, predictions, fitted = run_candidate(name, pipe, d, folds, task)
            rows.append(row)
            all_scores.append(scores)
            all_pred.append(predictions)
            models[(task, name)] = fitted
        comparisons[task] = pd.DataFrame(rows)
        choices[task] = choose_by_cv(comparisons[task], task)
        print(f'[{task}] CV 기준 선택: {choices[task]}')
    for task,table in comparisons.items():
        table.to_csv(output/f'{task}_comparison.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_scores,ignore_index=True).to_csv(output/'cv_scores.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_pred,ignore_index=True).to_csv(output/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_folds,ignore_index=True).to_csv(output/'fold_manifest.csv',index=False,encoding='utf-8-sig')
    importance=importance_table(models[('classification','Forest')],d,include_permutation=level>=2)
    importance.to_csv(output/'feature_importance.csv',index=False,encoding='utf-8-sig')
    check=boundary_checks(models[('classification',choices['classification'])],d)
    if level>=2:
        import joblib
        # 직접 생성한 파일만 같은 환경에서 복원합니다. 외부 pickle/joblib은 로드하지 않습니다.
        for task,name in choices.items():
            filename=output/f'{task}_pipeline.joblib'
            joblib.dump(models[(task,name)],filename)
            restored=joblib.load(filename)
            a=models[(task,name)].predict(d.X_valid); b=restored.predict(d.X_valid)
            if not np.allclose(a,b): raise AssertionError('저장·복원 예측 불일치')
        write_json(output/'persistence_check.json',{'same_predictions':True,'trusted_local_files_only':True})
    if level>=3:
        # 별도 탐색 실험은 기본 비교표를 변경하지 않습니다.
        from search_extension import run_search
        run_search(d,cls_folds,output)
    summary={'level':level,'choices_by_CV':choices,'tie_rule':'정확한 동점이면 후보 선언 순서 우선',
        'integrity':integrity,'boundary_checks':check,'test_evaluated':False,
        'selection_metric':{'classification':'mean fold F1','regression':'mean fold RMSE'},
        'data_sha256':d.source_hashes,'versions':environment(),
        'warnings':['CV는 모델 선택용이다. valid도 이전 실습부터 반복 진단했으므로 독립 최종 평가가 아니다.',
                    '중요도는 인과관계가 아니다. valid 중요도만으로 입력 특성을 자동 삭제하지 않는다.']}
    write_json(output/'run_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--level',type=int,choices=[1,2,3],default=DEFAULT_LEVEL)
    args=p.parse_args()
    run_project(args.data_dir,args.output,args.level)
if __name__=='__main__': main()
