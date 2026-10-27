# GitHub에 학습 과정을 기록하는 방법

이 프로젝트의 저장소는 https://github.com/ik990224-eng/day7-request-prediction 입니다. 포트폴리오용 공개 저장소이므로 로그인 없이 코드·설명·결과를 열람할 수 있습니다.

## 이미 만들어진 저장소에서 이어서 학습하기

```powershell
git clone https://github.com/ik990224-eng/day7-request-prediction.git
cd day7-request-prediction
```

README의 단계별 명령을 실행하고, 변경 의도와 결과를 먼저 확인합니다.

```powershell
git status --short
git diff -- main.py preprocessing.py evaluate.py
git diff -- README.md reports/ensemble_report.md
```

예를 들어 evaluate.py를 수정했다면 관련 파일만 선택합니다.

```powershell
git add evaluate.py test_project.py
git add reports/ensemble_report.md reports/test_record.md
git add outputs/classification_comparison.csv outputs/regression_comparison.csv
# 다시 실행한 실험의 나머지 CSV/JSON도 확인 후 명시적으로 추가합니다.
git diff --cached --stat
git diff --cached
git commit -m "Explain and verify updated model comparison"
git push
git log --oneline -5
git status --short
```

git add는 올릴 변경 선택, commit은 로컬 이력 저장, push는 GitHub로 전송입니다. commit만으로 웹 저장소가 갱신되지는 않습니다.

## 처음부터 새 저장소를 직접 만들 때

GitHub의 New repository에서 이름과 공개 범위를 선택합니다. 로컬 프로젝트를 올릴 때는 웹에서 README를 미리 만들지 않은 빈 저장소가 간단합니다. 이미 같은 이름이 있으면 다른 이름을 쓰세요.

기존 이력이 없는 새 프로젝트 폴더에서만 실행합니다. 이미 저장소가 있으면 재초기화하거나 이력을 삭제하지 않습니다.

```powershell
git init -b main
git config user.name "본인 GitHub 사용자명"
git config user.email "본인 GitHub 계정의 noreply 이메일"
# .gitignore와 파일 목록을 확인하고 필요한 파일/폴더만 명시적으로 git add
git diff --cached --stat
git commit -m "Complete request prediction experiment with explanations"
git remote add origin https://github.com/ik990224-eng/day7-request-prediction.git
git push -u origin main
```

user.name/email은 --global 없이 해당 저장소에만 설정합니다. 첫 커밋에는 Python 소스, data, requirements, LICENSE, 문서와 실제 outputs의 CSV/JSON을 포함합니다. 원본 LICENSE의 저작권을 유지합니다.

.venv·캐시·비밀키·모델 joblib은 .gitignore로 제외합니다. 모델은 main.py로 다시 생성합니다. 인증은 GitHub/Git Credential Manager의 정상 로그인 절차를 이용하고 토큰을 코드에 적지 않습니다.
