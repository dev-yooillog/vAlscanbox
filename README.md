# vAlscanbox
> PE 구조 엔트로피 및 정적 분석 기반의 AI 멀웨어 탐지 시스템

## 프로젝트 개요

현대의 사이버 위협은 하루에도 수만 개씩 쏟아지며, 기존의 패턴(시그니처) 기반 탐지는
신·변종 악성코드를 차단하는 데 한계가 있습니다.

vAlscanbox는 PE(Portable Executable) 파일의 정적 특징을 머신러닝으로 학습시켜
파일이 실행되기 전 악성 여부를 빠르게 탐지하는 분류 파이프라인입니다.

## 프로젝트 구조
```
vAlscanbox/
├── data/
│   └── uci_malware_detection.xls   # 원본 데이터셋
├── src/
│   ├── 01_eda.py                   # EDA 및 전처리
│   ├── 02_modeling.py              # 모델 학습 및 하이퍼파라미터 튜닝
│   └── 03_evaluate.py              # 평가 및 XAI 시각화
├── models/
│   └── best_model.pkl              # 최종 저장 모델 (02 실행 후 생성)
├── outputs/
│   ├── preprocessed_data.csv       # 전처리 완료 데이터
│   ├── test_set.pkl                # 테스트 셋
│   ├── model_results.csv           # 모델별 성능 비교
│   ├── final_report.csv            # 최종 평가 리포트
│   ├── 01_class_distribution.png
│   ├── 02_feature_activation.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_model_comparison.png
│   ├── 05_evaluation.png
│   ├── 06_feature_importance.png
│   └── 07_learning_curve.png
└── README.md
```

## 데이터셋

| 항목 | 내용 |
|---|---|
| 출처 | UCI Malware Detection Dataset |
| 샘플 수 | 373개 |
| 피처 수 | 531개 (바이너리, F_1 ~ F_531) |
| 클래스 | malicious: 301개 / non-malicious: 72개 |
| 클래스 불균형 | 약 4.2 : 1 |

## 사용 기술

- **언어**: Python 3.10
- **핵심 라이브러리**:
  - `scikit-learn` — 모델 학습, GridSearchCV, 평가
  - `xgboost` — XGBoost 분류기
  - `pandas` / `numpy` — 데이터 처리
  - `matplotlib` / `seaborn` — 시각화
  - `joblib` / `pickle` — 모델 직렬화

## 실행 방법

### 1. 패키지 설치

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 2. 순서대로 실행

```bash
cd vAlscanbox/src

python 01_eda.py       # EDA + 전처리 → outputs/preprocessed_data.csv 생성
python 02_modeling.py  # 모델 학습   → models/best_model.pkl 생성
python 03_evaluate.py  # 평가 + XAI → outputs/final_report.csv 생성
```

## 모델 파이프라인원본 데이터
```
│
▼
[01_eda.py]

클래스 분포 시각화
Top-20 차별적 피처 분석
상관관계 히트맵
상수 피처 제거 (분산=0)
│
▼
[02_modeling.py]
Stratified Train/Test Split (8:2)
GridSearchCV (5-Fold CV)
DecisionTree / RandomForest / XGBoost 비교
최고 모델 .pkl 저장
│
▼
[03_evaluate.py]
Classification Report
Confusion Matrix (오탐/미탐 분석)
ROC Curve / AUC
Feature Importance (XAI)
Learning Curve (과적합 진단)
```
## 평가 결과

| 모델 | CV F1 | Test F1 | Test AUC |
|---|---|---|---|
| DecisionTree | 0.9731 | 1.0000 | 1.0000 |
| RandomForest | 0.9826 | 1.0000 | 1.0000 |
| XGBoost | 0.9818 | 1.0000 | 1.0000 |

> **최종 선택 모델**: DecisionTree
> `criterion=entropy, max_depth=5, min_samples_split=2`

> Test F1/AUC = 1.0은 과적합이 아닌 데이터의 선형 분리 가능성에 기인합니다.
> Learning Curve Train-Val Gap = 0.0213 (정상 수렴 확인).

## 핵심 피처 (XAI)

| 순위 | 피처 | 중요도 |
|---|---|---|
| 1 | F_20 | 0.9043 |
| 2 | F_19 | 0.0487 |
| 3 | F_179 | 0.0349 |

`F_20` 단일 피처가 전체 분류 기여도의 약 **90%** 를 차지합니다.

## 평가 지표 설명

| 지표 | 설명 |
|---|---|
| **F1-Score** | 오탐(FP)과 미탐(FN)을 동시에 고려하는 핵심 지표 |
| **ROC-AUC** | 임계값 변화에 따른 전체 분류 성능 |
| **False Positive (오탐)** | 정상 파일을 악성으로 판단 |
| **False Negative (미탐)** | 악성 파일을 정상으로 판단 → 보안상 치명적 |

## 윈도우 한글 폰트 설정

```python
plt.rcParams['font.family']       = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```
