import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import f1_score, roc_auc_score
from xgboost                 import XGBClassifier
import joblib

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')
MODELS_DIR  = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#c9d1d9',
    'ytick.color':      '#c9d1d9',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.alpha':       0.5,
})

plt.rcParams['font.family'] = 'Malgun Gothic'  
plt.rcParams['axes.unicode_minus'] = False  

df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'preprocessed_data.csv'))
X  = df.drop(columns=['label'])
y  = df['label']
print(f"    로드: {X.shape[0]}개 샘플, {X.shape[1]}개 피처")
print(f"    클래스 분포: {dict(y.value_counts())}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} / Test: {len(X_test)} (stratified)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models_config = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'max_depth':         [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'criterion':         ['gini', 'entropy'],
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth':    [10, 20, None],
            'max_features': ['sqrt', 'log2'],
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss',
                               scale_pos_weight=pos_weight),
        'params': {
            'n_estimators':  [100, 200],
            'max_depth':     [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
        }
    },
}

results     = {}
best_models = {}

for name, cfg in models_config.items():
    print(f"\n[·] {name} GridSearchCV 시작...")
    gs = GridSearchCV(cfg['model'], cfg['params'],
                      scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best    = gs.best_estimator_
    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    results[name] = {
        'best_params': gs.best_params_,
        'cv_f1':       round(gs.best_score_, 4),
        'test_f1':     round(f1_score(y_test, y_pred), 4),
        'test_auc':    round(roc_auc_score(y_test, y_proba), 4),
    }
    best_models[name] = best
    print(f"    Best : {gs.best_params_}")
    print(f"    CV F1={results[name]['cv_f1']}  |  "
          f"Test F1={results[name]['test_f1']}  |  "
          f"AUC={results[name]['test_auc']}")

names  = list(results.keys())
cv_f1s = [results[n]['cv_f1']   for n in names]
tf1s   = [results[n]['test_f1'] for n in names]
aucs   = [results[n]['test_auc'] for n in names]

x, w = np.arange(len(names)), 0.28
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - w, cv_f1s, w, label='CV F1',    color='#3fb950', alpha=0.85)
b2 = ax.bar(x,     tf1s,   w, label='Test F1',  color='#58a6ff', alpha=0.85)
b3 = ax.bar(x + w, aucs,   w, label='Test AUC', color='#f0883e', alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=13)
ax.set_ylim(0.94, 1.05)
ax.set_ylabel('Score')
ax.set_title('vAlscanbox | 모델 성능 비교 (GridSearchCV)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '04_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

best_name = max(results, key=lambda n: results[n]['test_f1'])
print(f"\n최고 모델: {best_name}  (Test F1={results[best_name]['test_f1']})")

with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
    pickle.dump({
        'model':        best_models[best_name],
        'name':         best_name,
        'feature_cols': list(X.columns),
    }, f)

joblib.dump((X_test, y_test), os.path.join(OUTPUTS_DIR, 'test_set.pkl'))
pd.DataFrame(results).T.to_csv(os.path.join(OUTPUTS_DIR, 'model_results.csv'))
print("완료: models/best_model.pkl / outputs/model_results.csv")