import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
    f1_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')
MODELS_DIR  = os.path.join(BASE_DIR, '..', 'models')

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

with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
    saved = pickle.load(f)

model        = saved['model']
model_name   = saved['name']
feature_cols = saved['feature_cols']

X_test, y_test = joblib.load(os.path.join(OUTPUTS_DIR, 'test_set.pkl'))

df    = pd.read_csv(os.path.join(OUTPUTS_DIR, 'preprocessed_data.csv'))
X_all = df.drop(columns=['label'])
y_all = df['label']

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"모델: {model_name}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred,
      target_names=['malicious(0)', 'non-malicious(1)']))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='f1', n_jobs=-1)
print(f"    전체 5-Fold CV F1: {cv_scores}")
print(f"    평균={cv_scores.mean():.4f}  표준편차={cv_scores.std():.4f}")

# Confusion Matrix + ROC Curve 
cm  = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)
tn, fp, fn, tp = cm.ravel()

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#161b22')
disp = ConfusionMatrixDisplay(cm, display_labels=['Malicious', 'Non-malicious'])
disp.plot(ax=ax1, colorbar=False, cmap='Blues')
for text in ax1.texts:
    text.set_color('#f0f6ff')
    text.set_fontsize(16)
    text.set_fontweight('bold')
ax1.set_title(f'Confusion Matrix  [{model_name}]', fontsize=13, fontweight='bold', pad=12)
ax1.text(0.5, -0.16,
         f'TN={tn}  FP={fp} (오탐)  FN={fn} (미탐)  TP={tp}',
         transform=ax1.transAxes, ha='center', fontsize=10, color='#f0883e')

ax2 = fig.add_subplot(gs[1])
ax2.plot(fpr, tpr, color='#58a6ff', lw=2.5, label=f'ROC (AUC={roc_auc_val:.4f})')
ax2.plot([0, 1], [0, 1], '--', color='#555', lw=1.2, label='Random')
ax2.fill_between(fpr, tpr, alpha=0.1, color='#58a6ff')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=13, fontweight='bold', pad=12)
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True)

plt.suptitle(f'vAlscanbox  |  Final Evaluation  [{model_name}]',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(os.path.join(OUTPUTS_DIR, '05_evaluation.png'), dpi=150, bbox_inches='tight')
plt.show()

# Feature Importance (XAI) 
importances = model.feature_importances_
feat_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_df = feat_df.sort_values('importance', ascending=False).head(25).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 9))
colors = ['#ff4444' if i < 5 else '#58a6ff' if i < 15 else '#3fb950'
          for i in range(len(feat_df))]
ax.barh(feat_df['feature'][::-1], feat_df['importance'][::-1],
        color=colors[::-1], edgecolor='#21262d', linewidth=0.5)

for i, (val, fname) in enumerate(zip(feat_df['importance'][:5], feat_df['feature'][:5])):
    ax.text(val + 0.001, len(feat_df)-1-i, f'{val:.4f}',
            va='center', fontsize=9, fontweight='bold', color='#ff4444')

ax.set_xlabel('Feature Importance (Gini Impurity Reduction)')
ax.set_title(f'Top-25 Feature Importance  [{model_name}]\n'
             f'Red=Top5 / Blue=Top6~15 / Green=16~25',
             fontsize=13, fontweight='bold', pad=12)
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '06_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\nTop-10 Feature Importance:")
print(feat_df[['feature', 'importance']].head(10).to_string(index=False))

train_sizes, train_scores, val_scores = learning_curve(
    model, X_all, y_all,
    cv=cv, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, 'o-', color='#3fb950', lw=2, label='Train F1')
ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std,
                alpha=0.15, color='#3fb950')
ax.plot(train_sizes, val_mean, 's-', color='#58a6ff', lw=2, label='CV Val F1')
ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std,
                alpha=0.15, color='#58a6ff')

gap = train_mean[-1] - val_mean[-1]
ax.text(0.97, 0.08,
        f'Train={train_mean[-1]:.4f}\nVal={val_mean[-1]:.4f}\nGap={gap:.4f}',
        transform=ax.transAxes, ha='right', fontsize=11, color='#f0f6ff',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d'))
ax.set_xlabel('Training Samples')
ax.set_ylabel('F1 Score')
ax.set_title(f'Learning Curve  [{model_name}]', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=11)
ax.set_ylim(0.85, 1.05)
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '07_learning_curve.png'), dpi=150, bbox_inches='tight')
plt.show()

summary = {
    'model':            model_name,
    'cv_f1_mean':       round(cv_scores.mean(), 4),
    'cv_f1_std':        round(cv_scores.std(), 4),
    'test_f1':          round(f1_score(y_test, y_pred), 4),
    'test_auc':         round(roc_auc_score(y_test, y_proba), 4),
    'true_positive':    int(tp),
    'false_positive':   int(fp),
    'false_negative':   int(fn),
    'true_negative':    int(tn),
    'false_alarm_rate': round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
    'miss_rate':        round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0,
}
pd.DataFrame([summary]).to_csv(os.path.join(OUTPUTS_DIR, 'final_report.csv'), index=False)

for k, v in summary.items():
    print(f"  {k:22s}: {v}")