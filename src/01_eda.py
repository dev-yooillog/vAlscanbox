import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, '..', 'data', 'uci_malware_detection.xls')
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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

PALETTE = {'malicious': '#ff4444', 'non-malicious': '#58a6ff'}

df = pd.read_csv(DATA_PATH)
print(f"완료: {df.shape[0]}행 × {df.shape[1]}열")

feature_cols = [c for c in df.columns if c != 'Label']
X = df[feature_cols]
y = df['Label']

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"인코딩: {dict(zip(le.classes_, le.transform(le.classes_)))}")

counts = y.value_counts()
colors = [PALETTE[c] for c in counts.index]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0d1117')

bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='#30363d')
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 3, str(v),
                 ha='center', fontweight='bold', fontsize=13)
axes[0].set_title('클래스 분포', fontsize=14, fontweight='bold')
axes[0].set_ylabel('샘플 수')
axes[0].grid(axis='y')

axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct='%1.1f%%', startangle=140,
            wedgeprops=dict(edgecolor='#0d1117', linewidth=2))
axes[1].set_title('클래스 비율', fontsize=14, fontweight='bold')

imbalance = counts.max() / counts.min()
fig.suptitle(f'EDA — 클래스 불균형 비율: {imbalance:.1f}:1', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '01_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

mal_mask = y == 'malicious'
non_mask = y == 'non-malicious'
act_mal  = X[mal_mask].mean()
act_non  = X[non_mask].mean()
diff     = (act_mal - act_non).abs()
top20    = diff.nlargest(20).index

fig, ax = plt.subplots(figsize=(14, 6))
x_pos = np.arange(len(top20))
w = 0.38
ax.bar(x_pos - w/2, act_mal[top20], w, label='Malicious',     color='#ff4444', alpha=0.85)
ax.bar(x_pos + w/2, act_non[top20], w, label='Non-malicious', color='#58a6ff', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(top20, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('활성화율 (Feature=1 비율)')
ax.set_title('Top-20 차별적 피처 활성화율 비교', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '02_feature_activation.png'), dpi=150, bbox_inches='tight')
plt.show()

top30 = diff.nlargest(30).index
corr_matrix = X[top30].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, linewidths=0.3, ax=ax)
ax.set_title('Top-30 피처 상관관계 히트맵', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '03_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

df_clean = X.copy()
df_clean['label'] = y_enc

zero_var  = df_clean[feature_cols].var() == 0
drop_cols = zero_var[zero_var].index.tolist()
df_clean.drop(columns=drop_cols, inplace=True)

print(f"상수 피처 제거: {len(drop_cols)}개 → 잔여 {df_clean.shape[1]-1}개")
df_clean.to_csv(os.path.join(OUTPUTS_DIR, 'preprocessed_data.csv'), index=False)
print("저장 완료: outputs/preprocessed_data.csv")