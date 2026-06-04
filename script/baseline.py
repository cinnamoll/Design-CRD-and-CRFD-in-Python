import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

df = pd.read_csv('data/mlc_churn.csv')
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

cat_cols = df.select_dtypes(include=['object', 'string']).columns

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

corr_matrix = df.corr().abs()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True,        
            cmap='coolwarm', 
            fmt=".2f",        
            linewidths=0.5)   

plt.title('Biểu đồ Ma trận tương quan')
plt.tight_layout()
plt.show()

cols_to_drop = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge', 'voice_mail_plan', 'churn']
X = df.drop(columns=cols_to_drop)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1234)

model_M = RandomForestClassifier(random_state=1234)
model_M.fit(X_train, y_train)

y_pred = model_M.predict(X_test)
f1_M = f1_score(y_test, y_pred, pos_label=1)

print(f1_M)

os.makedirs('results/baseline', exist_ok=True)

baseline = pd.DataFrame([{'model': 'Baseline Random Forest (M)', 'f1_score': f1_M}])
baseline.to_csv('results/baseline/baseline_res.csv', index=False)

importances = model_M.feature_importances_
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
importance.to_csv('results/baseline/feature_importances.csv', index=False)