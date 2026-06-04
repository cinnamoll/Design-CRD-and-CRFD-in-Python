import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data/mlc_churn.csv')
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

cols_to_drop = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge', 'number_vmail_messages', 'churn']
X = df.drop(columns=cols_to_drop)
y = df['churn']

k = [3, 5, 10]
max_depth = [3, 5, None]
crfd_results = []

for i in k:
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=1234)
    
    for depth in max_depth:
        depth_str = "None" if depth is None else str(depth)
        
        rf = RandomForestClassifier(max_depth=depth, random_state=1234)
        
        for train_idx, test_idx in rskf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            
            crfd_results.append({
                'k': i, 
                'max_depth': depth_str, 
                'f1_score': f1
            })

os.makedirs('results/crfd', exist_ok=True)
f1_res_crfd = pd.DataFrame(crfd_results)
f1_res_crfd.to_csv('results/crfd/crfd_results.csv', index=False)

model_crfd = ols('f1_score ~ C(k) + C(max_depth) + C(k):C(max_depth)', data=f1_res_crfd).fit()
anova_table = sm.stats.anova_lm(model_crfd, typ=2)

with open('results/crfd/statistical_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("ANOVA 2 chiều \n")
    f.write(str(anova_table) + "\n\n")

    f.write("OLS\n")
    f.write(str(model_crfd.summary().tables[1]) + "\n")

fig, ax = plt.subplots(figsize=(8, 6))
interaction_plot(x=f1_res_crfd['k'], trace=f1_res_crfd['max_depth'], response=f1_res_crfd['f1_score'], 
                 colors=['red', 'green', 'blue'], markers=['o', 'D', '^'], ax=ax)

plt.title('Đồ thị Tương tác giữa Số fold (k) và Độ sâu cây (max_depth) lên F1-Score', fontsize=12)
plt.xlabel('Số fold (k)', fontsize=11)
plt.ylabel('Trung bình F1-Score', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()