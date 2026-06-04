import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data/mlc_churn.csv')
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

print(df.dtypes)

cols_to_drop = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge', 'number_vmail_messages', 'churn']
X = df.drop(columns=cols_to_drop)
y = df['churn']

k = [3, 5, 10]
crd_results = []

for i in k:
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=1234)
    rf = RandomForestClassifier(random_state=1234)
    
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        crd_results.append({'k': i, 'f1_score': f1})

os.makedirs('results/crd', exist_ok=True)
f1_res_crd = pd.DataFrame(crd_results)
f1_res_crd.to_csv('results/crd/crd_res.csv', index=False)

f1_k3 = f1_res_crd[f1_res_crd['k'] == 3]['f1_score']
f1_k5 = f1_res_crd[f1_res_crd['k'] == 5]['f1_score']
f1_k10 = f1_res_crd[f1_res_crd['k'] == 10]['f1_score']

stat, p_levene = stats.levene(f1_k3, f1_k5, f1_k10)
print(p_levene)

model = ols('f1_score ~ C(k)', data=f1_res_crd).fit()
tukey = pairwise_tukeyhsd(endog=f1_res_crd['f1_score'], 
                             groups=f1_res_crd['k'], 
                             alpha=0.05)

with open('results/crd/statistical_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(f"p-value = {p_levene:.4f}\n")

    f.write("OLS\n")
    f.write(str(model.summary().tables[1]) + "\n\n")

    f.write("TUKEYHSD\n")
    f.write(str(tukey.summary()) + "\n")

tukey.plot_simultaneous()
plt.title("Tukey HSD")
plt.tight_layout()
plt.show()