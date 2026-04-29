import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mean_squared_error
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ── Create output folder (IMPORTANT FIX) ──
os.makedirs("outputs", exist_ok=True)

# ── Style ───────────────────────────────────────────────
C = {'dark': '#1F4E79', 'mid': '#2E75B6', 'light': '#5BA3D9',
     'pale': '#D6E8F7', 'orange': '#C55A11', 'green': '#375623',
     'red': '#C00000', 'grey': '#888888'}

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     'figure.dpi': 150})

# ── Load ───────────────────────────────────────────────
df = pd.read_csv('fuel_prices_1970_2026.csv', parse_dates=['Date'])

df = df.sort_values('Date').reset_index(drop=True)

df['Rolling_Vol'] = df['Crude_Oil_Price'].rolling(12).std()
df['MA_3M'] = df['Crude_Oil_Price'].rolling(3).mean()
df['MA_12M'] = df['Crude_Oil_Price'].rolling(12).mean()

df['Decade'] = (df['Date'].dt.year // 10 * 10).astype(str) + 's'
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Year_Float'] = df['Year'] + (df['Month'] - 1) / 12

df_clean = df.dropna(subset=['Rolling_Vol']).copy()
df_clean['Log_Vol'] = np.log1p(df_clean['Rolling_Vol'])

df_clean['Pct_Change'] = df['Crude_Oil_Price'].pct_change() * 100
df_clean['Pct_Change'] = df_clean['Pct_Change'].fillna(0)

print("="*65)
print("WEEK 6 — STATISTICAL MODELING")
print("="*65)

# ── MODULE A ───────────────────────────────────────────
ts = df.set_index('Date')['Crude_Oil_Price'].dropna()
decomp = seasonal_decompose(ts, model='multiplicative', period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for ax, data, lbl, col in zip(
    axes,
    [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
    ['Observed', 'Trend', 'Seasonal', 'Residual'],
    [C['dark'], C['mid'], C['orange'], C['red']]
):
    ax.plot(data.index, data.values, color=col)
    ax.set_ylabel(lbl)

plt.tight_layout()
plt.savefig('outputs/fig6_decomposition.png')
plt.close()

print("Fig 6 saved")

# ── MODULE C ───────────────────────────────────────────
X = df_clean[['Year_Float']].values
y = df_clean['Rolling_Vol'].values

X_sm = add_constant(X)
model = OLS(y, X_sm).fit()

y_pred = model.predict(X_sm)

dw = durbin_watson(model.resid)
r2 = model.rsquared
rmse = np.sqrt(mean_squared_error(y, y_pred))

fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(df_clean['Date'], y, alpha=0.5)
ax.plot(df_clean['Date'], y_pred, color='orange')

plt.tight_layout()
plt.savefig('outputs/fig7_regression.png')
plt.close()

print("Fig 7 saved")

# ── MODULE D ───────────────────────────────────────────
resid = model.resid

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(y_pred, resid)
axes[0].axhline(0, color='red')

stats.probplot(resid, dist="norm", plot=axes[1])

axes[2].scatter(y_pred, np.sqrt(np.abs(resid)))

plt.tight_layout()
plt.savefig('outputs/fig8_diagnostics.png')
plt.close()

print("Fig 8 saved")

# ── MODULE E ───────────────────────────────────────────
ci_data = {}

for dec in df_clean['Decade'].unique():
    grp = df_clean[df_clean['Decade'] == dec]['Rolling_Vol']
    m = grp.mean()
    ci_data[dec] = float(m)

fig, ax = plt.subplots()

ax.bar(ci_data.keys(), ci_data.values())

plt.tight_layout()
plt.savefig('outputs/fig9_ci.png')
plt.close()

print("Fig 9 saved")

# ── MODULE F ───────────────────────────────────────────
acf_vals = acf(df_clean['Rolling_Vol'], nlags=20)
pacf_vals = pacf(df_clean['Rolling_Vol'], nlags=20)

fig, ax = plt.subplots()
ax.bar(range(len(acf_vals)), acf_vals)

plt.tight_layout()
plt.savefig('outputs/fig10_acf.png')
plt.close()

print("Fig 10 saved")

# ── MODULE G ───────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(df_clean['Date'], df_clean['Rolling_Vol'])

plt.tight_layout()
plt.savefig('outputs/fig11_crisis.png')
plt.close()

print("Fig 11 saved")

# ── SAVE JSON ─────────────────────────────────────────
stats_data = {
    "r2": float(r2),
    "rmse": float(rmse),
    "dw": float(dw)
}

with open("outputs/w6_stats.json", "w") as f:
    json.dump(stats_data, f, indent=2)

print("DONE ✅ All outputs saved in /outputs folder")