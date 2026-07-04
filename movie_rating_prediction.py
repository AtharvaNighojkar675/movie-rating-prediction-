"""
Movie Rating Prediction — IMDb Movies India Dataset
=====================================================
Dataset  : IMDb Movies India (15,509 movies)
Features : Genre, Director, Actors, Duration, Year, Votes
Target   : IMDb Rating (1–10)
Models   : Linear Regression, Ridge, Decision Tree,
           Random Forest, Gradient Boosting
"""

import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# Extract zip and read the CSV file
# ─────────────────────────────────────────────
if not os.path.exists('data/IMDb Movies India.csv'):
    with zipfile.ZipFile('archive (3).zip', 'r') as z:
        z.extractall('data/')
    print("✅ ZIP extracted successfully!")

# Read the CSV with latin-1 encoding (needed for special characters)
df = pd.read_csv('data/IMDb Movies India.csv', encoding='latin-1')

print("=" * 65)
print("  IMDb INDIA MOVIES — RAW DATA OVERVIEW")
print("=" * 65)
print(f"  Total Movies  : {df.shape[0]}")
print(f"  Total Columns : {df.shape[1]}")
print(f"  Columns       : {list(df.columns)}")
print(f"\n  Missing Values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# STEP 2 — DATA CLEANING & PREPROCESSING
# Fix data types and handle missing values
# ─────────────────────────────────────────────

# Year column has values like "(2019)" — extract just the number
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(float)

# Duration has values like "120 min" — extract just the number
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)

# Votes has commas like "1,00,000" — remove commas and convert to number
df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Genre column may have multiple genres like "Action, Drama"
# We take only the first (primary) genre
df['Primary_Genre'] = df['Genre'].str.split(',').str[0].str.strip()

# Fill missing values in categorical columns with 'Unknown'
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Primary_Genre']:
    df[col] = df[col].fillna('Unknown')

# Remove rows where Rating is missing (we need this to train the model)
df = df.dropna(subset=['Rating'])
print(f"\n  Rows after removing missing Ratings : {len(df)}")

# Fill missing numeric values with median
df['Year']     = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes']    = df['Votes'].fillna(df['Votes'].median())

print(f"\n  Rating Statistics:\n{df['Rating'].describe().round(2)}")

# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# Convert text/categorical data into numbers
# so the ML model can understand it
# ─────────────────────────────────────────────

# Label Encoding — converts each unique category to a number
# e.g. "Drama" → 2, "Comedy" → 1, "Action" → 0
le_genre = LabelEncoder()
le_dir   = LabelEncoder()
le_a1    = LabelEncoder()
le_a2    = LabelEncoder()
le_a3    = LabelEncoder()

df['Genre_enc']    = le_genre.fit_transform(df['Primary_Genre'])
df['Director_enc'] = le_dir.fit_transform(df['Director'])
df['Actor1_enc']   = le_a1.fit_transform(df['Actor 1'])
df['Actor2_enc']   = le_a2.fit_transform(df['Actor 2'])
df['Actor3_enc']   = le_a3.fit_transform(df['Actor 3'])

# Log transformation on Votes — because votes are heavily skewed
# (a few movies have millions of votes, most have very few)
df['Log_Votes'] = np.log1p(df['Votes'])

# Movie Age — how old the movie is
df['Movie_Age'] = 2024 - df['Year']

# Target Mean Encoding — replace Director/Actor name with their
# average rating across all their movies (very strong predictor!)
dir_mean  = df.groupby('Director')['Rating'].transform('mean')
a1_mean   = df.groupby('Actor 1')['Rating'].transform('mean')
a2_mean   = df.groupby('Actor 2')['Rating'].transform('mean')
df['Dir_AvgRating'] = dir_mean
df['Act_AvgRating'] = (a1_mean + a2_mean) / 2

# Final list of features used for training
FEATURES = [
    'Genre_enc', 'Director_enc', 'Actor1_enc', 'Actor2_enc', 'Actor3_enc',
    'Duration', 'Log_Votes', 'Movie_Age',
    'Dir_AvgRating', 'Act_AvgRating'
]

X = df[FEATURES]
y = df['Rating']

# Split: 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Training Samples : {len(X_train)}")
print(f"  Testing  Samples : {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 4 — MODEL TRAINING & EVALUATION
# Train 5 different models and compare them
# ─────────────────────────────────────────────

# 5 models — from simple to complex
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Decision Tree':     DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                    max_depth=5, random_state=42),
}

# Scale features for linear models (helps them perform better)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

results     = {}
predictions = {}

print("\n" + "=" * 70)
print("  MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE%':>8}")
print("-" * 58)

for name, model in models.items():
    # Use scaled data for linear models, raw for tree-based
    Xtr = X_train_sc if 'Regression' in name else X_train
    Xts = X_test_sc  if 'Regression' in name else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xts)
    predictions[name] = preds

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    # MAPE = Mean Absolute Percentage Error (how wrong in % on average)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    print(f"{name:<22} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} {mape:>7.2f}%")

# Pick the best model based on highest R² score
best_name  = max(results, key=lambda k: results[k]['R2'])
best_model = models[best_name]
best_preds = predictions[best_name]
print(f"\n  ✅  Best Model    : {best_name}")
print(f"  ✅  Best R² Score : {results[best_name]['R2']:.4f}")

# Cross-validation — tests the model on 5 different splits to check consistency
cv = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"  ✅  Cross-Val R²  : {cv.mean():.4f} ± {cv.std():.4f}")

# ─────────────────────────────────────────────
# STEP 5 — VISUALISATIONS (10 panels)
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#2D6A9F', '#E07B39', '#3A9E64', '#C0392B', '#8E44AD']

fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('#F4F6F8')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── Panel 1: Rating Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['Rating'], bins=35, color=PALETTE[0], edgecolor='white', alpha=0.87)
ax1.axvline(df['Rating'].mean(), color='red', lw=2, ls='--',
            label=f"Mean = {df['Rating'].mean():.2f}")
ax1.axvline(df['Rating'].median(), color='green', lw=2, ls='-.',
            label=f"Median = {df['Rating'].median():.2f}")
ax1.set_title('Rating Distribution', fontsize=13, fontweight='bold')
ax1.set_xlabel('IMDb Rating')
ax1.set_ylabel('Number of Movies')
ax1.legend()

# ── Panel 2: Top 10 Genres by Avg Rating
ax2 = fig.add_subplot(gs[0, 1])
top_genres = df['Primary_Genre'].value_counts().head(10).index
genre_df   = df[df['Primary_Genre'].isin(top_genres)]
genre_avg  = genre_df.groupby('Primary_Genre')['Rating'].mean().sort_values(ascending=False)
bars = ax2.bar(genre_avg.index, genre_avg.values,
               color=plt.cm.viridis(np.linspace(0.15, 0.85, len(genre_avg))),
               edgecolor='white')
ax2.set_title('Avg Rating by Genre (Top 10)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Genre')
ax2.set_ylabel('Average Rating')
ax2.tick_params(axis='x', rotation=45)
for b, v in zip(bars, genre_avg.values):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
             f'{v:.2f}', ha='center', va='bottom', fontsize=8)

# ── Panel 3: Actual vs Predicted
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_test, best_preds, alpha=0.3, s=12, color=PALETTE[2])
lims = [max(1, min(y_test.min(), best_preds.min())-0.3),
        min(10, max(y_test.max(), best_preds.max())+0.3)]
ax3.plot(lims, lims, 'r--', lw=1.5, label='Perfect Prediction')
ax3.set_title(f'Actual vs Predicted\n({best_name})', fontsize=13, fontweight='bold')
ax3.set_xlabel('Actual Rating')
ax3.set_ylabel('Predicted Rating')
ax3.set_xlim(lims); ax3.set_ylim(lims)
ax3.legend(fontsize=9)
ax3.text(0.05, 0.92, f"R²={results[best_name]['R2']:.3f}",
         transform=ax3.transAxes, fontsize=10, color='navy', fontweight='bold')

# ── Panel 4: Residuals Plot
ax4 = fig.add_subplot(gs[1, 0])
residuals = y_test - best_preds
ax4.scatter(best_preds, residuals, alpha=0.3, s=10, color=PALETTE[3])
ax4.axhline(0, color='black', lw=1.3, ls='--')
ax4.set_title('Residuals vs Predicted\n(should be randomly scattered around 0)',
              fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted Rating')
ax4.set_ylabel('Residual (Actual - Predicted)')

# ── Panel 5: Feature Importance (Random Forest)
ax5 = fig.add_subplot(gs[1, 1])
rf_model = models['Random Forest']
fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
fi.plot(kind='barh', ax=ax5, color=PALETTE[0], edgecolor='white')
ax5.set_title('Feature Importance\n(Random Forest)', fontsize=13, fontweight='bold')
ax5.set_xlabel('Importance Score')

# ── Panel 6: Model Comparison — R²
ax6 = fig.add_subplot(gs[1, 2])
m_names = list(results.keys())
r2_vals = [results[m]['R2'] for m in m_names]
x_pos   = np.arange(len(m_names))
bars6   = ax6.bar(x_pos, r2_vals, color=PALETTE[:len(m_names)], edgecolor='white')
ax6.set_title('Model Comparison — R²\n(higher is better)', fontsize=13, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([m.replace(' ', '\n') for m in m_names], fontsize=8)
ax6.set_ylabel('R² Score')
ax6.set_ylim(0, 1.05)
for b, v in zip(bars6, r2_vals):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── Panel 7: Model Comparison — RMSE
ax7 = fig.add_subplot(gs[2, 0])
rmse_vals = [results[m]['RMSE'] for m in m_names]
bars7 = ax7.bar(x_pos, rmse_vals, color=PALETTE[:len(m_names)], edgecolor='white', alpha=0.85)
ax7.set_title('Model Comparison — RMSE\n(lower is better)', fontsize=13, fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels([m.replace(' ', '\n') for m in m_names], fontsize=8)
ax7.set_ylabel('RMSE')
for b, v in zip(bars7, rmse_vals):
    ax7.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
             f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── Panel 8: Top 10 Directors by Avg Rating (min 5 movies)
ax8 = fig.add_subplot(gs[2, 1])
dir_stats = df.groupby('Director')['Rating'].agg(['mean', 'count'])
dir_stats = dir_stats[dir_stats['count'] >= 5].sort_values('mean', ascending=False).head(10)
ax8.barh(dir_stats.index[::-1], dir_stats['mean'][::-1],
         color=plt.cm.plasma(np.linspace(0.2, 0.8, 10)), edgecolor='white')
ax8.set_title('Top 10 Directors\n(min 5 movies, by Avg Rating)', fontsize=13, fontweight='bold')
ax8.set_xlabel('Average IMDb Rating')

# ── Panel 9: Votes vs Rating with trend line
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(np.log1p(df['Votes']), df['Rating'], alpha=0.2, s=8, color=PALETTE[4])
ax9.set_title('Log(Votes) vs Rating\n(more votes = higher rating?)', fontsize=13, fontweight='bold')
ax9.set_xlabel('log(Number of Votes)')
ax9.set_ylabel('IMDb Rating')
z  = np.polyfit(np.log1p(df['Votes'].dropna()), df.loc[df['Votes'].notna(), 'Rating'], 1)
px = np.linspace(df['Log_Votes'].min(), df['Log_Votes'].max(), 100)
ax9.plot(px, np.poly1d(z)(px), 'r-', lw=2, label='Trend Line')
ax9.legend()

# ── Panel 10: Correlation Heatmap (NEW)
ax10 = fig.add_subplot(gs[3, 0])
corr_cols = ['Duration', 'Log_Votes', 'Movie_Age', 'Dir_AvgRating', 'Act_AvgRating', 'Rating']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax10, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax10.set_title('Correlation Heatmap\n(how features relate to Rating)', fontsize=13, fontweight='bold')
ax10.tick_params(axis='x', rotation=30)

# ── Panel 11: Genre Movie Count (NEW)
ax11 = fig.add_subplot(gs[3, 1])
genre_counts = df['Primary_Genre'].value_counts().head(10)
ax11.barh(genre_counts.index[::-1], genre_counts.values[::-1],
          color=plt.cm.Set2(np.linspace(0, 1, 10)), edgecolor='white')
ax11.set_title('Top 10 Genres by Movie Count', fontsize=13, fontweight='bold')
ax11.set_xlabel('Number of Movies')

# ── Panel 12: Rating by Decade (NEW)
ax12 = fig.add_subplot(gs[3, 2])
df['Decade'] = (df['Year'] // 10 * 10).astype(int).astype(str) + 's'
dec_order = sorted(df['Decade'].unique())
dec_data  = [df.loc[df['Decade'] == d, 'Rating'].values for d in dec_order]
bp = ax12.boxplot(dec_data, patch_artist=True,
                  medianprops=dict(color='yellow', lw=2.5))
cols = plt.cm.coolwarm(np.linspace(0, 1, len(dec_order)))
for patch, c in zip(bp['boxes'], cols):
    patch.set_facecolor(c)
ax12.set_xticklabels(dec_order, rotation=45, fontsize=8)
ax12.set_title('Rating Distribution by Decade', fontsize=13, fontweight='bold')
ax12.set_xlabel('Decade')
ax12.set_ylabel('IMDb Rating')

fig.suptitle('🎬  IMDb India Movie Rating Prediction — Complete Analysis',
             fontsize=20, fontweight='bold', y=1.005, color='#1a1a2e')

plt.savefig('movie_rating_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='#F4F6F8')
plt.close()
print("\n  📊  Visualisation saved as movie_rating_analysis.png")

# ─────────────────────────────────────────────
# STEP 6 — SAMPLE PREDICTIONS
# Test the model on famous Bollywood movies
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SAMPLE PREDICTIONS — POPULAR BOLLYWOOD MOVIES")
print("=" * 65)

sample_movies = [
    {'Name': 'Dilwale Dulhania Le Jayenge', 'Year': 1995, 'Duration': 189,
     'Primary_Genre': 'Romance', 'Director': 'Aditya Chopra',
     'Actor 1': 'Shah Rukh Khan', 'Actor 2': 'Kajol', 'Actor 3': 'Amrish Puri', 'Votes': 100000},
    {'Name': '3 Idiots', 'Year': 2009, 'Duration': 170,
     'Primary_Genre': 'Comedy', 'Director': 'Rajkumar Hirani',
     'Actor 1': 'Aamir Khan', 'Actor 2': 'Madhavan', 'Actor 3': 'Sharman Joshi', 'Votes': 350000},
    {'Name': 'Dangal', 'Year': 2016, 'Duration': 161,
     'Primary_Genre': 'Drama', 'Director': 'Nitesh Tiwari',
     'Actor 1': 'Aamir Khan', 'Actor 2': 'Fatima Sana Shaikh', 'Actor 3': 'Sanya Malhotra', 'Votes': 280000},
    {'Name': 'KGF Chapter 2', 'Year': 2022, 'Duration': 168,
     'Primary_Genre': 'Action', 'Director': 'Prashanth Neel',
     'Actor 1': 'Yash', 'Actor 2': 'Sanjay Dutt', 'Actor 3': 'Raveena Tandon', 'Votes': 200000},
    {'Name': 'RRR', 'Year': 2022, 'Duration': 187,
     'Primary_Genre': 'Action', 'Director': 'S. S. Rajamouli',
     'Actor 1': 'N. T. Rama Rao Jr.', 'Actor 2': 'Ram Charan', 'Actor 3': 'Alia Bhatt', 'Votes': 320000},
]

sdf = pd.DataFrame(sample_movies)

# Encode the sample movies using the same encoders from training
for col, le, orig in [
    ('Primary_Genre', le_genre, 'Primary_Genre'),
    ('Director',      le_dir,   'Director'),
    ('Actor 1',       le_a1,    'Actor 1'),
    ('Actor 2',       le_a2,    'Actor 2'),
    ('Actor 3',       le_a3,    'Actor 3'),
]:
    known = set(le.classes_)
    sdf[orig] = sdf[orig].apply(lambda x: x if x in known else 'Unknown')
    sdf[col + '_enc'] = le.transform(sdf[orig])

sdf['Log_Votes'] = np.log1p(sdf['Votes'])
sdf['Movie_Age'] = 2024 - sdf['Year']

# Use average ratings from training data
dir_map     = df.groupby('Director')['Rating'].mean().to_dict()
a1_map      = df.groupby('Actor 1')['Rating'].mean().to_dict()
a2_map      = df.groupby('Actor 2')['Rating'].mean().to_dict()
global_mean = df['Rating'].mean()

sdf['Dir_AvgRating'] = sdf['Director'].map(dir_map).fillna(global_mean)
sdf['Act_AvgRating'] = (
    sdf['Actor 1'].map(a1_map).fillna(global_mean) +
    sdf['Actor 2'].map(a2_map).fillna(global_mean)
) / 2

sdf = sdf.rename(columns={
    'Primary_Genre_enc': 'Genre_enc',
    'Director_enc':      'Director_enc',
    'Actor 1_enc':       'Actor1_enc',
    'Actor 2_enc':       'Actor2_enc',
    'Actor 3_enc':       'Actor3_enc',
})

preds_s = best_model.predict(sdf[FEATURES])

print(f"\n  {'Movie':<35} {'Actual':>8} {'Predicted':>10}")
print("  " + "-" * 55)
for i, row in sdf.iterrows():
    actual     = df[df['Name'].str.strip() == row['Name']]['Rating'].values
    actual_str = f"{actual[0]:.1f}" if len(actual) > 0 else "N/A"
    print(f"  {row['Name']:<35} {actual_str:>8} {preds_s[i]:>10.2f}")

print("\n✅  Project Complete! Check movie_rating_analysis.png for the charts.")
