"""
Run this script ONCE in your project folder to regenerate pipe.pkl, team.pkl, city.pkl.
Usage:  python retrain.py
Requires: pandas, scikit-learn (whatever version you have installed)

NOTE: Uses IPL 2008-2025 dataset from Kaggle:
https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025
"""

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Load raw data ─────────────────────────────────────────────────────────────
match_df    = pd.read_csv('matches.csv')
delivery_df = pd.read_csv('deliveries.csv')

# ── Drop unneeded match columns ───────────────────────────────────────────────
drop_cols = ['toss_winner','toss_decision','player_of_match','umpire1','umpire2','umpire3']
match_df.drop(columns=[c for c in drop_cols if c in match_df.columns], inplace=True)

# ── First-innings total per match ─────────────────────────────────────────────
total_score_df = (delivery_df
                  .groupby(['match_id','inning'])['total_runs']
                  .sum().reset_index())
total_score_df = total_score_df[total_score_df['inning'] == 1]
match_df = match_df.merge(total_score_df[['match_id','total_runs']],
                          left_on='id', right_on='match_id')

# ── ONLY the 10 current active IPL franchises (NO renaming, NO merging) ───────
# Each team is treated as its own independent franchise.
# Defunct teams (Deccan Chargers, Delhi Daredevils, etc.) are simply excluded.
teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',            # formerly Kings XI Punjab — same franchise, rebranded name only
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
]

# ── Filter matches to only those where BOTH teams are current active teams ────
match_df = match_df[match_df['team1'].isin(teams) & match_df['team2'].isin(teams)]

# ── Remove DL applied matches ─────────────────────────────────────────────────
dl_col = 'dl_applied' if 'dl_applied' in match_df.columns else None
if dl_col:
    match_df = match_df[match_df[dl_col] == 0]

match_df = match_df[['match_id','city','winner','total_runs']]

print(f"✅ Matches after filtering: {len(match_df)}")

# ── Second innings deliveries only ───────────────────────────────────────────
del2 = delivery_df[delivery_df['inning'] == 2].copy()
del2 = match_df.merge(del2, on='match_id')
del2.rename(columns={'total_runs_x':'total_runs','total_runs_y':'Ball_score'}, inplace=True)

# ── Keep only deliveries where both teams are current active teams ─────────────
del2 = del2[del2['batting_team'].isin(teams) & del2['bowling_team'].isin(teams)]

# ── Feature engineering ───────────────────────────────────────────────────────
del2['Score']           = del2.groupby('match_id')['Ball_score'].cumsum()
del2['target_left']     = (del2['total_runs'] + 1) - del2['Score']
del2['Remaining Balls'] = 120 - ((del2['over'] - 1) * 6 + del2['ball'])

del2['player_dismissed'] = del2['player_dismissed'].fillna('0')
del2['player_dismissed'] = (del2['player_dismissed']
                             .apply(lambda x: x if x == '0' else '1')
                             .astype('int64'))
del2['Wickets'] = 10 - del2.groupby('match_id')['player_dismissed'].cumsum()

del2['crr'] = del2['Score'] * 6 / (120 - del2['Remaining Balls'])
del2['rrr'] = del2['target_left'] * 6 / del2['Remaining Balls']
del2['result'] = (del2['batting_team'] == del2['winner']).astype(int)

# ── Final model dataframe ─────────────────────────────────────────────────────
cols = ['batting_team','bowling_team','city','Score','Wickets',
        'Remaining Balls','target_left','crr','rrr','result']
Model_df = del2[cols].dropna()
Model_df = Model_df[Model_df['Remaining Balls'] != 0]
Model_df = Model_df.sample(Model_df.shape[0], random_state=42)

print(f"✅ Training rows: {len(Model_df)}")
print(f"✅ Teams in data: {sorted(set(Model_df['batting_team']) | set(Model_df['bowling_team']))}")

# ── Train / test split ────────────────────────────────────────────────────────
X = Model_df.iloc[:, :-1]
Y = Model_df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# ── Pipeline ──────────────────────────────────────────────────────────────────
trf = ColumnTransformer([
    ('trf', OneHotEncoder(drop='first', handle_unknown='ignore'),
     ['batting_team','bowling_team','city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train, Y_train)
acc = accuracy_score(Y_test, pipe.predict(X_test))
print(f"✅ Model trained — accuracy: {acc:.4f}")

# ── Save pickles ──────────────────────────────────────────────────────────────
cities = sorted(Model_df['city'].unique().tolist())

pkl.dump(pipe,   open('pipe.pkl',  'wb'))
pkl.dump(teams,  open('team.pkl',  'wb'))
pkl.dump(cities, open('city.pkl',  'wb'))
print("✅ pipe.pkl, team.pkl, city.pkl saved successfully.")