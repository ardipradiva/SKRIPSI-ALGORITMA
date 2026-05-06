import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

print("Starting training process...")

# 1. Load Data
df = pd.read_csv("cardio_train.csv", sep=";")

# 2. Feature Engineering & Cleaning
print("Cleaning data and engineering features...")
df['age'] = round(df['age']/365.25, 2)
df.insert(3, "female", (df['gender']==1).astype(int))
df.insert(4, 'male', (df['gender']==2).astype(int))
df.drop(['gender', 'id'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.insert(5, 'bmi', round((df['weight']/(df['height']/100)**2), 2))
df.drop(df.query('bmi >60 or bmi <15').index, axis=0, inplace=True)
df.drop(df.query('ap_hi >220 or ap_lo >180 or ap_hi<40 or ap_lo<40').index, axis=0, inplace=True)

# 3. Define Features (X) and Target (y)
print("Defining features and target...")
X = df.drop(['cardio'], axis=1)
y = df['cardio']

print(f"Total rows for training: {X.shape[0]}")

# 4. Train Models
print("Training Models...")

models = {
    'XGBoost': XGBClassifier(verbosity=0, seed=0, n_estimators=150, gamma=0.24, max_depth=4, learning_rate=0.13, reg_lambda=50.0, scale_pos_weight=1),
    'Random Forest': RandomForestClassifier(n_estimators=51, max_depth=10, random_state=0),
    'KNN': KNeighborsClassifier(weights='uniform', n_neighbors=300, leaf_size=1, algorithm='ball_tree'),
    'SVM': SVC(C=100, gamma=0.00001, kernel="rbf", random_state=42, probability=True)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X, y)
    
    filename = f"{name.lower().replace(' ', '_')}_cvd_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name} to {filename}")

print("All models successfully trained and saved!")
