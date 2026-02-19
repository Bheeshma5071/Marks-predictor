"""
train_model.py
---------------
Trains a Linear Regression model to predict student marks
based on study hours and saves the trained model as model.pkl
"""

# ================================
# IMPORT LIBRARIES
# ================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib


# ================================
# LOAD DATA
# ================================

data = pd.read_csv("student_info.csv")


# ================================
# HANDLE MISSING VALUES
# ================================

data = data.fillna(data.mean())


# ================================
# PREPARE FEATURES & TARGET
# ================================

X = data[["Hours"]].values
y = data[["Score"]].values


# ================================
# TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================================
# TRAIN MODEL
# ================================

model = LinearRegression()
model.fit(X_train, y_train)


# ================================
# EVALUATE MODEL
# ================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2_score = model.score(X_test, y_test)

print("Model Training Complete")
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2_score, 2))


# ================================
# SAVE MODEL
# ================================

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
