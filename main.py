# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 AI-Powered Student Performance Predictor")

# Load the trained model
model = pickle.load(open("student_performance_model.pkl", "rb"))

# Load data sample for SHAP
data_sample = pd.read_csv("student_data.csv").drop("final_grade", axis=1)

# Sidebar inputs
st.sidebar.header("Student Info")
study_time = st.sidebar.slider("Study Time (hours/week)", 0, 40, 10)
absences = st.sidebar.slider("Number of Absences", 0, 50, 5)
past_score = st.sidebar.slider("Past Exam Score (0-100)", 0, 100, 70)
health = st.sidebar.selectbox("Health Status", [1, 2, 3, 4, 5], index=2)

# Predict button
if st.sidebar.button("Predict Performance"):
    input_data = pd.DataFrame([[study_time, absences, past_score, health]], 
                               columns=["study_time", "absences", "past_score", "health"])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Final Grade: {prediction:.2f}")

    # SHAP Explainability
    explainer = shap.Explainer(model.predict, data_sample)
    shap_values = explainer(input_data)

    st.subheader("Feature Contribution")
    shap.plots.waterfall(shap_values[0], max_display=4)


# train_model.ipynb
"""
Train a regression model to predict student performance (final grade).
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_csv("student_data.csv")
X = df.drop("final_grade", axis=1)
y = df["final_grade"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, preds):.2f}")

# Save model
with open("student_performance_model.pkl", "wb") as f:
    pickle.dump(model, f)


# student_data.csv (Sample Data)
"""
study_time,absences,past_score,health,final_grade
10,5,70,3,72
15,3,85,4,88
7,8,60,2,65
20,1,90,5,95
5,10,55,3,58
"""


# requirements.txt
streamlit
scikit-learn
pandas
matplotlib
seaborn
shap
pickle5
