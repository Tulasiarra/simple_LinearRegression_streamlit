import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(page_title="Ridge & Lasso Regression", layout="centered")

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h1>Ridge & Lasso Regression</h1>
    <p>
        Heart Disease Prediction using <b>Ridge</b> and <b>Lasso</b> Regression
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Dataset (YOUR CSV)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Encode Categorical Columns
# --------------------------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

# --------------------------------------------------
# Features & Target
# --------------------------------------------------
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Ridge Regression", "Lasso Regression"]
)

alpha = st.sidebar.slider("Regularization Strength (α)", 0.01, 10.0, 1.0)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
if model_type == "Ridge Regression":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

# --------------------------------------------------
# Performance
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.3f}")
c2.metric("RMSE", f"{rmse:.3f}")

c3, c4 = st.columns(2)
c3.metric("R2 Score", f"{r2:.3f}")
c4.metric("Adj R2", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Coefficients
# --------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Coefficients")
st.dataframe(coef_df.head(10))
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prediction (Numeric Inputs Only)
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Heart Disease Risk")

age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 45)
resting_bp = st.slider("Resting BP", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

input_df = pd.DataFrame({
    "Age": [age],
    "RestingBP": [resting_bp],
    "Cholesterol": [chol],
    "MaxHR": [max_hr],
    "Oldpeak": [oldpeak]
})

input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Heart Disease Score: {prediction:.3f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
