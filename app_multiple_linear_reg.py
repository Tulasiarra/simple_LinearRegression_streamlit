import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(page_title="Multiple Linear Regression", layout="centered")

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------
# Title Section
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression Model</h1>
    <p>
        Predict <b>Tip Amount</b> using <b>Total Bill</b> and <b>Party Size</b>
        with a machine learning <b>Multiple Linear Regression</b> model.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prepare Data (MULTIPLE FEATURES)
# --------------------------------------------------
X = df[["total_bill", "size"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
model = LinearRegression()
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
# Visualization
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount (Colored by Size)")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df["total_bill"],
    df["tip"],
    c=df["size"],
    cmap="viridis",
    alpha=0.7
)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
plt.colorbar(scatter, label="Party Size")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Performance
# --------------------------------------------------
st.markdown('<div class="card"', unsafe_allow_html=True)
st.subheader("Model Performance Metrics")

c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R2 Score", f"{r2:.3f}")
c4.metric("Adj R2 Score", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Coefficients
# --------------------------------------------------
st.markdown(f"""
<div class="card">
<h3>Model Intercept and Coefficients</h3>
<p>
<b>Total Bill Coefficient:</b> {model.coef_[0]:.3f}<br>
<b>Party Size Coefficient:</b> {model.coef_[1]:.3f}<br>
<b>Intercept:</b> {model.intercept_:.3f}
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown('<div class="card"', unsafe_allow_html=True)
st.subheader("Predict Tip")

bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)
size = st.slider(
    "Party Size",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)


input_data = scaler.transform([[bill, size]])
tip = model.predict(input_data)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
