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
st.set_page_config(page_title="Linear Regression", layout="centered")

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
    <h1>Linear Regression Model</h1>
    <p>
        Predict <b>Tip Amount</b> from <b>Total Bill</b> using a
        machine learning <b>Linear Regression</b> model.
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
# Prepare Data
# --------------------------------------------------
X = df[["total_bill"]]
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
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)



# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount")

# Create smooth regression line
X_sorted = np.sort(X.values, axis=0)
X_sorted_scaled = scaler.transform(X_sorted)
y_line = model.predict(X_sorted_scaled)

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6, label="Actual Data")
ax.plot(X_sorted, y_line, color="red", linewidth=2, label="Regression Line")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
ax.legend()

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)


# performance

st.markdown('<div class="card"',unsafe_allow_html=True)
st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("r2_score",f"{r2: .3f}")
c4.metric("adj R2",f"{adj_r2: .3f}")
st.markdown('<div class="card"',unsafe_allow_html=True)



# m & c

st.markdown(f"""
<div class="card">
<h3>Model Intercept and coefficient</h3>
<p> <b> co-efficient: </b> {model.coef_[0]:.3f}<br>
<b> Intercept :<b> {model.intercept_:.3f}</p>
</div>
""",unsafe_allow_html=True)


# Prediction

st.markdown('<div class="card"',unsafe_allow_html=True)
st.subheader("Predict Tip")
bill=st.slider("Total bill",float(df.total_bill.min()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
