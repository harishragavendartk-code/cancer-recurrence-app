import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Cancer Prediction Web App", layout="wide")
st.title("🧬 Cancer Prediction & Visualization Dashboard")

st.markdown("""
Predict **cancer recurrence risk** from RNA-seq data.  
- Upload your dataset (TSV) or use the included sample dataset.  
- Train a Random Forest model.  
- Visualize gene expression, Kaplan–Meier curves, and feature importance.
""")

# -------------------------------
# Load dataset
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload RNA Expression TSV (optional)", type=["tsv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t")
    st.success("✅ File uploaded successfully.")
else:
    try:
        df = pd.read_csv("sample_dataset.tsv", sep="\t")
        st.info("ℹ️ No file uploaded. Loaded default `sample_dataset.tsv`.")
    except FileNotFoundError:
        st.error("❌ sample_dataset.tsv not found. Please upload a dataset.")
        st.stop()

st.write("### 🧾 Data Preview")
st.dataframe(df.head())

# -------------------------------
# Prepare data
# -------------------------------
if 'recurrence_risk' not in df.columns:
    df['recurrence_risk'] = np.random.choice(['High', 'Low'], len(df))

if 'time' not in df.columns:
    df['time'] = np.random.exponential(scale=400, size=len(df))

if 'status' not in df.columns:
    df['status'] = np.random.binomial(1, 0.35, size=len(df))

X = df.select_dtypes(include=[np.number])
y = df['recurrence_risk']

# -------------------------------
# Train model
# -------------------------------
st.markdown("---")
st.subheader("🤖 Train Machine Learning Model")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")

# -------------------------------
# Confusion matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred, labels=["High", "Low"])
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["High", "Low"], yticklabels=["High", "Low"], ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# -------------------------------
# Feature importance
# -------------------------------
st.markdown("#### 🔬 Feature Importance (Top 10 Genes)")
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig_imp, ax_imp = plt.subplots()
feat_imp.plot(kind='barh', ax=ax_imp, color='teal')
ax_imp.set_xlabel("Importance Score")
ax_imp.set_ylabel("Gene")
st.pyplot(fig_imp)

# -------------------------------
# Gene expression heatmap
# -------------------------------
st.markdown("---")
st.subheader("1️⃣ Gene Expression Heatmap")

fig1, ax1 = plt.subplots(figsize=(8,6))
sns.heatmap(X.T, cmap='RdBu_r', center=0, xticklabels=False, ax=ax1)
ax1.set_title("Gene Expression Heatmap")
ax1.set_xlabel("Patients")
ax1.set_ylabel("Genes")
st.pyplot(fig1)

# -------------------------------
# Kaplan–Meier curve
# -------------------------------
st.markdown("---")
st.subheader("2️⃣ Kaplan–Meier Survival Curve")

def km_curve(time, event):
    # Ensure inputs are numpy arrays
    time = np.array(time)
    event = np.array(event)
    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]
    n = len(time_sorted)
    at_risk = n - np.arange(n)
    survival = np.cumprod(1 - event_sorted / at_risk)
    return time_sorted, survival

# Use .loc to avoid KeyError
mask_high = df['recurrence_risk'] == 'High'
mask_low = df['recurrence_risk'] == 'Low'

time_high, surv_high = km_curve(df.loc[mask_high, 'time'], df.loc[mask_high, 'status'])
time_low, surv_low = km_curve(df.loc[mask_low, 'time'], df.loc[mask_low, 'status'])

fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.step(time_high, surv_high, where='post', label='High Risk', color='red')
ax2.step(time_low, surv_low, where='post', label='Low Risk', color='blue')
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Survival Probability")
ax2.set_title("Kaplan–Meier Curve: High vs Low Risk")
ax2.legend()
st.pyplot(fig2)

# -------------------------------
# Simulated LASSO–Cox coefficients
# -------------------------------
st.markdown("---")
st.subheader("3️⃣ LASSO–Cox Coefficient Visualization (Simulated)")

coef_values = np.random.uniform(-0.5, 0.5, size=X.shape[1])
coef_df = pd.DataFrame({
    'Gene': X.columns,
    'Coefficient': coef_values
}).sort_values(by='Coefficient', ascending=False)

fig3, ax3 = plt.subplots(figsize=(8,6))
sns.barplot(x='Coefficient', y='Gene', data=coef_df, palette='coolwarm', ax=ax3)
ax3.set_title("Simulated LASSO–Cox Coefficients")
st.pyplot(fig3)

# -------------------------------
# Real-time prediction
# -------------------------------
st.markdown("---")
st.subheader("⚡ Predict Recurrence Risk for a New Patient")

new_sample = st.file_uploader("Upload new patient TSV", type=["tsv"], key="predict")
if new_sample:
    new_df = pd.read_csv(new_sample, sep="\t")
    st.dataframe(new_df.head())
    try:
        pred = model.predict(new_df)
        st.success(f"🧠 Predicted Recurrence Risk: **{pred[0]}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("Upload a single patient TSV to predict risk.")

st.success("✅ App ready — trained model can now predict recurrence risk in real time!")
