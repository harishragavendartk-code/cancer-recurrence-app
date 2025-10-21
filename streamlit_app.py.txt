import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cancer Prediction Web App", layout="wide")
st.title("🧬 Cancer Prediction & Visualization Dashboard")

st.markdown("""
This app visualizes *RNA Expression Data* to explore:
- 🧠 Gene Expression Heatmap  
- 📈 Kaplan–Meier Survival Curve  
- ⚙ LASSO–Cox Coefficient Visualization  

Select *a dataset* below to get started 👇
""")

# =========================================================
# Dataset Selection
# =========================================================
choice = st.radio(
    "Choose Data Source:",
    ("Use sample TCGA dataset", "Upload my own dataset (.csv)")
)

# =========================================================
# Load Sample CSV Files
# =========================================================
def load_sample_dataset():
    # Read expression and patient data from CSVs
    expr_data = pd.read_csv("sample_datasets.csv")
    patient_data = pd.read_csv("sample_patients.csv")
    
    # Merge expression and survival info
    df = pd.concat([expr_data, patient_data[['time', 'status']]], axis=1)
    return df

# =========================================================
# Load chosen dataset
# =========================================================
if choice == "Use sample TCGA dataset":
    df = load_sample_dataset()
    st.success("✅ Loaded sample dataset from CSV files.")
else:
    uploaded_file = st.file_uploader("📁 Upload RNA Expression CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.info("👆 Please upload your CSV file to continue.")
        st.stop()

st.write("### Preview of Data:")
st.dataframe(df.head())

# =========================================================
# Separate expression and survival data
# =========================================================
expr_data = df.select_dtypes(include=[np.number]).copy()
if 'time' not in df.columns or 'status' not in df.columns:
    df['time'] = np.random.exponential(scale=365, size=len(df))
    df['status'] = np.random.binomial(1, 0.3, size=len(df))

# Compute risk score
risk_score = expr_data.drop(columns=['time', 'status'], errors='ignore').sum(axis=1)
median_score = np.median(risk_score)
risk_group = np.where(risk_score >= median_score, 'High', 'Low')

# =========================================================
# 1️⃣ Gene Expression Heatmap
# =========================================================
st.markdown("---")
st.subheader("1️⃣ Gene Expression Heatmap")

fig1, ax1 = plt.subplots(figsize=(8,6))
sns.heatmap(expr_data.drop(columns=['time', 'status'], errors='ignore').T,
            cmap='RdBu_r', center=0, ax=ax1, xticklabels=False,
            cbar_kws={'label': 'Expression Level'})
ax1.set_title("Gene Expression Heatmap")
ax1.set_xlabel("Patients")
ax1.set_ylabel("Genes")
st.pyplot(fig1)

# =========================================================
# 2️⃣ Kaplan-Meier Survival Curve
# =========================================================
st.markdown("---")
st.subheader("2️⃣ Kaplan–Meier Survival Curve")

def km_curve(time, event):
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    n = len(time)
    at_risk = n - np.arange(n)
    survival = np.cumprod(1 - event / at_risk)
    return time, survival

mask_high = risk_group == 'High'
mask_low = risk_group == 'Low'
time_high, surv_high = km_curve(df['time'][mask_high], df['status'][mask_high])
time_low, surv_low = km_curve(df['time'][mask_low], df['status'][mask_low])

fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.step(time_high, surv_high, where='post', label='High Risk', color='red')
ax2.step(time_low, surv_low, where='post', label='Low Risk', color='blue')
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Survival Probability")
ax2.set_title("Kaplan–Meier Curve: High vs Low Risk")
ax2.legend()
st.pyplot(fig2)

# =========================================================
# 3️⃣ LASSO–Cox Coefficients (Simulated)
# =========================================================
st.markdown("---")
st.subheader("3️⃣ LASSO–Cox Coefficient Visualization (Simulated)")

coef_values = np.random.uniform(-0.5, 0.5, size=expr_data.shape[1]-2)
coef_df = pd.DataFrame({
    'Gene': expr_data.drop(columns=['time', 'status'], errors='ignore').columns,
    'Coefficient': coef_values
}).sort_values(by='Coefficient', ascending=False)

fig3, ax3 = plt.subplots(figsize=(8,6))
sns.barplot(x='Coefficient', y='Gene', data=coef_df, palette='coolwarm', ax=ax3)
ax3.set_title("Simulated LASSO–Cox Coefficients")
st.pyplot(fig3)

st.success("✅ Analysis Complete!")
