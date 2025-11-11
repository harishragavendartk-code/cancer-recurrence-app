import numpy as np
import pandas as pd

# -------------------------------
# Generate Sample Cancer Recurrence Prediction Datasets
# -------------------------------

# This script creates two TSV files:
# 1. sample_dataset.tsv   → for model training and visualization
# 2. sample_patient.tsv   → for single patient prediction testing

np.random.seed(42)  # For reproducibility

# -------------------------------
# Configuration
# -------------------------------
n_samples = 100   # number of patients
n_genes = 20      # number of genes (features)

# Create synthetic gene names
genes = [f"Gene_{i+1}" for i in range(n_genes)]

# -------------------------------
# Create Main Training Dataset
# -------------------------------
df = pd.DataFrame(np.random.randn(n_samples, n_genes), columns=genes)

# Add survival time and status
df["time"] = np.random.exponential(scale=400, size=n_samples)
df["status"] = np.random.binomial(1, 0.35, size=n_samples)

# Add recurrence risk labels (binary classification)
df["recurrence_risk"] = np.random.choice(["High", "Low"], size=n_samples)

# Save as TSV (tab-separated)
df.to_csv("sample_dataset.tsv", sep="\t", index=False)

# -------------------------------
# Create Single Patient Sample
# -------------------------------
single = pd.DataFrame(np.random.randn(1, n_genes), columns=genes)
single["time"] = np.random.exponential(scale=400, size=1)
single["status"] = np.random.binomial(1, 0.35, size=1)
single["recurrence_risk"] = np.random.choice(["High", "Low"], size=1)

# Save single-patient file as TSV
single.to_csv("sample_patient.tsv", sep="\t", index=False)

print("✅ Sample TSV datasets generated successfully:")
print(" - sample_dataset.tsv (training dataset)")
print(" - sample_patient.tsv (single patient)")
