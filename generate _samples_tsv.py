import numpy as np
import pandas as pd

# --------------------------------------------------------
# 🎯 Generate Sample Cancer Recurrence Prediction Datasets
# --------------------------------------------------------
# This script creates two TSV files:
# 1. sample_dataset.tsv — for model training and visualization
# 2. sample_patient.tsv — for single-patient prediction testing
# --------------------------------------------------------

np.random.seed(42)

# Number of samples (patients) and genes
n_samples = 100
n_genes = 20

# Create synthetic gene names
genes = [f"Gene_{i+1}" for i in range(n_genes)]

# -----------------------------
# Create Main Training Dataset
# -----------------------------
df = pd.DataFrame(np.random.randn(n_samples, n_genes), columns=genes)

# Add time-to-event and survival status
df["time"] = np.random.exponential(scale=400, size=n_samples)   # survival time
df["status"] = np.random.binomial(1, 0.35, size=n_samples)      # 1 = event occurred
df["recurrence_risk"] = np.random.choice(["High", "Low"], size=n_samples)

# Save as TSV (Tab-Separated Values)
df.to_csv("sample_dataset.tsv", sep="\t", index=False)

# -----------------------------
# Create Single Patient Dataset
# -----------------------------
single = pd.DataFrame(np.random.randn(1, n_genes), columns=genes)
single.to_csv("sample_patient.tsv", sep="\t", index=False)

print("✅ Sample TSV files created successfully:")
print(" - sample_dataset.tsv")
print(" - sample_patient.tsv")
