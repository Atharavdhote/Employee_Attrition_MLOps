
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Basic information
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:", df.isnull().sum().sum())
print("Duplicate Records:", df.duplicated().sum())

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())

# Create output folder
import os
os.makedirs("outputs", exist_ok=True)

# 1. Attrition Distribution
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.savefig("outputs/attrition_distribution.png")
plt.close()

# 2. Overtime vs Attrition
sns.countplot(data=df, x="OverTime", hue="Attrition")
plt.title("Overtime vs Attrition")
plt.savefig("outputs/overtime_attrition.png")
plt.close()

# 3. Job Satisfaction vs Attrition
sns.countplot(data=df, x="JobSatisfaction", hue="Attrition")
plt.title("Job Satisfaction vs Attrition")
plt.savefig("outputs/job_satisfaction.png")
plt.close()

# 4. Monthly Income vs Attrition
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
plt.title("Monthly Income vs Attrition")
plt.savefig("outputs/monthly_income.png")
plt.close()

# 5. Years at Company vs Attrition
sns.boxplot(data=df, x="Attrition", y="YearsAtCompany")
plt.title("Years at Company vs Attrition")
plt.savefig("outputs/years_at_company.png")
plt.close()

# 6. Correlation Heatmap
numeric = df.select_dtypes(include="number")

plt.figure(figsize=(14, 9))
sns.heatmap(numeric.corr(), cmap="coolwarm")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("\nEDA completed successfully!")
print("Charts saved inside outputs folder.")
