
import pandas as pd
import os

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Remove unnecessary columns
df.drop(columns=[
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
], inplace=True)

# Convert target column
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Check missing values
print("Missing values:", df.isnull().sum().sum())

# Save processed dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/employee_attrition_clean.csv",
    index=False
)

print("Preprocessing completed!")
print("Final dataset shape:", df.shape)
print(df.head())
