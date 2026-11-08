# Task 2: Exploratory Data Analysis

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
# Check missing values

print("\nMissing Values:")
print(df.isnull().sum())
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize missing values

plt.figure(figsize=(12, 6))

sns.heatmap(df.isnull(), cbar=False)

plt.title("Missing Values Heatmap")
plt.savefig("missing_values_heatmap.png", dpi=300)

plt.show()
# Distribution of numerical features

df.hist(figsize=(12, 10))

plt.suptitle("Distribution of Numerical Features")

plt.tight_layout()
plt.savefig("histogram.png", dpi=300)

plt.show()
# Correlation analysis

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

# Correlation heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(correlation, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=300)

plt.show()
# Survival analysis by gender

print("\nSurvival by Gender:")

print(df.groupby("sex")["survived"].mean())
# Survival analysis by passenger class

print("\nSurvival by Passenger Class:")

print(df.groupby("pclass")["survived"].mean())
# Survival by age group

# Survival analysis by age group

df["age_group"] = pd.NA

df.loc[df["age"].notna(), "age_group"] = df["age"].apply(
    lambda x: "Child" if x < 18 else "Adult"
)

print("\nSurvival by Age Group:")
print(df.groupby("age_group")["survived"].mean())