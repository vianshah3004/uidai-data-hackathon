import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# defining the base directory
BASE_DIR = '/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning'

# loading the demographic dataset
df = pd.read_csv(os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv"))
print(df.duplicated().sum())
print(df["district"].unique())
print(len(df["district"].unique()))
print(df["state"].unique())
print(len(df["state"].unique()))
print(df.isna().sum())
df["date"] = pd.to_datetime(df["date"])
df["month_year"] = df["date"].dt.to_period("M")

# organizing data by month
summary = (
    df.groupby("month_year")
    .agg({"demo_age_5_17": "sum", "demo_age_17_": "sum"})
    .reset_index()
)

summary["total"] = summary["demo_age_5_17"] + summary["demo_age_17_"]
summary["month_str"] = summary["month_year"].astype(str)

# calculating the percentage split between kids and adults
summary["Kids_Share"] = (summary["demo_age_5_17"] / summary["total"]) * 100
summary["Adults_Share"] = (summary["demo_age_17_"] / summary["total"]) * 100

# plotting the 100% stacked area chart
plt.figure(figsize=(14, 8))
plt.stackplot(
    summary["month_str"],
    summary["Adults_Share"],
    summary["Kids_Share"],
    labels=["Adults (17+)", "Kids (5-17)"],
    colors=["#66b3ff", "#ff9999"],
    alpha=0.8,
)

# styling the chart
plt.ylim(0, 100)
plt.xlim(summary["month_str"].iloc[0], summary["month_str"].iloc[-1])
plt.title(
    "Monthly Demographic Share (100% Stacked Area)", fontsize=16, fontweight="bold"
)
plt.xlabel("Timeline", fontsize=12)
plt.ylabel("Percentage Share (%)", fontsize=12)
plt.legend(loc="upper right", frameon=True)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
IMAGE_DIR = os.path.join(BASE_DIR, "image")
os.makedirs(IMAGE_DIR, exist_ok=True)
plt.savefig(os.path.join(IMAGE_DIR, "demographic_share_stacked.png"))
plt.close()
