import pandas as pd
import matplotlib.pyplot as plt
import os

# set up the base directory for files
BASE_DIR = '/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning'

# -------------------------------------------------
# loading the dataset
# -------------------------------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv"))
print(df.duplicated().sum())
print(df["district"].unique())
print(len(df["district"].unique()))
print(df["state"].unique())
print(len(df["state"].unique()))
print(df.isna().sum())

# changing date column to datetime object
df["date"] = pd.to_datetime(df["date"], errors="coerce")


# -------------------------------------------------
# adding a month column for grouping
# -------------------------------------------------
df["month"] = df["date"].dt.to_period("M")

# -------------------------------------------------
# analyzing the data month by month
# -------------------------------------------------
monthly = df.groupby("month")[["demo_age_5_17", "demo_age_17_"]].sum()
monthly["total_updates"] = monthly["demo_age_5_17"] + monthly["demo_age_17_"]

# finding which months had the most and least updates
max_month = monthly["total_updates"].idxmax()
min_month = monthly["total_updates"].idxmin()

max_value = monthly.loc[max_month, "total_updates"]
min_value = monthly.loc[min_month, "total_updates"]

# plotting the monthly updates graph
plt.figure(figsize=(10, 5))
plt.plot(monthly.index.astype(str), monthly["total_updates"], marker="o")

# marking the peak and lowest points on the chart
plt.scatter(str(max_month), max_value)
plt.scatter(str(min_month), min_value)

plt.annotate(
    f"Highest\n{max_month}\n{max_value:,}",
    (str(max_month), max_value),
    xytext=(10, 10),
    textcoords="offset points",
)

plt.annotate(
    f"Lowest\n{min_month}\n{min_value:,}",
    (str(min_month), min_value),
    xytext=(10, -15),
    textcoords="offset points",
)

plt.xlabel("Month")
plt.ylabel("Total Demographic Updates")
plt.title("Monthly Demographic Updates (Highest & Lowest Highlighted)")
plt.xticks(rotation=45)
plt.tight_layout()
IMAGE_DIR = os.path.join(BASE_DIR, "image")
os.makedirs(IMAGE_DIR, exist_ok=True)
plt.savefig(os.path.join(IMAGE_DIR, "highlight_analysis_updates.png"))
plt.close()

# -------------------------------------------------
# calculating the difference from one month to next
# -------------------------------------------------
monthly["monthly_change"] = monthly["total_updates"].diff()

plt.figure(figsize=(10, 5))
plt.plot(monthly.index.astype(str), monthly["monthly_change"], marker="o")
plt.axhline(0)

plt.xlabel("Month")
plt.ylabel("Change in Updates")
plt.title("Month-to-Month Change in Demographic Updates")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "highlight_analysis_monthly_change.png"))
plt.close()

# printing the key findings
print("📌 Month with highest updates:", max_month)
print("📌 Month with lowest updates:", min_month)
print("📌 Month with biggest drop:", monthly["monthly_change"].idxmin())

# -------------------------------------------------
# breaking down the updates by age group
# -------------------------------------------------
age_totals = {
    "Age 5–17": df["demo_age_5_17"].sum(),
    "Age 17+": df["demo_age_17_"].sum(),
}

age_df = pd.DataFrame(list(age_totals.items()), columns=["Age Group", "Total Updates"])

# figuring out the percentage share for each group
total_updates_sum = age_df["Total Updates"].sum()
age_df["Percentage"] = (age_df["Total Updates"] / total_updates_sum) * 100

# plotting percentages for age groups
plt.figure(figsize=(6, 5))
bars = plt.bar(age_df["Age Group"], age_df["Total Updates"])

for bar, pct in zip(bars, age_df["Percentage"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
    )

plt.xlabel("Age Group")
plt.ylabel("Total Demographic Updates")
plt.title("Demographic Updates by Age Group (Percentage Share)")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "highlight_analysis_age_share.png"))
plt.close()

# printing the key findings
print(
    "📌 Age group with most updates:",
    age_df.loc[age_df["Total Updates"].idxmax(), "Age Group"],
)

print(
    "📌 Age group with least updates:",
    age_df.loc[age_df["Total Updates"].idxmin(), "Age Group"],
)
