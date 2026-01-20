import pandas as pd
import os

# defining the base folder where my files are kept
BASE_DIR = '/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning'

# loading the demographic data from the csv file
df = pd.read_csv(os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv"))
print(df.duplicated().sum())
print(df["district"].unique())
print(len(df["district"].unique()))
print(df["state"].unique())
print(len(df["state"].unique()))
print(df.isna().sum())

# fixing the column names to be standard
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# converting the date column to proper datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["month"] = df["date"].dt.month_name()
df["year"] = df["date"].dt.year
df["month_num"] = df["date"].dt.month

# making sure these columns are numbers
numeric_cols = ["demo_age_5_17", "demo_age_17_"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# adding up the two age groups to get total updates
df["total_updates"] = df["demo_age_5_17"] + df["demo_age_17_"]

# grouping by month to see the trends
monthly_summary = (
    df.groupby(["year", "month", "month_num"])["total_updates"]
    .sum()
    .reset_index()
    .sort_values(["year", "month_num"])
)

# showing the monthly summary
print("\n📊 MONTH-WISE TOTAL DEMOGRAPHIC UPDATES\n")

for _, row in monthly_summary.iterrows():
    print(
        f"In {row['month']} {int(row['year'])}, "
        f"total updates = {int(row['total_updates']):,}"
    )
