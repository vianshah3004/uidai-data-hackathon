import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# defining the base directory
BASE_DIR = '/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning'

# loading the demographic data
df = pd.read_csv(os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv"))
print(df.duplicated().sum())
print(df["district"].unique())
print(len(df["district"].unique()))
print(df["state"].unique())
print(len(df["state"].unique()))
print(df.isna().sum())
df["date"] = pd.to_datetime(df["date"])
df["total_updates"] = df["demo_age_5_17"] + df["demo_age_17_"]
df["month_year"] = df["date"].dt.to_period("M")

# grouping all updates by month
monthly_summary = (
    df.groupby("month_year")
    .agg({"total_updates": "sum", "demo_age_5_17": "sum", "demo_age_17_": "sum"})
    .reset_index()
)

monthly_summary = monthly_summary.sort_values("month_year")
monthly_summary["month_str"] = monthly_summary["month_year"].astype(str)
monthly_summary["growth_rate"] = monthly_summary["total_updates"].pct_change() * 100
monthly_summary["Kids %"] = (
    monthly_summary["demo_age_5_17"] / monthly_summary["total_updates"]
) * 100
monthly_summary["Adults %"] = (
    monthly_summary["demo_age_17_"] / monthly_summary["total_updates"]
) * 100

# finding the peak and lowest months
max_val = monthly_summary["total_updates"].max()
min_val = monthly_summary["total_updates"].min()
max_month = monthly_summary.loc[
    monthly_summary["total_updates"] == max_val, "month_str"
].values[0]
min_month = monthly_summary.loc[
    monthly_summary["total_updates"] == min_val, "month_str"
].values[0]

# plotting the total volume graph
sns.set_theme(style="whitegrid")
fig1, ax1 = plt.subplots(figsize=(14, 8))
# using a single color to keep it clean
bars1 = sns.barplot(
    data=monthly_summary, x="month_str", y="total_updates", color="skyblue", ax=ax1
)

ax1.set_title(
    "Monthly Update Volume: Growth, Highest & Lowest", fontsize=18, fontweight="bold"
)
ax1.set_xlabel("Month", fontsize=14)
ax1.set_ylabel("Total Count of Updates", fontsize=14)

# formatting numbers to be readable
ax1.ticklabel_format(style="plain", axis="y")
ticks = ax1.get_yticks()
ax1.set_yticklabels([f"{int(x):,}" for x in ticks])
ax1.set_ylim(0, max_val * 1.35)

for i, bar in enumerate(ax1.patches):
    month = monthly_summary["month_str"].iloc[i]
    val = monthly_summary["total_updates"].iloc[i]
    growth = monthly_summary["growth_rate"].iloc[i]
    height = bar.get_height()

    if month == max_month:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_val * 0.15),
            f"HIGHEST\n({val:,})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="darkblue",
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="darkblue", boxstyle="round"
            ),
        )
    elif month == min_month:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_val * 0.15),
            f"LOWEST\n({val:,})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="darkred",
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="darkred", boxstyle="round"
            ),
        )

    if pd.notnull(growth):
        color = "green" if growth >= 0 else "red"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_val * 0.02),
            f"{growth:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

plt.xticks(rotation=45)
plt.tight_layout()
IMAGE_DIR = os.path.join(BASE_DIR, "image")
os.makedirs(IMAGE_DIR, exist_ok=True)
plt.savefig(os.path.join(IMAGE_DIR, "Monthly_Update_Volume_Fixed.png"))
plt.close()

# comparing the age groups side by side
melted_df = monthly_summary.melt(
    id_vars="month_str",
    value_vars=["Kids %", "Adults %"],
    var_name="Age Group",
    value_name="Percentage",
)

fig2, ax2 = plt.subplots(figsize=(14, 7))
sns.barplot(
    data=melted_df,
    x="month_str",
    y="Percentage",
    hue="Age Group",
    palette=["#ff9999", "#66b3ff"],
    ax=ax2,
)

for p in ax2.patches:
    if p.get_height() > 0:
        ax2.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

ax2.set_title(
    "Monthly Age Distribution Side-by-Side (%)", fontsize=16, fontweight="bold"
)
ax2.set_ylabel("Percentage (%)")
ax2.set_ylim(0, 115)
plt.xticks(rotation=45)
plt.legend(title="Age Bracket", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "Age_Distribution_SideBySide_Fixed.png"))
plt.close()

# drawing a donut chart for age share
total_kids = monthly_summary["demo_age_5_17"].sum()
total_adults = monthly_summary["demo_age_17_"].sum()
plt.figure(figsize=(8, 8))
plt.pie(
    [total_kids, total_adults],
    labels=[f"Kids (5-17)\n({total_kids:,})", f"Adults (17+)\n({total_adults:,})"],
    colors=["#ff9999", "#66b3ff"],
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.85,
)
centre_circle = plt.Circle((0, 0), 0.70, fc="white")
plt.gca().add_artist(centre_circle)
plt.title("Total Annual Age Distribution Share", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "Age_Share_Donut_Fixed.png"))
plt.close()

print("All graphs generated successfully.")
