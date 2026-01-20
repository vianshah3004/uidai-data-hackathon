import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
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

# summarising data by month and state
state_monthly = (
    df.groupby(["month_year", "state"])
    .agg({"demo_age_5_17": "sum", "demo_age_17_": "sum"})
    .reset_index()
)

state_monthly["month_str"] = state_monthly["month_year"].astype(str)

# reshaping data to compare age groups
melted_state_age = state_monthly.melt(
    id_vars=["month_str", "state"],
    value_vars=["demo_age_5_17", "demo_age_17_"],
    var_name="Age_Group",
    value_name="Count",
)

melted_state_age["Age_Group"] = melted_state_age["Age_Group"].map(
    {"demo_age_5_17": "Kids (5-17)", "demo_age_17_": "Adults (17+)"}
)

# creating separate charts for each state
# arranging small charts in a grid
# height=4 and aspect=1.3 for clear viewing
# sharex=False to ensure every plot has its own date labels
g = sns.FacetGrid(
    melted_state_age,
    col="state",
    hue="Age_Group",
    col_wrap=3,
    height=4,
    aspect=1.3,
    sharex=False,
    sharey=False,
)

g.map(sns.lineplot, "month_str", "Count", marker="o", linewidth=2)

# styling the individual charts
for ax in g.axes.flat:
    # making y-axis numbers look clean
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # making dates readable
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Update Count", fontsize=10)

    # making titles bigger
    if ax.get_title():
        ax.set_title(ax.get_title().split("=")[-1], fontweight="bold", fontsize=12)

# adding a main title and legend
g.add_legend(
    title="Age Category", fontsize=10, bbox_to_anchor=(1.05, 0.5), loc="center left"
)
plt.subplots_adjust(top=0.96, hspace=0.6, wspace=0.3)
g.fig.suptitle(
    "State-wise Monthly Demographic Updates: 36 States/UTs Breakdown",
    fontsize=24,
    fontweight="bold",
)

# saving the detailed chart
IMAGE_DIR = os.path.join(BASE_DIR, "image")
os.makedirs(IMAGE_DIR, exist_ok=True)
plt.savefig(os.path.join(IMAGE_DIR, "clear_faceted_36_states_counts.png"), bbox_inches="tight", dpi=100)
plt.close()

print("Clear faceted chart generated: clear_faceted_36_states_counts.png")
