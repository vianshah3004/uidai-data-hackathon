import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# defining the base directory
BASE_DIR = '/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning'

# loading and prepping the data
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

# summarizing the data by month
monthly_summary = df.groupby("month_year")["total_updates"].sum().reset_index()
monthly_summary = monthly_summary.sort_values("month_year")

# setting up the linear regression model
monthly_summary["month_index"] = np.arange(len(monthly_summary))
X = monthly_summary[["month_index"]]
y = monthly_summary["total_updates"]
model = LinearRegression().fit(X, y)

# forecasting growth for the next quarter
future_indices = np.array(
    [[len(monthly_summary)], [len(monthly_summary) + 1], [len(monthly_summary) + 2]]
)
future_preds = model.predict(future_indices)
last_date = monthly_summary["month_year"].max()
future_dates = [(last_date + i).strftime("%Y-%m") for i in range(1, 4)]

# connecting the historical data to the forecast line
# getting the last data point
last_hist_month = monthly_summary["month_year"].iloc[-1].strftime("%Y-%m")
last_hist_val = monthly_summary["total_updates"].iloc[-1]

# prepping the forecast data bridging the gap
forecast_months = [last_hist_month] + future_dates
forecast_vals = [last_hist_val] + list(future_preds)

# Visualization
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(14, 7))

# plotting the past trend
ax.plot(
    monthly_summary["month_year"].astype(str),
    monthly_summary["total_updates"],
    marker="o",
    color="blue",
    label="Historical Trend",
    linewidth=3,
)

# plotting the future predictions
ax.plot(
    forecast_months,
    forecast_vals,
    marker="s",
    linestyle="--",
    color="red",
    label="Future Forecast (Predicted)",
    linewidth=3,
)

# labeling the future points
for i in range(1, len(forecast_months)):
    ax.text(
        forecast_months[i],
        forecast_vals[i] + (max(y) * 0.05),
        f"{int(forecast_vals[i]):,}",
        color="red",
        ha="center",
        fontweight="bold",
        fontsize=10,
    )

# making y-axis numbers readable
ax.ticklabel_format(style="plain", axis="y")
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

ax.set_title(
    "Continuous Trend Analysis & Predictive Forecast",
    fontsize=18,
    fontweight="bold",
    pad=20,
)
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("Total Updates", fontsize=14)
ax.legend(fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

IMAGE_DIR = os.path.join(BASE_DIR, "image")
os.makedirs(IMAGE_DIR, exist_ok=True)
plt.savefig(os.path.join(IMAGE_DIR, "connected_predictive_trend.png"))
plt.close()

print("Graph saved as 'connected_predictive_trend.png'. Line gap is now connected.")
