import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# defining the base directory
BASE_DIR = '/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning'

# just loading the demographic csv
df = pd.read_csv(os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv"))
print(df.duplicated().sum())
print(df["district"].unique())
print(len(df["district"].unique()))
print(df["state"].unique())
print(len(df["state"].unique()))
print(df.isna().sum())
df["date"] = pd.to_datetime(df["date"])
df["month_year"] = df["date"].dt.to_period("M")

# organizing the data monthly
monthly_data = (
    df.groupby("month_year")
    .agg({"demo_age_5_17": "sum", "demo_age_17_": "sum"})
    .reset_index()
)

monthly_data = monthly_data.sort_values("month_year")
monthly_data["month_index"] = np.arange(len(monthly_data))

# creating separate forecast models for kids and adults
X = monthly_data[["month_index"]]
y_kids = monthly_data["demo_age_5_17"]
y_adults = monthly_data["demo_age_17_"]

model_kids = LinearRegression().fit(X, y_kids)
model_adults = LinearRegression().fit(X, y_adults)

# looking 3 months ahead
future_indices = np.array(
    [[len(monthly_data)], [len(monthly_data) + 1], [len(monthly_data) + 2]]
)
preds_kids = model_kids.predict(future_indices)
preds_adults = model_adults.predict(future_indices)

# connecting the dots for the chart
last_hist_month = monthly_data["month_year"].iloc[-1].strftime("%Y-%m")
future_dates = [
    (monthly_data["month_year"].max() + i).strftime("%Y-%m") for i in range(1, 4)
]
forecast_months = [last_hist_month] + future_dates

# bridging the gap for kids data
kids_forecast_y = [monthly_data["demo_age_5_17"].iloc[-1]] + list(preds_kids)
# bridging the gap for adults data
adults_forecast_y = [monthly_data["demo_age_17_"].iloc[-1]] + list(preds_adults)

# summarizing the predictions into a table
pred_summary = pd.DataFrame(
    {
        "Month": future_dates,
        "Kids_Forecast": preds_kids.astype(int),
        "Adults_Forecast": preds_adults.astype(int),
    }
)
pred_summary["Total_Forecast"] = (
    pred_summary["Kids_Forecast"] + pred_summary["Adults_Forecast"]
)
pred_summary["Kids_Share_%"] = (
    pred_summary["Kids_Forecast"] / pred_summary["Total_Forecast"] * 100
).round(2)
pred_summary["Adults_Share_%"] = (
    pred_summary["Adults_Forecast"] / pred_summary["Total_Forecast"] * 100
).round(2)


IMAGE_DIR = os.path.join(BASE_DIR, "image")
os.makedirs(IMAGE_DIR, exist_ok=True)
pred_summary.to_csv(os.path.join(IMAGE_DIR, "Age_Group_Future_Predictions.csv"), index=False)

# drawing the chart
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))

# plotting the past data
ax.plot(
    monthly_data["month_year"].astype(str),
    monthly_data["demo_age_5_17"],
    marker="o",
    color="#ff9999",
    label="Kids (5-17) History",
    linewidth=3,
)
ax.plot(
    monthly_data["month_year"].astype(str),
    monthly_data["demo_age_17_"],
    marker="o",
    color="#66b3ff",
    label="Adults (17+) History",
    linewidth=3,
)

# plotting the predictions
ax.plot(
    forecast_months,
    kids_forecast_y,
    marker="s",
    linestyle="--",
    color="#cc7a7a",
    label="Kids Forecast",
    linewidth=2,
)
ax.plot(
    forecast_months,
    adults_forecast_y,
    marker="s",
    linestyle="--",
    color="#4d8acc",
    label="Adults Forecast",
    linewidth=2,
)

# Formatting
ax.ticklabel_format(style="plain", axis="y")
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

ax.set_title("Age-Wise Future Trend Prediction", fontsize=18, fontweight="bold", pad=20)
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("Total Updates", fontsize=14)
ax.legend(fontsize=12)

# adding percentage labels to the forecast
for i in range(1, len(forecast_months)):
    # Total for that month
    m_total = preds_kids[i - 1] + preds_adults[i - 1]
    k_share = preds_kids[i - 1] / m_total * 100
    a_share = preds_adults[i - 1] / m_total * 100

    # Text for Kids
    ax.text(
        forecast_months[i],
        kids_forecast_y[i] + (monthly_data["demo_age_17_"].max() * 0.02),
        f"{k_share:.1f}%",
        color="#cc7a7a",
        ha="center",
        fontweight="bold",
        fontsize=9,
    )
    # Text for Adults
    ax.text(
        forecast_months[i],
        adults_forecast_y[i] + (monthly_data["demo_age_17_"].max() * 0.02),
        f"{a_share:.1f}%",
        color="#4d8acc",
        ha="center",
        fontweight="bold",
        fontsize=9,
    )

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "Age_Group_Future_Trend.png"))
plt.close()

print(pred_summary.to_string(index=False))
