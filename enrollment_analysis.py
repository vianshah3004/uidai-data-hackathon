import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Visualization Settings
sns.set_theme(style="darkgrid", context="notebook", palette="viridis")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"
DATA_FILE = os.path.join(BASE_DIR, "final_UIDAI_cleaned_enrol_data.csv")


IMAGE_DIR = os.path.join(BASE_DIR, "image")
os.makedirs(IMAGE_DIR, exist_ok=True)


def save_chart(filename):
    path = os.path.join(IMAGE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def load_data():
    print("Loading enrollment data...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    cols = ["age_0_5", "age_5_17", "age_18_greater"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "total_enrollment" not in df.columns:
        df["total_enrollment"] = df[cols].sum(axis=1)

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.day_name()

    print(
        f"Data Loaded: {len(df)} records. Range: {df['date'].min().date()} to {df['date'].max().date()}"
    )
    return df


def generate_charts(df):
    # Aggregations
    daily = (
        df.groupby("date")[
            ["age_0_5", "age_5_17", "age_18_greater", "total_enrollment"]
        ]
        .sum()
        .reset_index()
    )
    monthly = (
        df.groupby("month")[
            ["age_0_5", "age_5_17", "age_18_greater", "total_enrollment"]
        ]
        .sum()
        .reset_index()
    )
    state_agg = (
        df.groupby("state")[
            ["total_enrollment", "age_0_5", "age_5_17", "age_18_greater"]
        ]
        .sum()
        .sort_values("total_enrollment", ascending=False)
    )
    district_agg = (
        df.groupby("district")["total_enrollment"].sum().sort_values(ascending=False)
    )

    # 1. Age Group Pie Chart (Distribution)
    total_counts = [
        df["age_0_5"].sum(),
        df["age_5_17"].sum(),
        df["age_18_greater"].sum(),
    ]
    plt.figure(figsize=(8, 8))
    plt.pie(
        total_counts,
        labels=["0-5 Years", "5-17 Years", "18+ Years"],
        autopct="%1.1f%%",
        colors=["#ff9999", "#66b3ff", "#99ff99"],
        explode=(0.05, 0.05, 0.05),
    )
    plt.title("Age Group Distribution", fontsize=16, fontweight="bold")
    save_chart("Chart_01_Age_Distribution.png")

    # 2. Daily Distribution Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=daily[["age_0_5", "age_5_17", "age_18_greater"]], palette="Set2")
    plt.title("Daily Enrollment Distribution by Age Group")
    plt.ylabel("Enrollments")
    save_chart("Chart_02_Age_BoxPlot.png")

    # 3. Average Daily Enrollment
    avg_counts = daily[["age_0_5", "age_5_17", "age_18_greater"]].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_counts.index, y=avg_counts.values, palette="viridis")
    plt.title("Average Daily Enrollment by Age Group")
    plt.ylabel("Average Count")
    save_chart("Chart_03_Avg_Enrollment.png")

    # 4. Statistics Summary
    desc_stats = (
        daily[["total_enrollment", "age_0_5", "age_5_17", "age_18_greater"]]
        .describe()
        .round(2)
    )
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    table = plt.table(
        cellText=desc_stats.values,
        colLabels=desc_stats.columns,
        rowLabels=desc_stats.index,
        loc="center",
        cellLoc="center",
        colWidths=[0.2] * 4,
    )
    table.scale(1.2, 1.2)
    plt.title("Daily Enrollment Statistics Summary", y=1.05)
    save_chart("Chart_04_Enrollment_Stats.png")

    # 5. Histogram of Enrollments
    plt.figure(figsize=(10, 6))
    sns.histplot(daily["total_enrollment"], kde=True, bins=30, color="purple")
    plt.title("Distribution of Total Daily Enrollments")
    plt.xlabel("Total Enrollment")
    save_chart("Chart_05_Enrollment_Dist.png")

    # 7. Top 15 States
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=state_agg.head(15)["total_enrollment"],
        y=state_agg.head(15).index,
        palette="mako",
    )
    plt.title("Top 15 States by Total Enrollment")
    plt.xlabel("Total Enrollment")
    save_chart("Chart_07_Top15_States.png")

    # 10. Moving Average
    daily["MA_7"] = daily["total_enrollment"].rolling(window=7).mean()
    plt.figure(figsize=(14, 6))
    plt.plot(
        daily["date"],
        daily["total_enrollment"],
        alpha=0.5,
        label="Daily Actual",
        color="gray",
    )
    plt.plot(daily["date"], daily["MA_7"], color="blue", linewidth=2, label="7-Day MA")
    plt.title("Daily Enrollment Trend (with 7-Day MA)")
    plt.legend()
    save_chart("Chart_10_Daily_Trend_MA.png")

    # 11. Age Group Trends
    plt.figure(figsize=(14, 6))
    plt.plot(daily["date"], daily["age_0_5"], label="0-5 Years", alpha=0.8)
    plt.plot(daily["date"], daily["age_5_17"], label="5-17 Years", alpha=0.8)
    plt.plot(daily["date"], daily["age_18_greater"], label="18+ Years", alpha=0.8)
    plt.title("Daily Trends by Age Group")
    plt.legend()
    save_chart("Chart_11_Age_Group_Trends.png")

    # 13. Monthly Pattern
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=monthly["month"].dt.strftime("%Y-%m"),
        y=monthly["total_enrollment"],
        palette="Blues_d",
    )
    plt.xticks(rotation=45)
    plt.title("Monthly Enrollment Totals")
    save_chart("Chart_13_Monthly_Pattern.png")

    # 15. Top Districts
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=district_agg.head(15), y=district_agg.head(15).index, palette="rocket"
    )
    plt.title("Top 15 Districts by Total Enrollment")
    save_chart("Chart_15_Top15_Districts.png")

    # 16. Monthly Heatmap (Top 10 States)
    top_10_states = state_agg.head(10).index
    heatmap_data = (
        df[df["state"].isin(top_10_states)]
        .groupby(["state", "month"])["total_enrollment"]
        .sum()
        .unstack()
        .fillna(0)
    )
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.5)
    plt.title("Monthly Enrollment Heatmap (Top 10 States)")
    save_chart("Chart_16_State_Month_Heatmap.png")

    # 17. Age Group Composition
    monthly_melt = monthly.melt(
        id_vars=["month"],
        value_vars=["age_0_5", "age_5_17", "age_18_greater"],
        var_name="Age Group",
        value_name="Count",
    )
    pivoted = monthly_melt.pivot(index="month", columns="Age Group", values="Count")
    pivoted.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="viridis")
    plt.title("Monthly Age Group Composition")
    plt.xticks(rotation=45)
    save_chart("Chart_17_Age_Composition_Month.png")

    # 18. Top 5 States Monthly Trend
    top_5_states = state_agg.head(5).index
    monthly_state = (
        df[df["state"].isin(top_5_states)]
        .groupby(["month", "state"])["total_enrollment"]
        .sum()
        .unstack()
    )
    plt.figure(figsize=(14, 7))
    for state in top_5_states:
        plt.plot(monthly_state.index, monthly_state[state], marker="o", label=state)
    plt.title("Monthly Trend: Top 5 States")
    plt.legend()
    save_chart("Chart_18_Top5_State_Trends.png")

    # 24. Spike Detection
    daily["mean"] = (
        daily["total_enrollment"].rolling(window=30, min_periods=1, center=True).mean()
    )
    daily["std"] = (
        daily["total_enrollment"].rolling(window=30, min_periods=1, center=True).std()
    )
    daily["spike"] = daily["total_enrollment"] > (daily["mean"] + 2 * daily["std"])

    plt.figure(figsize=(14, 6))
    plt.plot(
        daily["date"],
        daily["total_enrollment"],
        color="lightgray",
        label="Daily Enrollment",
    )
    plt.scatter(
        daily[daily["spike"]]["date"],
        daily[daily["spike"]]["total_enrollment"],
        color="red",
        s=50,
        label="Spike (> 2 STD)",
        zorder=5,
    )
    plt.title("Significant Anomalies/Spikes")
    plt.legend()
    save_chart("Chart_24_Spike_Detection.png")

    # 27 & 29. Trend with Extremes
    max_idx = daily["total_enrollment"].idxmax()
    min_idx = daily["total_enrollment"].idxmin()

    plt.figure(figsize=(14, 6))
    plt.plot(daily["date"], daily["total_enrollment"], label="Trend")
    plt.scatter(
        daily.loc[max_idx, "date"],
        daily.loc[max_idx, "total_enrollment"],
        color="green",
        s=100,
        label=f"Max: {daily.loc[max_idx, 'total_enrollment']}",
        zorder=5,
    )
    plt.scatter(
        daily.loc[min_idx, "date"],
        daily.loc[min_idx, "total_enrollment"],
        color="red",
        s=100,
        label=f"Min: {daily.loc[min_idx, 'total_enrollment']}",
        zorder=5,
    )
    plt.title("Enrollment Trend with Highs & Lows")
    plt.legend()
    save_chart("Chart_27_Trend_Extremes.png")

    print(f"\nSpike Summary: {daily['spike'].sum()} spikes detected.")

    # Time Series Decomposition (Assume weekly period=7 for daily data)
    decomposition = seasonal_decompose(
        daily.set_index("date")["total_enrollment"], model="additive", period=7
    )

    plt.figure(figsize=(12, 4))
    plt.plot(decomposition.observed)
    plt.title("Original Time Series")
    save_chart("Chart_35_TS_Observed.png")

    plt.figure(figsize=(12, 4))
    plt.plot(decomposition.trend, color="orange")
    plt.title("Time Series Trend")
    save_chart("Chart_36_TS_Trend.png")

    plt.figure(figsize=(12, 4))
    plt.plot(decomposition.seasonal, color="green")
    plt.title("Time Series Seasonality (Weekly)")
    save_chart("Chart_37_TS_Seasonal.png")

    # ARIMA Forecast
    try:
        model = ARIMA(daily["total_enrollment"], order=(5, 1, 0))
        model_fit = model.fit()
        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps)

        last_date = daily["date"].max()
        future_dates = [
            last_date + pd.Timedelta(days=i) for i in range(1, forecast_steps + 1)
        ]

        plt.figure(figsize=(14, 6))
        plt.plot(daily["date"], daily["total_enrollment"], label="Historical")
        plt.plot(
            future_dates, forecast, label="30-Day Forecast", color="red", linestyle="--"
        )
        plt.title("ARIMA 30-Day Forecast")
        plt.legend()
        save_chart("Chart_40_ARIMA_Forecast.png")
    except Exception as e:
        print(f"ARIMA Forecast failed: {e}")

    # Bivariate Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=daily["age_0_5"], y=daily["total_enrollment"], color="teal", alpha=0.6
    )
    plt.title("Age 0-5 vs Total Enrollments")
    save_chart("Chart_Bivariate_Scatter.png")

    # District Anomaly Check
    check_district_anomalies(df, district_agg, state_agg)


def check_district_anomalies(df, district_agg, state_agg):
    # Top 15 districts where parent state is NOT in Top 10 states
    top_15_districts = district_agg.head(15).index
    top_10_states = set(state_agg.head(10).index)

    # Map district to state
    district_to_state = (
        df[["district", "state"]]
        .drop_duplicates()
        .set_index("district")["state"]
        .to_dict()
    )

    anomaly_data = []
    for district in top_15_districts:
        state = district_to_state.get(district)
        is_anomaly = state not in top_10_states
        anomaly_data.append(
            {
                "District": district,
                "State": state,
                "Volume": district_agg.loc[district],
                "Is_Anomaly": is_anomaly,
            }
        )

    anomaly_df = pd.DataFrame(anomaly_data)
    colors = ["red" if x else "green" for x in anomaly_df["Is_Anomaly"]]

    plt.figure(figsize=(14, 8))
    bp = sns.barplot(x="Volume", y="District", data=anomaly_df, palette=colors)
    plt.title("Top 15 Districts Analysis (Red = Anomaly: State not in Top 10)")

    for index, row in anomaly_df.iterrows():
        bp.text(row.Volume, index, f" {row.State}", va="center", fontsize=9)

    save_chart("Chart_19_District_State_Anomaly.png")


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        generate_charts(df)
        print("\nAll charts generated successfully.")
