import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import zscore

# Configuration
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Verdana", "sans-serif"]

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def load_and_prep(filename, value_cols):
    """Loads a CSV and standardizes date/numeric columns."""
    filepath = os.path.join(BASE_DIR, filename)
    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return pd.DataFrame()


def save_chart(filename):
    IMAGE_DIR = os.path.join(BASE_DIR, "image")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def aggregate_dataset(df, value_col, name_prefix):
    agg_cols = ["date", "state", "district"]
    return (
        df.groupby(agg_cols)[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: name_prefix})
    )


def main():
    print("Loading datasets...")

    # 1. Enrollment
    df_enrol = load_and_prep(
        "final_UIDAI_cleaned_enrol_data.csv", ["age_0_5", "age_5_17", "age_18_greater"]
    )
    if not df_enrol.empty:
        df_enrol["Enrollment_Total"] = (
            df_enrol["age_0_5"] + df_enrol["age_5_17"] + df_enrol["age_18_greater"]
        )

    # 2. Biometrics
    df_bio = load_and_prep(
        "final_UIDAI_cleaned_biometrics_data.csv", ["bio_age_5_17", "bio_age_17_"]
    )
    if not df_bio.empty:
        df_bio["Biometric_Total"] = df_bio["bio_age_5_17"] + df_bio["bio_age_17_"]

    # 3. Demographics
    df_demo = load_and_prep(
        "final_UIDAI_cleaned_demographic_data.csv", ["demo_age_5_17", "demo_age_17_"]
    )
    if not df_demo.empty:
        df_demo["Demographic_Total"] = (
            df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
        )

    print("Aggregating and merging data...")

    # Aggregation
    agg_enrol = aggregate_dataset(df_enrol, "Enrollment_Total", "Enrollments")
    agg_bio = aggregate_dataset(df_bio, "Biometric_Total", "Biometrics")
    agg_demo = aggregate_dataset(df_demo, "Demographic_Total", "Demographics")

    # Merge
    agg_cols = ["date", "state", "district"]
    merged = pd.merge(agg_enrol, agg_bio, on=agg_cols, how="outer")
    merged = pd.merge(merged, agg_demo, on=agg_cols, how="outer")
    merged[["Enrollments", "Biometrics", "Demographics"]] = merged[
        ["Enrollments", "Biometrics", "Demographics"]
    ].fillna(0)

    merged["Total_Transactions"] = (
        merged["Enrollments"] + merged["Biometrics"] + merged["Demographics"]
    )
    merged["Month"] = merged["date"].dt.to_period("M").dt.to_timestamp()

    # Daily Aggregation
    daily = (
        merged.groupby("date")[
            ["Enrollments", "Biometrics", "Demographics", "Total_Transactions"]
        ]
        .sum()
        .reset_index()
    )

    # State Aggregation
    state_agg = (
        merged.groupby("state")[
            ["Enrollments", "Biometrics", "Demographics", "Total_Transactions"]
        ]
        .sum()
        .sort_values("Total_Transactions", ascending=False)
    )

    # --- Visualizations ---
    print("Generating charts...")

    # 1. Total Activity Stacked Area Chart
    plt.figure(figsize=(14, 7))
    plt.stackplot(
        daily["date"],
        daily["Enrollments"],
        daily["Biometrics"],
        daily["Demographics"],
        labels=["Enrollments", "Biometrics", "Demographics"],
        alpha=0.8,
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    plt.title(
        "Cross-Dataset: Daily Activity Composition", fontsize=16, fontweight="bold"
    )
    plt.legend(loc="upper left")
    save_chart("Cross_01_Activity_Stack.png")

    # 2. Correlation Heatmap
    corr_matrix = merged[["Enrollments", "Biometrics", "Demographics"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1)
    plt.title("Correlation between Transaction Types")
    save_chart("Cross_02_Correlation.png")

    # 3. State-wise Operational Mix (Top 10)
    top_10 = state_agg.head(10)
    top_10_norm = top_10[["Enrollments", "Biometrics", "Demographics"]].div(
        top_10[["Enrollments", "Biometrics", "Demographics"]].sum(axis=1), axis=0
    )

    top_10_norm.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="viridis")
    plt.title("Operational Mix by Top 10 States (Normalized)")
    plt.ylabel("Share of Transactions")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    save_chart("Cross_03_State_Mix.png")

    # 4. Operational Ratio Analysis: Updates vs Enrollments
    district_agg = merged.groupby("district")[
        ["Enrollments", "Biometrics", "Demographics"]
    ].sum()
    district_agg = district_agg[district_agg.sum(axis=1) > 1000]

    plt.figure(figsize=(10, 8))
    updates = district_agg["Biometrics"] + district_agg["Demographics"]
    sns.scatterplot(
        x=district_agg["Enrollments"],
        y=updates,
        size=district_agg["Enrollments"],
        sizes=(20, 400),
        alpha=0.6,
        color="purple",
    )
    plt.title("Enrollments vs Updates (Bio + Demo) per District")
    plt.xlabel("Total Enrollments")
    plt.ylabel("Total Updates")
    save_chart("Cross_04_Ratio_Scatter.png")

    # 5. Anomaly Detection
    daily["zscore"] = zscore(daily["Total_Transactions"])
    daily["anomaly"] = np.abs(daily["zscore"]) > 2.5

    plt.figure(figsize=(14, 6))
    plt.plot(
        daily["date"],
        daily["Total_Transactions"],
        color="gray",
        alpha=0.5,
        label="Daily Volume",
    )
    plt.scatter(
        daily.loc[daily["anomaly"], "date"],
        daily.loc[daily["anomaly"], "Total_Transactions"],
        color="red",
        s=100,
        label="Anomaly (> 2.5 Z-Score)",
        zorder=5,
    )

    max_idx = daily["Total_Transactions"].idxmax()
    plt.annotate(
        "Peak",
        xy=(daily.iloc[max_idx]["date"], daily.iloc[max_idx]["Total_Transactions"]),
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    plt.title("Multivariate Time Series Anomaly Detection")
    plt.legend()
    save_chart("Cross_05_Anomaly_Detection.png")

    # 6. Bubble Chart (3D Analysis)
    plt.figure(figsize=(12, 8))
    weekly = merged.groupby("Month")[
        ["Enrollments", "Biometrics", "Demographics"]
    ].sum()
    sns.scatterplot(
        data=weekly,
        x="Enrollments",
        y="Biometrics",
        size="Demographics",
        sizes=(100, 1000),
        hue="Demographics",
        palette="flare",
        legend=False,
    )
    plt.title("Dimensional Analysis (Monthly): Enrol vs Bio vs Demo")
    save_chart("Cross_06_Bubble_Month.png")

    # 7. District Discrepancy Heatmap
    district_agg["Bio_Share"] = district_agg["Biometrics"] / (
        district_agg["Biometrics"] + district_agg["Demographics"]
    )
    skewed = district_agg.sort_values("Bio_Share", ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        skewed[["Biometrics", "Demographics"]], annot=True, fmt=".0f", cmap="YlGnBu"
    )
    plt.title("Top 20 Districts with Biometric-Heavy Skew")
    save_chart("Cross_07_Skewed_Districts.png")

    # 8. Forecasting
    try:
        print("Forecasting...")
        model = ARIMA(daily["Total_Transactions"], order=(5, 1, 1))
        fit = model.fit()
        forecast_steps = 30
        forecast = fit.forecast(steps=forecast_steps)

        last_date = daily["date"].max()
        future_dates = [
            last_date + pd.Timedelta(days=i) for i in range(1, forecast_steps + 1)
        ]

        plt.figure(figsize=(14, 6))
        plt.plot(daily["date"], daily["Total_Transactions"], label="Historical")
        plt.plot(future_dates, forecast, label="Forecast", color="red", linestyle="--")
        plt.fill_between(
            future_dates, forecast * 0.9, forecast * 1.1, color="red", alpha=0.1
        )
        plt.title("30-Day Forecast of Combined Transactions")
        plt.legend()
        save_chart("Cross_08_Forecast.png")
    except Exception as e:
        print(f"Forecast failed: {e}")

    # 9. Violin Plot
    melted = daily.melt(
        id_vars="date",
        value_vars=["Enrollments", "Biometrics", "Demographics"],
        var_name="Type",
        value_name="Volume",
    )
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Type", y="Volume", data=melted, palette="muted")
    plt.title("Distribution Density of Transaction Types")
    save_chart("Cross_09_Violin_Dist.png")

    # 10. Weekly Pattern
    daily["DayOfWeek"] = daily["date"].dt.day_name()
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    daily["DayOfWeek"] = pd.Categorical(
        daily["DayOfWeek"], categories=days_order, ordered=True
    )

    pivot_day = daily.groupby("DayOfWeek", observed=True)[
        ["Enrollments", "Biometrics", "Demographics"]
    ].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_day, annot=True, fmt=".0f", cmap="Greens")
    plt.title("Average Daily Volume by Day of Week")
    save_chart("Cross_10_Weekly_Pattern.png")

    print("\nAnomaly Report:")
    anomalies = daily[daily["anomaly"]]
    if not anomalies.empty:
        print(f"Detected {len(anomalies)} anomalies (Z-Score > 2.5):")
        print(anomalies[["date", "Total_Transactions", "zscore"]])
    else:
        print("No significant anomalies detected (Z-Score > 2.5).")

    print("\nCross-Analysis Complete.")


if __name__ == "__main__":
    main()
