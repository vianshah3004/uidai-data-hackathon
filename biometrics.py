import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
sns.set_theme(style="darkgrid", context="notebook", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"
DATA_FILE = os.path.join(BASE_DIR, "final_UIDAI_cleaned_biometrics_data.csv")


def save_chart(filename):
    IMAGE_DIR = os.path.join(BASE_DIR, "image")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["total_updates"] = df["bio_age_5_17"] + df["bio_age_17_"]
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def analyze_and_plot(df):
    # Monthly Aggregation
    monthly = (
        df.groupby("month")
        .agg({"bio_age_5_17": "sum", "bio_age_17_": "sum", "total_updates": "sum"})
        .reset_index()
    )
    monthly["growth_pct"] = monthly["total_updates"].pct_change() * 100

    # IQR Outlier Detection
    Q1 = monthly["total_updates"].quantile(0.25)
    Q3 = monthly["total_updates"].quantile(0.75)
    IQR = Q3 - Q1
    monthly["iqr_outlier"] = (monthly["total_updates"] < (Q1 - 1.5 * IQR)) | (
        monthly["total_updates"] > (Q3 + 1.5 * IQR)
    )

    # 1. Monthly Trend
    plt.figure(figsize=(14, 6))
    plt.plot(monthly["month"], monthly["total_updates"], marker="o")
    plt.title("Monthly Biometric Trend")
    save_chart("01_monthly_trend.png")

    # 2. Highest & Lowest
    plt.figure(figsize=(14, 6))
    plt.plot(monthly["month"], monthly["total_updates"], marker="o")
    plt.scatter(
        monthly.loc[monthly["total_updates"].idxmax(), "month"],
        monthly["total_updates"].max(),
        s=150,
        label="Highest",
    )
    plt.scatter(
        monthly.loc[monthly["total_updates"].idxmin(), "month"],
        monthly["total_updates"].min(),
        s=150,
        label="Lowest",
    )
    plt.legend()
    plt.title("Highest & Lowest Month")
    save_chart("02_highest_lowest.png")

    # 3. Growth Rate
    plt.figure(figsize=(14, 6))
    plt.bar(monthly["month"], monthly["growth_pct"])
    plt.title("Monthly Growth Rate (%)")
    save_chart("03_growth_rate.png")

    # 4. Age-wise Pattern
    plt.figure(figsize=(14, 6))
    plt.stackplot(
        monthly["month"],
        monthly["bio_age_5_17"],
        monthly["bio_age_17_"],
        labels=["Kids", "Adults"],
    )
    plt.legend()
    plt.title("Age-wise Biometric Dependency")
    save_chart("04_age_pattern.png")

    # 5. State Dominance
    state_sum = df.groupby("state")["total_updates"].sum().sort_values(ascending=False)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=state_sum.values, y=state_sum.index)
    plt.title("State-wise Biometric Dominance")
    save_chart("05_state_dominance.png")

    # 6. Heatmap
    heatmap_df = (
        df.groupby(["state", "month"])["total_updates"].sum().unstack().fillna(0)
    )
    plt.figure(figsize=(18, 10))
    sns.heatmap(heatmap_df, cmap="YlOrRd")
    plt.title("Month x State Heatmap")
    save_chart("06_heatmap.png")

    # 7. Spike Detection
    monthly["spike"] = monthly["growth_pct"] > 40
    plt.figure(figsize=(14, 6))
    plt.plot(monthly["month"], monthly["total_updates"])
    plt.scatter(
        monthly.loc[monthly["spike"], "month"],
        monthly.loc[monthly["spike"], "total_updates"],
        s=150,
    )
    plt.title("Spike Detection")
    save_chart("07_spike_detection.png")

    # 8. IQR Outliers
    plt.figure(figsize=(14, 6))
    sns.barplot(
        x=monthly["month"], y=monthly["total_updates"], hue=monthly["iqr_outlier"]
    )
    plt.xticks(rotation=45)
    plt.title("IQR Outlier Months")
    save_chart("08_iqr_outliers.png")

    # 9. Boxplot
    plt.figure(figsize=(6, 8))
    sns.boxplot(y=monthly["total_updates"])
    plt.title("IQR Boxplot")
    save_chart("09_iqr_boxplot.png")

    # 10. Z-Score
    monthly["zscore"] = (
        monthly["total_updates"] - monthly["total_updates"].mean()
    ) / monthly["total_updates"].std()
    plt.figure(figsize=(14, 6))
    plt.scatter(monthly["month"], monthly["zscore"])
    plt.axhline(2, linestyle="--")
    plt.axhline(-2, linestyle="--")
    plt.title("Z-score Outlier Detection")
    save_chart("10_zscore.png")

    # 11. Outliers on Trend
    plt.figure(figsize=(14, 6))
    plt.plot(monthly["month"], monthly["total_updates"])
    plt.scatter(
        monthly.loc[monthly["iqr_outlier"], "month"],
        monthly.loc[monthly["iqr_outlier"], "total_updates"],
        s=200,
    )
    plt.title("IQR Outliers on Trend")
    save_chart("11_outliers_on_trend.png")

    # 12. State Distribution
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="state", y="total_updates", data=df)
    plt.xticks(rotation=90)
    plt.title("State-wise Distribution")
    save_chart("12_state_distribution.png")

    # 13. State Contribution Pie
    plt.figure(figsize=(7, 7))
    plt.pie(state_sum.head(5), labels=state_sum.head(5).index, autopct="%1.1f%%")
    plt.title("Top State Contribution")
    save_chart("13_state_contribution.png")

    # 14. Univariate Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(monthly["total_updates"], kde=True, color="teal", bins=20)
    plt.title("Distribution of Monthly Updates")
    save_chart("14_univariate_dist.png")

    # 15. Bivariate Relationship
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x="bio_age_5_17",
        y="bio_age_17_",
        data=monthly,
        scatter_kws={"s": 100, "alpha": 0.7, "color": "purple"},
        line_kws={"color": "salmon", "lw": 2},
    )
    plt.title("Bivariate: 5-17 vs 17+ Updates")
    save_chart("15_bivariate_scatter.png")

    # 16. Trivariate Bubble
    monthly["growth_direction"] = np.where(
        monthly["growth_pct"] >= 0, "Positive", "Negative"
    )
    monthly["growth_size"] = monthly["growth_pct"].abs() * 10 + 50
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=monthly,
        x="bio_age_5_17",
        y="total_updates",
        size="growth_size",
        hue="growth_direction",
        sizes=(50, 500),
        alpha=0.7,
        palette={"Positive": "green", "Negative": "red"},
    )
    plt.title("Trivariate Bubble Chart: Kids vs Total vs Growth")
    save_chart("16_trivariate_bubble.png")

    # 17. District Heatmap
    top_10_districts = df.groupby("district")["total_updates"].sum().nlargest(10).index
    district_monthly = (
        df[df["district"].isin(top_10_districts)]
        .groupby(["district", "month"])["total_updates"]
        .sum()
        .unstack()
        .fillna(0)
    )
    plt.figure(figsize=(16, 8))
    sns.heatmap(district_monthly, cmap="viridis", linewidths=0.5, linecolor="gray")
    plt.title("Top 10 Districts Monthly Activity")
    save_chart("17_top10_district_heatmap.png")

    # 18. Predictive Analysis (Polynomial)
    time_series_df = monthly[["month", "total_updates"]].copy()
    time_series_df["date_ordinal"] = time_series_df["month"].map(pd.Timestamp.toordinal)
    z = np.polyfit(time_series_df["date_ordinal"], time_series_df["total_updates"], 2)
    p = np.poly1d(z)

    last_date = time_series_df["month"].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
    future_ordinals = [d.toordinal() for d in future_dates]
    future_values = p(future_ordinals)

    plt.figure(figsize=(14, 7))
    plt.plot(
        monthly["month"],
        monthly["total_updates"],
        marker="o",
        label="Historical",
        color="teal",
    )
    plt.plot(
        future_dates,
        future_values,
        linestyle="--",
        marker="x",
        color="orange",
        label="Forecast",
    )
    plt.fill_between(
        future_dates,
        future_values * 0.9,
        future_values * 1.1,
        color="orange",
        alpha=0.2,
    )
    plt.title("6-Month Volume Forecast")
    plt.legend()
    save_chart("18_predictive_forecast.png")

    # 19. Anomaly: District vs State
    top_10_states = (
        df.groupby("state")["total_updates"].sum().nlargest(10).index.tolist()
    )
    district_state_map = (
        df[["district", "state"]]
        .drop_duplicates()
        .set_index("district")["state"]
        .to_dict()
    )

    anomaly_data = []
    colors = []
    print("\nCheck Top 10 Districts vs States:")
    for district in top_10_districts:
        state = district_state_map.get(district, "Unknown")
        volume = df[df["district"] == district]["total_updates"].sum()
        is_anomaly = state not in top_10_states
        anomaly_data.append({"District": district, "State": state, "Volume": volume})
        colors.append("red" if is_anomaly else "teal")
        if is_anomaly:
            print(
                f"  - ANOMALY: {district} ({state}) in Top 10 District but State NOT in Top 10."
            )

    anomaly_df = pd.DataFrame(anomaly_data)
    plt.figure(figsize=(14, 8))
    bp = sns.barplot(x="Volume", y="District", data=anomaly_df, palette=colors)
    plt.title("Top 10 Districts Anomaly Check")
    for index, row in anomaly_df.iterrows():
        bp.text(row.Volume, index, f" {row.State}", va="center", fontsize=10)
    save_chart("19_district_state_anomaly.png")

    print("\nAnalysis complete. 19 charts generated.")


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        analyze_and_plot(df)
