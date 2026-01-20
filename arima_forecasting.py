import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA

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


def run_and_plot_arima(
    daily_df,
    series_col,
    name,
    order=(5, 1, 0),
    forecast_days=30,
    color="blue",
    output_filename="forecast.png",
):
    print(f"Forecasting {name}...")

    series = daily_df[series_col]
    model = ARIMA(series, order=order)
    model_fit = model.fit()

    # Forecast
    forecast_result = model_fit.get_forecast(steps=forecast_days)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Dates
    last_date = daily_df["date"].max()
    future_dates = [
        last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)
    ]

    # Plotting
    plt.figure(figsize=(14, 7))

    # Show last 180 days history
    plot_start_idx = max(0, len(series) - 180)
    history_dates = daily_df["date"][plot_start_idx:]
    history_values = series.iloc[plot_start_idx:]

    plt.plot(
        history_dates,
        history_values,
        label=f"History (Last 6 Months)",
        color=color,
        linewidth=2,
    )
    plt.plot(
        future_dates,
        forecast_mean,
        label="Forecast",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    plt.fill_between(
        future_dates,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color="red",
        alpha=0.15,
        label="95% Confidence Interval",
    )

    plt.title(
        f"ARIMA Forecast: {name} (Next {forecast_days} Days)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    IMAGE_DIR = os.path.join(BASE_DIR, "image")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    output_path = os.path.join(IMAGE_DIR, output_filename)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved chart: {output_path}")


def main():
    print("Loading datasets...")

    # 1. Enrollment
    df_enrol = load_and_prep(
        "final_UIDAI_cleaned_enrol_data.csv", ["age_0_5", "age_5_17", "age_18_greater"]
    )
    if not df_enrol.empty:
        df_enrol["Enrollment_Total"] = df_enrol[
            ["age_0_5", "age_5_17", "age_18_greater"]
        ].sum(axis=1)

    # 2. Biometrics
    df_bio = load_and_prep(
        "final_UIDAI_cleaned_biometrics_data.csv", ["bio_age_5_17", "bio_age_17_"]
    )
    if not df_bio.empty:
        df_bio["Biometric_Total"] = df_bio[["bio_age_5_17", "bio_age_17_"]].sum(axis=1)

    # 3. Demographics
    df_demo = load_and_prep(
        "final_UIDAI_cleaned_demographic_data.csv", ["demo_age_5_17", "demo_age_17_"]
    )
    if not df_demo.empty:
        df_demo["Demographic_Total"] = df_demo[["demo_age_5_17", "demo_age_17_"]].sum(
            axis=1
        )

    # Aggregation
    print("Aggregating data...")
    agg_cols = ["date", "state", "district"]

    def aggregate(df, val_col, name):
        if df.empty:
            return pd.DataFrame(columns=agg_cols + [name])
        return (
            df.groupby(agg_cols)[val_col]
            .sum()
            .reset_index()
            .rename(columns={val_col: name})
        )

    agg_enrol = aggregate(df_enrol, "Enrollment_Total", "Enrollments")
    agg_bio = aggregate(df_bio, "Biometric_Total", "Biometrics")
    agg_demo = aggregate(df_demo, "Demographic_Total", "Demographics")

    merged = pd.merge(agg_enrol, agg_bio, on=agg_cols, how="outer")
    merged = pd.merge(merged, agg_demo, on=agg_cols, how="outer")
    merged[["Enrollments", "Biometrics", "Demographics"]] = merged[
        ["Enrollments", "Biometrics", "Demographics"]
    ].fillna(0)

    # Daily Time Series
    daily = (
        merged.groupby("date")[["Enrollments", "Biometrics", "Demographics"]]
        .sum()
        .reset_index()
    )
    daily = daily.sort_values("date")

    # Fill gaps
    if not daily.empty:
        idx = pd.date_range(daily["date"].min(), daily["date"].max())
        daily = (
            daily.set_index("date")
            .reindex(idx, fill_value=0)
            .reset_index()
            .rename(columns={"index": "date"})
        )

        print(f"Daily Data Prepared: {daily.shape[0]} days")

        try:
            run_and_plot_arima(
                daily,
                "Enrollments",
                "Enrollments",
                order=(5, 1, 1),
                color="#1f77b4",
                output_filename="ARIMA_01_Enrollments.png",
            )
            run_and_plot_arima(
                daily,
                "Biometrics",
                "Biometrics",
                order=(5, 1, 1),
                color="#ff7f0e",
                output_filename="ARIMA_02_Biometrics.png",
            )
            run_and_plot_arima(
                daily,
                "Demographics",
                "Demographics",
                order=(5, 1, 1),
                color="#2ca02c",
                output_filename="ARIMA_03_Demographics.png",
            )
            print("\nForecasting complete. Graphs generated.")
        except Exception as e:
            print(f"Forecasting error: {e}")
    else:
        print("No data available for forecasting.")


if __name__ == "__main__":
    main()
