import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def analyze_biometric_data():
    print("--- Loading Biometric Data ---")

    try:
        path = os.path.join(BASE_DIR, "final_UIDAI_cleaned_biometrics_data.csv")
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        return

    # Data Preparation
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce", dayfirst=True, format="mixed"
    )
    df = df.dropna(subset=["date"])
    df.columns = df.columns.str.strip()

    # Aggregations
    df["Total_Biometric_Updates"] = df["bio_age_5_17"] + df["bio_age_17_"]
    df["Month_Year"] = df["date"].dt.to_period("M")

    monthly_data = (
        df.groupby("Month_Year")[
            ["bio_age_5_17", "bio_age_17_", "Total_Biometric_Updates"]
        ]
        .sum()
        .reset_index()
    )
    monthly_data["Month_Year"] = monthly_data["Month_Year"].astype(str)

    print("\nMonthly Overview:")
    print(monthly_data.head())

    # Visualization
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=monthly_data,
        x="Month_Year",
        y="bio_age_5_17",
        marker="o",
        label="Age 5-17 (Mandatory)",
        color="blue",
        linewidth=2,
    )

    sns.lineplot(
        data=monthly_data,
        x="Month_Year",
        y="bio_age_17_",
        marker="o",
        label="Age 17+ (Voluntary)",
        color="orange",
        linewidth=2,
    )

    sns.lineplot(
        data=monthly_data,
        x="Month_Year",
        y="Total_Biometric_Updates",
        marker="s",
        label="Total Updates",
        color="green",
        linestyle="--",
        linewidth=2.5,
    )

    plt.title("Monthly Biometric Update Trends by Age Group", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Updates", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    IMAGE_DIR = os.path.join(BASE_DIR, "image")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    output_path = os.path.join(IMAGE_DIR, "visual_B1_biometric_monthly_trend.png")
    plt.savefig(output_path)
    print(f"\nSaved visualization to: {output_path}")
    # plt.show()


if __name__ == "__main__":
    analyze_biometric_data()
