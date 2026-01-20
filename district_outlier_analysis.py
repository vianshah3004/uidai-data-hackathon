import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def generate_district_outliers():
    print("--- Loading & Merging Datasets ---")
    try:
        df_bio = pd.read_csv(
            os.path.join(BASE_DIR, "final_UIDAI_cleaned_biometrics_data.csv")
        )
        df_demo = pd.read_csv(
            os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv")
        )
        df_enrol = pd.read_csv(
            os.path.join(BASE_DIR, "final_UIDAI_cleaned_enrol_data.csv")
        )

        for df in [df_bio, df_demo, df_enrol]:
            df.columns = df.columns.str.strip()
            df["district"] = df["district"].astype(str).str.title().str.strip()
            df["state"] = df["state"].astype(str).str.title().str.strip()

        # Aggregation
        bio_agg = (
            df_bio.groupby(["state", "district"])["bio_age_17_"]
            .sum()
            .reset_index()
            .rename(columns={"bio_age_17_": "Bio_Updates"})
        )

        df_demo["Total_Demo"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
        demo_agg = (
            df_demo.groupby(["state", "district"])["Total_Demo"].sum().reset_index()
        )

        enrol_agg = (
            df_enrol.groupby(["state", "district"])["age_18_greater"]
            .sum()
            .reset_index()
            .rename(columns={"age_18_greater": "New_Adults"})
        )

        merged_df = pd.merge(demo_agg, bio_agg, on=["state", "district"], how="inner")
        merged_df = pd.merge(
            merged_df, enrol_agg, on=["state", "district"], how="inner"
        )

        # Z-Scores
        merged_df["Demo_Z"] = zscore(merged_df["Total_Demo"])
        merged_df["Bio_Z"] = zscore(merged_df["Bio_Updates"])
        merged_df["Adult_Z"] = zscore(merged_df["New_Adults"])

        outliers = merged_df[
            (merged_df["Demo_Z"] > 2)
            | (merged_df["Bio_Z"] > 2)
            | (merged_df["Adult_Z"] > 2)
        ].copy()

        print(f"--- Identified {len(outliers)} Statistical Outliers ---")

        plt.figure(figsize=(14, 8))
        sns.scatterplot(
            data=outliers,
            x="Demo_Z",
            y="Bio_Z",
            size="Adult_Z",
            sizes=(50, 600),
            hue="state",
            alpha=0.7,
            palette="tab10",
            legend=False,
        )

        for i in range(outliers.shape[0]):
            row = outliers.iloc[i]
            if row["Adult_Z"] > 3 or row["Demo_Z"] > 3 or row["Bio_Z"] > 3:
                plt.text(
                    row["Demo_Z"] + 0.1,
                    row["Bio_Z"],
                    row["district"],
                    fontsize=9,
                    weight="bold",
                )

        plt.axhline(2, color="red", linestyle="--", alpha=0.5)
        plt.axvline(2, color="red", linestyle="--", alpha=0.5)

        plt.text(
            3,
            3,
            "CRITICAL RISK ZONE",
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        plt.title("District Outlier Analysis: Multi-Dimensional Anomalies", fontsize=16)
        plt.xlabel("Demographic Churn (Z-Score)")
        plt.ylabel("Biometric Volatility (Z-Score)")

        IMAGE_DIR = os.path.join(BASE_DIR, "image")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        output_path = os.path.join(IMAGE_DIR, "visual_C2_district_outliers.png")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nSaved: {output_path}")
        # plt.show() # Removed show() for non-interactive execution

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    generate_district_outliers()
