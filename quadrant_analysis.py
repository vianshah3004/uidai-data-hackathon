import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def generate_quadrant_zones():
    print("--- Loading Datasets ---")
    try:
        df_demo = pd.read_csv(
            os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv")
        )
        df_enrol = pd.read_csv(
            os.path.join(BASE_DIR, "final_UIDAI_cleaned_enrol_data.csv")
        )

        for df in [df_demo, df_enrol]:
            df.columns = df.columns.str.strip()
            df["district"] = df["district"].astype(str).str.title().str.strip()
            df["state"] = df["state"].astype(str).str.title().str.strip()

        # Metrics
        df_demo["Total_Updates"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
        demo_agg = (
            df_demo.groupby(["state", "district"])["Total_Updates"].sum().reset_index()
        )

        df_enrol["Total_Enrol"] = (
            df_enrol["age_0_5"] + df_enrol["age_5_17"] + df_enrol["age_18_greater"]
        )
        enrol_agg = (
            df_enrol.groupby(["state", "district"])["Total_Enrol"].sum().reset_index()
        )

        merged_df = pd.merge(demo_agg, enrol_agg, on=["state", "district"], how="inner")

        # Z-Scores
        merged_df["Churn_Z"] = zscore(merged_df["Total_Updates"])
        merged_df["Growth_Z"] = zscore(merged_df["Total_Enrol"])

        print(f"--- Classified {len(merged_df)} Districts ---")

        plt.figure(figsize=(12, 10))
        sns.set_style("whitegrid")

        sns.scatterplot(
            data=merged_df,
            x="Growth_Z",
            y="Churn_Z",
            hue="state",
            s=100,
            alpha=0.6,
            legend=False,
        )

        plt.axhline(0, color="black", linewidth=1.5)
        plt.axvline(0, color="black", linewidth=1.5)

        # Annotations
        plt.text(
            2,
            2,
            "High Growth, High Churn\nAction: Infrastructure Upgrade",
            fontsize=11,
            weight="bold",
            color="green",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="green"),
        )

        plt.text(
            -2,
            2,
            "Low Growth, High Churn\nAction: Migration Support",
            fontsize=11,
            weight="bold",
            color="orange",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="orange"),
        )

        plt.text(
            2,
            -2,
            "High Growth, Low Churn\nAction: New Service Centers",
            fontsize=11,
            weight="bold",
            color="blue",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="blue"),
        )

        plt.text(
            -2,
            -2,
            "Low Growth, Low Churn\nAction: Awareness Campaigns",
            fontsize=11,
            weight="bold",
            color="gray",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        for i in range(merged_df.shape[0]):
            row = merged_df.iloc[i]
            if abs(row["Growth_Z"]) > 2.5 or abs(row["Churn_Z"]) > 2.5:
                plt.text(
                    row["Growth_Z"] + 0.1, row["Churn_Z"], row["district"], fontsize=8
                )

        plt.title("District Classification Matrix", fontsize=15)
        plt.xlabel("Organic Growth (New Enrollments Z-Score)", fontsize=12)
        plt.ylabel("Demographic Churn (Updates Z-Score)", fontsize=12)

        IMAGE_DIR = os.path.join(BASE_DIR, "image")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        output_path = os.path.join(IMAGE_DIR, "visual_C4_quadrant_zones.png")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nSaved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    generate_quadrant_zones()
