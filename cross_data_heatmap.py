import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def generate_risk_heatmap():
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
            df["state"] = df["state"].astype(str).str.title().str.strip()

        # Aggregations
        bio_agg = (
            df_bio.groupby("state")["bio_age_17_"]
            .sum()
            .reset_index()
            .rename(columns={"bio_age_17_": "Bio_Updates"})
        )

        df_demo["Total_Demo"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
        demo_agg = df_demo.groupby("state")["Total_Demo"].sum().reset_index()

        enrol_agg = (
            df_enrol.groupby("state")["age_18_greater"]
            .sum()
            .reset_index()
            .rename(columns={"age_18_greater": "New_Adults"})
        )

        # Merge
        risk_df = pd.merge(demo_agg, bio_agg, on="state", how="outer")
        risk_df = pd.merge(risk_df, enrol_agg, on="state", how="outer").fillna(0)

        # Normalize
        scaler = MinMaxScaler()
        risk_df[["Norm_Demo", "Norm_Bio", "Norm_Adults"]] = scaler.fit_transform(
            risk_df[["Total_Demo", "Bio_Updates", "New_Adults"]]
        )

        # Risk Score (Weighted)
        risk_df["Risk_Score"] = (
            (risk_df["Norm_Adults"] * 0.5)
            + (risk_df["Norm_Bio"] * 0.3)
            + (risk_df["Norm_Demo"] * 0.2)
        )

        risk_df = risk_df.sort_values("Risk_Score", ascending=False).head(15)

        print("--- Top 5 High Risk States ---")
        print(risk_df[["state", "Risk_Score", "New_Adults"]].head(5))

        plt.figure(figsize=(12, 8))
        heatmap_data = risk_df.set_index("state")[
            ["Norm_Adults", "Norm_Bio", "Norm_Demo"]
        ]

        sns.heatmap(heatmap_data, cmap="Reds", annot=True, linewidths=0.5, fmt=".2f")

        plt.title("Cross-Data Risk Intensity Map (Normalized)", fontsize=16)
        plt.xlabel("Risk Factors")
        plt.ylabel("State")
        plt.xticks(
            [0.5, 1.5, 2.5],
            ["New Adult Enrol", "Biometric Volatility", "Demographic Churn"],
        )

        IMAGE_DIR = os.path.join(BASE_DIR, "image")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        output_path = os.path.join(IMAGE_DIR, "visual_C1_risk_heatmap.png")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nSaved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    generate_risk_heatmap()
