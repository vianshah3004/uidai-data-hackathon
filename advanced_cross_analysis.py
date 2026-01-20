import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as dist

# Settings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def save_chart(filename, print_msg=None):
    IMAGE_DIR = os.path.join(BASE_DIR, "image")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    if print_msg:
        print(f"Interpretation: {print_msg}")
    print("-" * 50)


def load_and_prep(filename, value_cols):
    filepath = os.path.join(BASE_DIR, filename)
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

        for col in value_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                print(f"Warning: Column {col} not found in {filename}")
                df[col] = 0

        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.title().str.strip()
        if "district" in df.columns:
            df["district"] = df["district"].astype(str).str.title().str.strip()

        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return pd.DataFrame()


def main():
    print("Loading datasets...")

    # 1. Enrollment
    df_enrol = load_and_prep(
        "final_UIDAI_cleaned_enrol_data.csv", ["age_0_5", "age_5_17", "age_18_greater"]
    )
    df_enrol["Enrollment_Total"] = (
        df_enrol["age_0_5"] + df_enrol["age_5_17"] + df_enrol["age_18_greater"]
    )

    # 2. Biometrics
    df_bio = load_and_prep(
        "final_UIDAI_cleaned_biometrics_data.csv", ["bio_age_5_17", "bio_age_17_"]
    )
    df_bio["Biometric_Total"] = df_bio["bio_age_5_17"] + df_bio["bio_age_17_"]

    # 3. Demographics
    df_demo = load_and_prep(
        "final_UIDAI_cleaned_demographic_data.csv", ["demo_age_5_17", "demo_age_17_"]
    )
    df_demo["Demographic_Total"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]

    # Aggregation
    print("Aggregating data...")
    grp_cols = ["state", "district"]
    dist_enrol = df_enrol.groupby(grp_cols)["Enrollment_Total"].sum().reset_index()
    dist_bio = df_bio.groupby(grp_cols)["Biometric_Total"].sum().reset_index()
    dist_demo = df_demo.groupby(grp_cols)["Demographic_Total"].sum().reset_index()

    merged_dist = pd.merge(dist_enrol, dist_bio, on=grp_cols, how="outer").fillna(0)
    merged_dist = pd.merge(merged_dist, dist_demo, on=grp_cols, how="outer").fillna(0)
    merged_dist.columns = [
        "state",
        "district",
        "Enrolment",
        "Biometrics",
        "Demographics",
    ]

    # Daily aggregation
    daily_enrol = df_enrol.groupby("date")["Enrollment_Total"].sum()
    daily_bio = df_bio.groupby("date")["Biometric_Total"].sum()
    daily_demo = df_demo.groupby("date")["Demographic_Total"].sum()

    daily_df = (
        pd.DataFrame(
            {
                "Enrolment": daily_enrol,
                "Biometrics": daily_bio,
                "Demographics": daily_demo,
            }
        )
        .fillna(0)
        .reset_index()
    )

    # --- Charts ---
    # 1. Ratio Heatmap
    print("\nGenerating Ratio-Based Heatmaps...")
    epsilon = 1e-9
    merged_dist["Bio_to_Enrol_Ratio"] = merged_dist["Biometrics"] / (
        merged_dist["Enrolment"] + epsilon
    )
    merged_dist["Demo_to_Enrol_Ratio"] = merged_dist["Demographics"] / (
        merged_dist["Enrolment"] + epsilon
    )

    state_ratios = merged_dist.groupby("state")[
        ["Enrolment", "Biometrics", "Demographics"]
    ].sum()
    state_ratios["Bio_Enrol_Ratio"] = state_ratios["Biometrics"] / (
        state_ratios["Enrolment"] + epsilon
    )
    state_ratios["Demo_Enrol_Ratio"] = state_ratios["Demographics"] / (
        state_ratios["Enrolment"] + epsilon
    )
    state_ratios["Update_Intensity"] = (
        state_ratios["Biometrics"] + state_ratios["Demographics"]
    ) / (state_ratios["Enrolment"] + epsilon)

    top_states_heat = state_ratios.sort_values(
        "Update_Intensity", ascending=False
    ).head(20)

    plt.figure(figsize=(10, 12))
    sns.heatmap(
        top_states_heat[["Bio_Enrol_Ratio", "Demo_Enrol_Ratio"]],
        annot=True,
        cmap="Reds",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Ratio-Based Heatmap: Update vs Enrolment Intensity (Top 20 States)")
    save_chart(
        "Adv_01_Ratio_Heatmap.png",
        "High Bio/Demo ratios indicate structural anomalies.",
    )

    # 2. Temporal Divergence
    print("Generating Temporal Divergence Plot...")
    scaler = MinMaxScaler()
    daily_scaled = daily_df.copy()
    daily_scaled[["Enrolment", "Biometrics", "Demographics"]] = scaler.fit_transform(
        daily_scaled[["Enrolment", "Biometrics", "Demographics"]]
    )

    plt.figure(figsize=(14, 7))
    plt.plot(
        daily_scaled["date"], daily_scaled["Enrolment"], label="Enrolment", alpha=0.7
    )
    plt.plot(
        daily_scaled["date"],
        daily_scaled["Biometrics"],
        label="Biometrics",
        alpha=0.7,
        linestyle="--",
    )
    plt.plot(
        daily_scaled["date"],
        daily_scaled["Demographics"],
        label="Demographics",
        alpha=0.7,
        linestyle=":",
    )
    plt.title("Temporal Divergence Analysis (Normalized)")
    plt.legend()
    save_chart(
        "Adv_02_Temporal_Divergence.png",
        "Aligns disparate time series to detect misalignment.",
    )

    # 3. Lorenz Curve
    print("Generating District Contribution Curve...")

    def get_lorenz_curve(data):
        sorted_data = np.sort(data)
        cum_data = np.cumsum(sorted_data)
        return np.linspace(0, 1, len(cum_data)), cum_data / cum_data[-1]

    plt.figure(figsize=(10, 10))
    for col, style in zip(
        ["Enrolment", "Biometrics", "Demographics"], ["-", "--", ":"]
    ):
        x, y = get_lorenz_curve(merged_dist[merged_dist[col] > 0][col].values)
        plt.plot(x, y, label=col, linestyle=style, linewidth=2)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="-.", alpha=0.5, label="Equality")
    plt.title("District Contribution Curves (Lorenz)")
    plt.legend()
    save_chart(
        "Adv_03_Contribution_Curve.png",
        "Convexity indicates concentration of operational load.",
    )

    # 4. Rolling Volatility
    print("Generating Rolling Volatility Graph...")
    daily_vol = daily_df.copy()
    for col in ["Enrolment", "Biometrics", "Demographics"]:
        daily_vol[col + "_Vol"] = daily_vol[col].rolling(window=7).std()

    plt.figure(figsize=(14, 7))
    plt.plot(
        daily_vol["date"], daily_vol["Enrolment_Vol"], label="Enrolment", alpha=0.8
    )
    plt.plot(
        daily_vol["date"],
        daily_vol["Biometrics_Vol"],
        label="Biometrics",
        alpha=0.8,
        linestyle="--",
    )
    plt.plot(
        daily_vol["date"],
        daily_vol["Demographics_Vol"],
        label="Demographics",
        alpha=0.8,
        linestyle=":",
    )
    plt.title("Rolling Volatility Analysis (7-Day Window)")
    plt.legend()
    save_chart(
        "Adv_04_Rolling_Volatility.png",
        "High volatility indicates unstable operations.",
    )

    # 5. Cross-Scatter
    print("Generating Cross-Dataset Scatter...")
    scatter_data = merged_dist[
        (merged_dist["Enrolment"] > 100) & (merged_dist["Biometrics"] > 100)
    ]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=scatter_data,
        x="Enrolment",
        y="Biometrics",
        size="Demographics",
        sizes=(20, 200),
        alpha=0.6,
        color="b",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Cross-Dataset Scatter: Enrolment vs Biometrics")
    save_chart("Adv_05_Cross_Scatter.png", "Clusters show relationship breaks.")

    # 6. Isolation Forest
    print("Running Isolation Forest...")
    daily_iso = daily_df.copy().set_index("date")
    daily_iso["Bio_Ratio"] = daily_iso["Biometrics"] / (daily_iso["Enrolment"] + 1)
    daily_iso["Vol_Bio"] = daily_iso["Biometrics"].rolling(7).std().fillna(0)

    features = ["Enrolment", "Biometrics", "Demographics", "Bio_Ratio", "Vol_Bio"]
    X_iso = StandardScaler().fit_transform(daily_iso[features])

    daily_iso["anomaly"] = IsolationForest(
        contamination=0.05, random_state=42
    ).fit_predict(X_iso)
    anom_dates = daily_iso[daily_iso["anomaly"] == -1].index

    plt.figure(figsize=(14, 6))
    plt.plot(
        daily_iso.index,
        daily_iso["Biometrics"],
        color="gray",
        alpha=0.5,
        label="Biometrics",
    )
    plt.scatter(
        anom_dates,
        daily_iso.loc[anom_dates, "Biometrics"],
        color="red",
        s=50,
        label="Anomaly",
    )
    plt.title("Isolation Forest Anomaly Detection")
    plt.legend()
    save_chart("Adv_06_Isolation_Forest.png", "Multivariate anomaly detection.")

    # 7. DBSCAN
    print("Running DBSCAN...")
    X_db = StandardScaler().fit_transform(scatter_data[["Enrolment", "Biometrics"]])
    scatter_data["cluster"] = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_db)

    plt.figure(figsize=(10, 8))
    unique_labels = set(scatter_data["cluster"])
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        label = "Noise" if k == -1 else f"Cluster {k}"
        color = [0, 0, 0, 1] if k == -1 else tuple(col)
        mask = scatter_data["cluster"] == k
        plt.plot(
            scatter_data[mask]["Enrolment"],
            scatter_data[mask]["Biometrics"],
            "o",
            markerfacecolor=color,
            markeredgecolor="k",
            markersize=6 if k != -1 else 4,
            label=label,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.title("DBSCAN Clustering of Operational Patterns")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_chart("Adv_07_DBSCAN.png", "Spatial density clustering.")

    # 8. Individual DBSCAN
    print("Running Individual DBSCANs...")

    def perform_dbscan(df_dist, col, filename):
        state_means = df_dist.groupby("state")[col].transform("mean")
        X = np.log1p(df_dist[[col]].assign(State_Mean=state_means))
        X_scaled = StandardScaler().fit_transform(X)
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)

        plt.figure(figsize=(10, 8))
        plt.scatter(df_dist[col], state_means, c=labels, cmap="Spectral", alpha=0.6)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"DBSCAN: {col} vs State Mean")
        plt.xlabel(f"{col} Volume")
        plt.ylabel("State Average")
        save_chart(filename)

    perform_dbscan(
        merged_dist[merged_dist["Enrolment"] > 0].copy(),
        "Enrolment",
        "Adv_10_DBSCAN_Enrolment.png",
    )
    perform_dbscan(
        merged_dist[merged_dist["Biometrics"] > 0].copy(),
        "Biometrics",
        "Adv_11_DBSCAN_Biometrics.png",
    )
    perform_dbscan(
        merged_dist[merged_dist["Demographics"] > 0].copy(),
        "Demographics",
        "Adv_12_DBSCAN_Demographics.png",
    )

    print("\nAdvanced Analysis Complete.")


if __name__ == "__main__":
    main()
