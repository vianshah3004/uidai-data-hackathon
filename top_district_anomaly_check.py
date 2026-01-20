import pandas as pd
import matplotlib.pyplot as plt
import os

# standard plot styling
plt.style.use("seaborn-v0_8-darkgrid")

# config paths
BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"
ENROL_FILE = os.path.join(BASE_DIR, "final_UIDAI_cleaned_enrol_data.csv")
DEMO_FILE = os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv")
BIO_FILE = os.path.join(BASE_DIR, "final_UIDAI_cleaned_biometrics_data.csv")


def load_data():
    """Load the CSVs."""
    print("Loading datasets...")
    enrol_df = pd.read_csv(ENROL_FILE)
    demo_df = pd.read_csv(DEMO_FILE)
    bio_df = pd.read_csv(BIO_FILE)
    return enrol_df, demo_df, bio_df


def process_data(df, dataset_name, count_cols):
    """
    Find top states and districts. If a top district isn't in a top state, flag it.
    """
    print(f"Processing {dataset_name}...")

    # sum up the relevant columns
    df["total_count"] = df[count_cols].sum(axis=1)

    # get the top 10 states by volume
    state_counts = df.groupby("state")["total_count"].sum().sort_values(ascending=False)
    top_10_states = state_counts.head(10).index.tolist()
    print(f"Top 10 States ({dataset_name}): {top_10_states}")

    # get the top 10 districts
    district_counts = (
        df.groupby(["state", "district"])["total_count"].sum().reset_index()
    )
    district_counts = district_counts.sort_values(
        by="total_count", ascending=False
    ).head(10)

    # check for anomalies
    anomalies = []
    colors = []

    print(f"Top 10 Districts ({dataset_name}):")
    for index, row in district_counts.iterrows():
        district = row["district"]
        state = row["state"]
        if state not in top_10_states:
            print(
                f"  - ANOMALY: {district} ({state}) - District in top 10, but State is NOT."
            )
            anomalies.append(district)
            colors.append("#e74c3c")  # red for anomaly
        else:
            print(f"  - Normal: {district} ({state})")
            colors.append("#3498db")  # blue for normal

    # plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # subplot 1: top states
    top_states_counts = state_counts.head(10).sort_values(ascending=True)
    ax1.barh(top_states_counts.index, top_states_counts.values, color="#2ecc71")
    ax1.set_xlabel("Total Count", fontsize=12)
    ax1.set_title(f"Top 10 States - {dataset_name}", fontsize=16)

    for i, v in enumerate(top_states_counts.values):
        ax1.text(v, i, f" {v:,.0f}", va="center", fontweight="bold")

    # subplot 2: top districts with anomaly coloring
    ax2.barh(district_counts["district"], district_counts["total_count"], color=colors)
    ax2.set_xlabel("Total Count", fontsize=12)
    ax2.set_title(f"Top 10 Districts - {dataset_name} (Red = Anomaly)", fontsize=16)
    ax2.invert_yaxis()

    for i, v in enumerate(district_counts["total_count"]):
        ax2.text(v, i, f" {v:,.0f}", va="center", fontweight="bold")

    IMAGE_DIR = os.path.join(BASE_DIR, "image")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    output_file = os.path.join(IMAGE_DIR, f"{dataset_name.lower()}_top_10_analysis.png")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved to: {output_file}")
    plt.close()


def main():
    try:
        enrol_df, demo_df, bio_df = load_data()

        # process enrollment
        process_data(enrol_df, "Enrollment", ["age_0_5", "age_5_17", "age_18_greater"])

        # process demographics
        process_data(demo_df, "Demographics", ["demo_age_5_17", "demo_age_17_"])

        # process biometrics
        process_data(bio_df, "Biometrics", ["bio_age_5_17", "bio_age_17_"])

        print("\nAnalysis complete.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
