import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def generate_migration_timeline():
    print("--- Loading Demographic Data ---")
    try:
        path = os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv")
        df = pd.read_csv(path)

        df.columns = df.columns.str.strip()
        df["district"] = df["district"].astype(str).str.title().str.strip()
        df["state"] = df["state"].astype(str).str.title().str.strip()

        df["date"] = pd.to_datetime(
            df["date"], errors="coerce", dayfirst=True, format="mixed"
        )
        df = df.dropna(subset=["date"])

        # Daily Velocity
        df["Total_Updates"] = df["demo_age_5_17"] + df["demo_age_17_"]

        daily_matrix = df.pivot_table(
            index="date", columns="district", values="Total_Updates", aggfunc="sum"
        ).fillna(0)
        rolling_velocity = daily_matrix.rolling(window=7).mean()

        # Top 5 Volatile Districts
        top_districts = (
            rolling_velocity.max().sort_values(ascending=False).head(5).index.tolist()
        )
        print(f"--- Top 5 High-Velocity Districts: {top_districts} ---")

        plot_data = rolling_velocity[top_districts].reset_index()
        plot_data_melted = plot_data.melt(
            id_vars="date", var_name="District", value_name="7-Day Avg Velocity"
        )

        plt.figure(figsize=(14, 7))
        sns.set_style("whitegrid")

        sns.lineplot(
            data=plot_data_melted,
            x="date",
            y="7-Day Avg Velocity",
            hue="District",
            linewidth=2.5,
            palette="bright",
        )

        max_val = plot_data_melted["7-Day Avg Velocity"].max()
        peak_row = plot_data_melted.loc[plot_data_melted["7-Day Avg Velocity"].idxmax()]

        plt.annotate(
            f"PEAK ANOMALY\n{peak_row['District']}: {int(max_val)} updates/day",
            xy=(peak_row["date"], max_val),
            xytext=(peak_row["date"], max_val + (max_val * 0.1)),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=10,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2),
        )

        plt.title("Migration Velocity Timeline (7-Day Moving Average)", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Updates per Day", fontsize=12)

        IMAGE_DIR = os.path.join(BASE_DIR, "image")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        output_path = os.path.join(IMAGE_DIR, "visual_C3_migration_velocity.png")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nSaved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    generate_migration_timeline()
