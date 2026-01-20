import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os

BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"


def main():
    print("Loading Demographic Data...")
    try:
        path = os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv")
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(df["date"])
        df["total_updates"] = df["demo_age_5_17"] + df["demo_age_17_"]
        df["month_year"] = df["date"].dt.to_period("M")

        # Aggregate
        state_monthly = (
            df.groupby(["month_year", "state"])["total_updates"].sum().reset_index()
        )
        state_monthly["month_str"] = state_monthly["month_year"].astype(str)

        pivot_data = state_monthly.pivot(
            index="state", columns="month_str", values="total_updates"
        ).fillna(0)

        plt.figure(figsize=(16, 12))
        ax = sns.heatmap(
            pivot_data,
            cmap="YlGnBu",
            annot=False,
            fmt=".0f",
            cbar_kws={"label": "Total Update Counts"},
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )

        plt.title(
            "Monthly Demographic Update Counts by State",
            fontsize=20,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Month", fontsize=14)
        plt.ylabel("State", fontsize=14)

        IMAGE_DIR = os.path.join(BASE_DIR, "image")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        output_path = os.path.join(IMAGE_DIR, "heatmap_all_states_integer_counts.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
