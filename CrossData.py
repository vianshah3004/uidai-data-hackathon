import pandas as pd
import os

# Configuration: File paths
BASE_DIR = "/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning"
FILES = {
    "Enrolment": "final_UIDAI_cleaned_enrol_data.csv",
    "Biometrics": "final_UIDAI_cleaned_biometrics_data.csv",
    "Demographics": "final_UIDAI_cleaned_demographic_data.csv",
}


def analyze_dataset(name, filepath):
    print(f"\n{'='*5} {name} Dataset {'='*5}")

    full_path = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(full_path):
        print(f"Error: File '{full_path}' not found.")
        return

    try:
        df = pd.read_csv(full_path)

        # 1. Date Range Analysis
        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"], format="mixed", dayfirst=True, errors="coerce"
            )
            min_date = df["date"].min()
            max_date = df["date"].max()
            print(
                f"Date Range: {min_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')}"
            )
            print(f"Duration: {(max_date - min_date).days} days")
        else:
            print("Date column missing.")

        # 2. Age Distribution Analysis
        print("\nAge Group Analysis:")

        # Identify age columns based on dataset type
        age_cols = []
        if name == "Enrolment":
            age_cols = ["age_0_5", "age_5_17", "age_18_greater"]
        elif name == "Biometrics":
            age_cols = ["bio_age_5_17", "bio_age_17_"]
        elif name == "Demographics":
            age_cols = ["demo_age_5_17", "demo_age_17_"]

        # Calculate sums and percentages
        if age_cols:
            total_vol = 0
            stats = {}

            for col in age_cols:
                if col in df.columns:
                    val = df[col].sum()
                    stats[col] = val
                    total_vol += val

            print(f"Total Volume: {total_vol:,}")

            for col, val in stats.items():
                pct = (val / total_vol * 100) if total_vol > 0 else 0
                # readable label
                label = (
                    col.replace("age_", "")
                    .replace("bio_", "")
                    .replace("demo_", "")
                    .replace("_", " ")
                )
                print(f"  - {label.title()}: {val:,} ({pct:.2f}%)")
        else:
            print("  No age columns defined for this dataset.")

        # 3. State-Level Analysis
        print("\nState-Level Analysis:")
        if "state" in df.columns:
            vol_cols = [c for c in age_cols if c in df.columns]
            if vol_cols:
                df["__vol"] = df[vol_cols].sum(axis=1)
                state_grp = (
                    df.groupby("state")["__vol"].sum().sort_values(ascending=False)
                )
                total_vol = state_grp.sum()

                print(f"Top 5 States:")
                for state, vol in state_grp.head(5).items():
                    pct = vol / total_vol * 100
                    print(f"  - {state}: {vol:,} ({pct:.2f}%)")

                top3_share = state_grp.head(3).sum() / total_vol * 100
                print(f"Insight: Top 3 States hold {top3_share:.1f}% of total load.")
            else:
                print("Cannot calculate volume for states (missing value columns).")

        # 4. District-Level Analysis
        print("\nDistrict-Level Analysis:")
        if "district" in df.columns and "state" in df.columns and vol_cols:
            dist_grp = (
                df.groupby(["state", "district"])["__vol"]
                .sum()
                .sort_values(ascending=False)
            )

            print(f"Top 5 Districts:")
            for (state, dist), vol in dist_grp.head(5).items():
                pct = vol / total_vol * 100
                lbl = f"{dist}, {state}"
                print(f"  - {lbl}: {vol:,} ({pct:.2f}%)")

        # 5. Monthly Trend (Seasonality)
        print("\nMonthly Trend Analysis:")
        if "date" in df.columns and vol_cols:
            df["__month"] = df["date"].dt.to_period("M")
            month_grp = df.groupby("__month")["__vol"].sum()

            if not month_grp.empty:
                peak_month = month_grp.idxmax()
                peak_val = month_grp.max()
                low_month = month_grp.idxmin()
                low_val = month_grp.min()

                print(f"  - Peak Month: {peak_month} (Vol: {peak_val:,})")
                print(f"  - Low Month : {low_month} (Vol: {low_val:,})")

                first = month_grp.iloc[0]
                last = month_grp.iloc[-1]
                trend = "Values Increased" if last > first else "Values Decreased"
                print(f"  - Overall Trend: {trend} (Start: {first:,} -> End: {last:,})")

        # Check for Gender columns
        print("\nGender Availability Check:")
        gender_keywords = ["male", "female", "gender", "sex"]
        found_gender = [
            c for c in df.columns if any(g in c.lower() for g in gender_keywords)
        ]
        if found_gender:
            print(f"Found Gender Columns: {found_gender}")
        else:
            print("No explicit Gender columns found.")

    except Exception as e:
        print(f"Error processing {name}: {e}")


if __name__ == "__main__":
    print("Generating Summary Report...")
    for name, path in FILES.items():
        analyze_dataset(name, path)
