import pandas as pd
import os

# defining the base directory
BASE_DIR = '/Users/vineetkumarshah/Downloads/api_data_aadhar/finalcleaning'

# loading the demographic data
df = pd.read_csv(os.path.join(BASE_DIR, "final_UIDAI_cleaned_demographic_data.csv"))
print(df.duplicated().sum())
print(df["district"].unique())
print(len(df["district"].unique()))
print(df["state"].unique())
print(len(df["state"].unique()))
print(df.isna().sum())

# choosing columns to check for repeats
match_cols = ["date", "state", "pincode", "district"]

# finding all the repeated rows
# making sure we catch every copy
all_duplicates = df[df.duplicated(subset=match_cols, keep=False)]

if not all_duplicates.empty:
    # sorting so duplicates appear next to each other
    all_duplicates = all_duplicates.sort_values(by=match_cols)

    print("--- ALL DUPLICATE INSTANCES ---")
    print(all_duplicates)
    print("\n" + "=" * 50 + "\n")

    # counting how many times each set repeats
    duplicate_counts = (
        all_duplicates.groupby(match_cols).size().reset_index(name="occurrence_count")
    )

    print("--- COUNTS PER COMBINATION ---")
    print(duplicate_counts)
    print("\n" + "=" * 50 + "\n")

    # summarizing the total duplicates found
    total_duplicate_rows = len(all_duplicates)
    total_unique_combinations = len(duplicate_counts)

    print(f"Total rows that are part of a duplicate set: {total_duplicate_rows}")
    print(
        f"Total unique combinations that have duplicates: {total_unique_combinations}"
    )

else:
    print("No duplicates found for the specified columns.")
