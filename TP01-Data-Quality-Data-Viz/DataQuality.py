import pandas as pd
from datetime import datetime
import warnings

# Part 0: Preparation
# We ignore warnings so the output is cleaner and easier to read
warnings.simplefilter("ignore")

# Part 1: Load the dataset
# Some data may have errors or inconsistencies, like missing values or wrong date formats
df = pd.read_csv("jobs.csv", sep=";")  # Read CSV with ';' separator
df.columns = df.columns.str.strip()    # Remove spaces from column names to avoid errors

# Part 1a: Check for missing values
# We look for columns that have empty cells or NULL values ->If more than 80% of a column is missing, we consider it problematic
def missing_report(df):
    report = []
    for col in df.columns:
        missing_count = df[col].isna().sum() + (df[col] == "").sum()  # Count NaN and empty strings
        missing_rate = round(100 * missing_count / len(df), 2)         # Calculate percentage
        report.append({
            "Column": col,
            "MissingCount": missing_count,
            "MissingRate": missing_rate,
            "Problematic": missing_rate > 80  # True if more than 80% missing
        })
    return pd.DataFrame(report)

# Generate the missing values report
missing_df = missing_report(df)

# List of columns that are very problematic (>80% missing)
high_missing = missing_df[missing_df["Problematic"]]["Column"].tolist()

# Part 1b: Check date formatting
# Some columns should be dates, but the format may not be consistent
# We check how many values are valid and how many are invalid
date_cols = [
    "SIGNOFF_DATE", "SPECIAL_ACTION_DATE", "DOBRunDate", "Fully Permitted",
    "Approved", "Assigned", "Fully Paid", "Paid",
    "Pre- Filing Date", "Latest Action Date"
]
existing_dates = [c for c in date_cols if c in df.columns]  # Keep only columns that exist in the dataset

date_summary = []
for col in existing_dates:
    # Convert the column to datetime, invalid values become NaT
    conv = pd.to_datetime(df[col], errors="coerce")
    invalid_count = conv.isna().sum()
    total = len(df[col])
    date_summary.append({
        "Column": col,
        "ValidDates": total - invalid_count,
        "InvalidDates": invalid_count,
        "InvalidRate": round(100 * invalid_count / total, 2)  # Percentage of invalid dates
    })

date_summary_df = pd.DataFrame(date_summary)

# Part 2: Cleaning the data
# We make a copy of the original dataframe to clean it
clean_df = df.copy()

# Part 2a: Fix missing values
# For numeric columns, we fill missing values with the average
for col in clean_df.select_dtypes(include="number").columns:
    clean_df[col].fillna(clean_df[col].mean(), inplace=True)

# For categorical columns, we fill missing values with the most frequent value (mode)
for col in clean_df.select_dtypes(include="object").columns:
    if col not in high_missing:  # Skip very problematic columns
        mode_val = clean_df[col].mode()
        if not mode_val.empty:
            clean_df[col].replace("", pd.NA, inplace=True)     # Replace empty strings with NA
            clean_df[col].fillna(mode_val[0], inplace=True)   # Fill missing values with mode

# Part 2b: Remove very problematic columns (>80% missing)
clean_df.drop(columns=high_missing, inplace=True, errors='ignore')

# Part 2c: Fix date formatting
# Convert all date columns to the same format: YYYY-MM-DD HH:MM:SS
for col in existing_dates:
    if col in clean_df.columns:
        clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce")          # Convert to datetime
        clean_df[col].fillna(pd.Timestamp("1900-01-01 00:00:00"), inplace=True)  # Fill missing with default
        clean_df[col] = clean_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")          # Format dates uniformly

# Part 2d: Save the cleaned dataset
clean_df.to_csv("jobs_clean.csv", index=False)

# Part 3: Display the reports required by the TP
# Only display what is needed: problematic columns and date summary
print("\nColumns with more than 80% missing (Problematic columns):")
print(high_missing if high_missing else "None")

print("\nDate columns summary:")
print(date_summary_df.to_string(index=False))
