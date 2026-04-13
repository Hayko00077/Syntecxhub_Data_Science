"""
============================================================
  Syntecxhub Internship — Data Science
  Project 3: Data Cleaning Utility
============================================================
"""

import pandas as pd
import numpy as np
import os

# ────────────────────────────────────────────────────────
# 0. DIRTY SAMPLE DATA ՍՏԵՂԾՈՒՄ
# ────────────────────────────────────────────────────────
print("=" * 55)
print("  0. CREATING DIRTY SAMPLE DATASET")
print("=" * 55)

dirty_data = {
    "Name":         ["Alice", "Bob", "charlie", "Diana", "Eve", "Bob",
                     "Frank", None, "Grace", "henry", "Iris", "Jack",
                     "alice", "Leo", None],
    "Age":          [25, 30, "twenty", 28, None, 30, 35, 42, 27, "31",
                     29, None, 25, 38, 44],
    "Salary":       [50000, 62000, 47000, None, 58000, 62000, 71000,
                     85000, None, 49000, 53000, 67000, 50000, 72000, 90000],
    "Department":   ["HR", "Engineering", "marketing", "Finance", None,
                     "Engineering", "Design", "HR", "Finance", "engineering",
                     "Marketing", "Design", "HR", None, "Finance"],
    "Join_Date":    ["2020-01-15", "2019/03/22", "March 5 2021", "2018-07-30",
                     "2022-11-01", "2019/03/22", "15-06-2017", "2016-09-10",
                     "2021-04-25", "2020-08-14", "invalid_date", "2017-12-01",
                     "2020-01-15", "2023-02-18", "2015-05-05"],
    "Rating":       [4.5, 3.8, 4.1, None, 3.5, 3.8, 4.7, 2.9, 4.0,
                     3.6, None, 4.3, 4.5, 3.9, 4.8],
    "Experience":   ["3", "5", "two", 4, None, "5", 8, 12, "3", 6,
                     4, "7", "3", 9, 15],
}

df_dirty = pd.DataFrame(dirty_data)
df_dirty.to_csv("dirty_employees.csv", index=False)
print("Created: dirty_employees.csv")
print(f"Shape: {df_dirty.shape[0]} rows, {df_dirty.shape[1]} columns")
print("\n--- DIRTY DATA PREVIEW ---")
print(df_dirty.to_string())


# ────────────────────────────────────────────────────────
# CLEANING LOG
# ────────────────────────────────────────────────────────
cleaning_log = []

def log(msg):
    print(f"  [LOG] {msg}")
    cleaning_log.append(msg)


# ────────────────────────────────────────────────────────
# 1. DETECT & HANDLE MISSING VALUES
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  1. MISSING VALUES")
print("=" * 55)

df = df_dirty.copy()

print("\n--- Missing values before cleaning ---")
missing = df.isnull().sum()
print(missing[missing > 0])

# Name — drop rows where Name is null
before = len(df)
df = df.dropna(subset=["Name"])
dropped = before - len(df)
log(f"Dropped {dropped} rows where 'Name' was null")

# Salary — fill with median
salary_median = pd.to_numeric(df["Salary"], errors="coerce").median()
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df["Salary"] = df["Salary"].fillna(salary_median)
log(f"Filled missing 'Salary' with median: {salary_median:,.0f}")

# Rating — fill with mean
rating_mean = pd.to_numeric(df["Rating"], errors="coerce").mean()
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Rating"] = df["Rating"].fillna(round(rating_mean, 1))
log(f"Filled missing 'Rating' with mean: {rating_mean:.2f}")

# Department — fill with mode
dept_mode = df["Department"].mode()[0]
df["Department"] = df["Department"].fillna(dept_mode)
log(f"Filled missing 'Department' with mode: '{dept_mode}'")

print("\n--- Missing values after cleaning ---")
print(df.isnull().sum())


# ────────────────────────────────────────────────────────
# 2. FIX INCORRECT DTYPES & PARSE DATES
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  2. FIX DTYPES & PARSE DATES")
print("=" * 55)

# Age — coerce non-numeric to NaN, then fill with median
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median).astype(int)
log(f"Fixed 'Age' dtype to int, filled invalid with median: {age_median:.0f}")

# Experience — coerce non-numeric to NaN, fill with median
df["Experience"] = pd.to_numeric(df["Experience"], errors="coerce")
exp_median = df["Experience"].median()
df["Experience"] = df["Experience"].fillna(exp_median).astype(int)
log(f"Fixed 'Experience' dtype to int, filled invalid with median: {exp_median:.0f}")

# Salary & Rating — ensure float
df["Salary"] = df["Salary"].astype(float)
df["Rating"] = df["Rating"].astype(float)
log("Ensured 'Salary' and 'Rating' are float")

# Join_Date — parse mixed formats
df["Join_Date"] = pd.to_datetime(df["Join_Date"], errors="coerce", dayfirst=False)
invalid_dates = df["Join_Date"].isnull().sum()
if invalid_dates > 0:
    date_mode = df["Join_Date"].mode()[0]
    df["Join_Date"] = df["Join_Date"].fillna(date_mode)
    log(f"Fixed {invalid_dates} invalid dates, filled with mode: {date_mode.date()}")
log("Parsed 'Join_Date' to datetime")

print("\n--- Data types after fixing ---")
print(df.dtypes)


# ────────────────────────────────────────────────────────
# 3. REMOVE DUPLICATES & STANDARDIZE COLUMN NAMES
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  3. DUPLICATES & STANDARDIZE")
print("=" * 55)

# Standardize Name & Department (title case, strip whitespace)
df["Name"] = df["Name"].str.strip().str.title()
df["Department"] = df["Department"].str.strip().str.title()
log("Standardized 'Name' and 'Department' to title case")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
dupes = before - len(df)
log(f"Removed {dupes} duplicate rows")

# Standardize column names (lowercase, replace space with _)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
log("Standardized column names to lowercase with underscores")

# Reset index
df = df.reset_index(drop=True)
log("Reset DataFrame index")

print(f"\nRows before: {before}  →  Rows after: {len(df)}")
print("\n--- Cleaned column names ---")
print(list(df.columns))


# ────────────────────────────────────────────────────────
# 4. OUTPUT CLEANED DATASET & CLEANING LOG
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  4. CLEANED DATASET & LOG")
print("=" * 55)

print("\n--- CLEANED DATA ---")
print(df.to_string())

# Save cleaned CSV
df.to_csv("cleaned_employees.csv", index=False)
print("\nSaved: cleaned_employees.csv")

# Save cleaning log
with open("cleaning_log.txt", "w") as f:
    f.write("DATA CLEANING LOG\n")
    f.write("=" * 40 + "\n")
    for i, entry in enumerate(cleaning_log, 1):
        f.write(f"{i}. {entry}\n")
print("Saved: cleaning_log.txt")

# Print log summary
print("\n--- CLEANING LOG SUMMARY ---")
for i, entry in enumerate(cleaning_log, 1):
    print(f"  {i}. {entry}")

print(f"\nOriginal shape : {df_dirty.shape}")
print(f"Cleaned shape  : {df.shape}")


# ────────────────────────────────────────────────────────
# CLEANUP
# ────────────────────────────────────────────────────────
for f in ["dirty_employees.csv", "cleaned_employees.csv", "cleaning_log.txt"]:
    if os.path.exists(f):
        os.remove(f)

print("\n" + "=" * 55)
print("  Data Cleaning Utility — COMPLETED ✓")
print("=" * 55)