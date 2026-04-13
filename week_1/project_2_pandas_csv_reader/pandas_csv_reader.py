"""
============================================================
  Syntecxhub Internship — Data Science
  Project 2: Pandas CSV Reader & Basic Analysis
============================================================
"""

import pandas as pd
import numpy as np
import os

# ────────────────────────────────────────────────────────
# 0. SAMPLE CSV ՍՏԵՂԾՈՒՄ
# ────────────────────────────────────────────────────────
print("=" * 55)
print("  0. CREATING SAMPLE CSV")
print("=" * 55)

np.random.seed(42)
n = 50

data = {
    "name": [
        "Alice", "Bob", "Charlie", "Diana", "Eve",
        "Frank", "Grace", "Henry", "Iris", "Jack"
    ] * 5,
    "age": np.random.randint(22, 55, n),
    "department": np.random.choice(["HR", "Engineering", "Marketing", "Finance", "Design"], n),
    "salary": np.random.randint(35000, 120000, n),
    "experience_years": np.random.randint(1, 20, n),
    "rating": np.round(np.random.uniform(2.5, 5.0, n), 1),
    "join_date": pd.date_range(start="2015-01-01", periods=n, freq="ME").strftime("%Y-%m-%d"),
}

df_sample = pd.DataFrame(data)
df_sample.to_csv("employees.csv", index=False)
df_sample.to_excel("employees.xlsx", index=False)
print("Created: employees.csv and employees.xlsx")


# ────────────────────────────────────────────────────────
# 1. READ CSV / EXCEL INTO DATAFRAME
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  1. READ CSV / EXCEL INTO DATAFRAME")
print("=" * 55)

df = pd.read_csv("employees.csv")
df_excel = pd.read_excel("employees.xlsx")

print(f"CSV loaded    : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Excel loaded  : {df_excel.shape[0]} rows, {df_excel.shape[1]} columns")


# ────────────────────────────────────────────────────────
# 2. HEAD / TAIL / DTYPES / INFO
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  2. HEAD / TAIL / TYPES / INFO")
print("=" * 55)

print("\n--- HEAD (first 5 rows) ---")
print(df.head())

print("\n--- TAIL (last 5 rows) ---")
print(df.tail())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- SHAPE ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- INFO ---")
df.info()


# ────────────────────────────────────────────────────────
# 3. SUMMARY STATISTICS (mean, median, min, max, count)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  3. SUMMARY STATISTICS")
print("=" * 55)

print("\n--- DESCRIBE (all numeric columns) ---")
print(df.describe().round(2))

print("\n--- SALARY stats ---")
print(f"Mean   : ${df['salary'].mean():,.2f}")
print(f"Median : ${df['salary'].median():,.2f}")
print(f"Min    : ${df['salary'].min():,}")
print(f"Max    : ${df['salary'].max():,}")
print(f"Count  : {df['salary'].count()}")

print("\n--- AGE stats ---")
print(f"Mean   : {df['age'].mean():.1f}")
print(f"Median : {df['age'].median():.1f}")
print(f"Min    : {df['age'].min()}")
print(f"Max    : {df['age'].max()}")

print("\n--- RATING stats ---")
print(f"Mean   : {df['rating'].mean():.2f}")
print(f"Median : {df['rating'].median():.2f}")

print("\n--- DEPARTMENT counts ---")
print(df['department'].value_counts())


# ────────────────────────────────────────────────────────
# 4. FILTER ROWS / SELECT COLUMNS / SLICE SUBSETS
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  4. FILTER / SELECT / SLICE")
print("=" * 55)

# Filter rows
high_salary = df[df['salary'] > 90000]
print(f"\n--- High salary (>$90,000): {len(high_salary)} employees ---")
print(high_salary[['name', 'department', 'salary']].head())

senior = df[df['experience_years'] >= 10]
print(f"\n--- Senior employees (10+ years): {len(senior)} ---")
print(senior[['name', 'experience_years', 'salary']].head())

eng_high = df[(df['department'] == 'Engineering') & (df['salary'] > 80000)]
print(f"\n--- Engineering + salary>$80k: {len(eng_high)} employees ---")
print(eng_high[['name', 'salary', 'rating']])

# Select columns
print("\n--- Selected columns (name, department, salary) ---")
print(df[['name', 'department', 'salary']].head(5))

# Slice subsets
print("\n--- Slice rows 10 to 15 ---")
print(df.iloc[10:16])

print("\n--- loc: Engineering department ---")
print(df.loc[df['department'] == 'Engineering', ['name', 'salary', 'rating']].head(5))

# Group by
print("\n--- Average salary by department ---")
print(df.groupby('department')['salary'].mean().round(2).sort_values(ascending=False))

print("\n--- Average rating by department ---")
print(df.groupby('department')['rating'].mean().round(2).sort_values(ascending=False))


# ────────────────────────────────────────────────────────
# 5. SAVE FILTERED RESULTS TO CSV / EXCEL
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  5. SAVE FILTERED RESULTS")
print("=" * 55)

# Save high salary filter
high_salary.to_csv("high_salary_employees.csv", index=False)
print("Saved: high_salary_employees.csv")

# Save senior employees
senior.to_excel("senior_employees.xlsx", index=False)
print("Saved: senior_employees.xlsx")

# Save department summary
dept_summary = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    avg_rating=('rating', 'mean'),
    count=('name', 'count')
).round(2).reset_index()

dept_summary.to_csv("department_summary.csv", index=False)
print("Saved: department_summary.csv")

print("\n--- Department Summary ---")
print(dept_summary)


# ────────────────────────────────────────────────────────
# CLEANUP
# ────────────────────────────────────────────────────────
for f in ["employees.csv", "employees.xlsx",
          "high_salary_employees.csv",
          "senior_employees.xlsx",
          "department_summary.csv"]:
    if os.path.exists(f):
        os.remove(f)

print("\n" + "=" * 55)
print("  Pandas CSV Reader & Analysis — COMPLETED ✓")
print("=" * 55)