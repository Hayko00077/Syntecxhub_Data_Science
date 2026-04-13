"""
============================================================
  Syntecxhub Internship — Data Science
  Week 2 | Project 2: Statistical Plots & Distribution Analysis
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

os.makedirs("plots", exist_ok=True)

# ────────────────────────────────────────────────────────
# 0. SAMPLE DATA ՍՏԵՂԾՈՒՄ
# ────────────────────────────────────────────────────────
print("=" * 55)
print("  0. CREATING SAMPLE DATASET")
print("=" * 55)

np.random.seed(42)
n = 300

# Region A — normal distribution, higher salaries
region_a_salary      = np.random.normal(loc=65000, scale=10000, size=n)
region_a_age         = np.random.normal(loc=35, scale=6, size=n)
region_a_experience  = np.random.normal(loc=8, scale=3, size=n).clip(0)
region_a_rating      = np.random.normal(loc=3.8, scale=0.5, size=n).clip(1, 5)

# Region B — right-skewed distribution, wider spread
region_b_salary      = np.random.exponential(scale=45000, size=n) + 30000
region_b_age         = np.random.normal(loc=40, scale=9, size=n)
region_b_experience  = np.random.normal(loc=11, scale=5, size=n).clip(0)
region_b_rating      = np.random.normal(loc=3.4, scale=0.8, size=n).clip(1, 5)

# Add outliers to Region B salary
outliers = np.array([180000, 200000, 220000, 250000, 300000])
region_b_salary = np.concatenate([region_b_salary, outliers])
region_a_salary = np.concatenate([region_a_salary, np.random.normal(65000, 5000, 5)])

extra_5 = np.zeros(5)

df = pd.DataFrame({
    "salary":     np.concatenate([region_a_salary,     region_b_salary]),
    "age":        np.concatenate([region_a_age,        extra_5, region_b_age, extra_5]),
    "experience": np.concatenate([region_a_experience, extra_5, region_b_experience, extra_5]),
    "rating":     np.concatenate([region_a_rating,     extra_5, region_b_rating,     extra_5]),
    "region":     ["Region A"] * (n + 5) + ["Region B"] * (n + 5),
})

# Fill the zero placeholders with column medians
for col in ["age", "experience", "rating"]:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

df["salary"] = df["salary"].clip(20000, 350000)
df = df.reset_index(drop=True)

print(f"Dataset: {len(df)} rows")
print(df.groupby("region")[["salary", "age", "experience", "rating"]].describe().round(2))

interpretations = []


# ────────────────────────────────────────────────────────
# 1. HISTOGRAM — Salary Distribution
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  1. HISTOGRAM — Salary Distribution")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Histogram — Salary Distribution by Region", fontsize=15, fontweight="bold")

for ax, (region, color) in zip(axes, [("Region A", "#2196F3"), ("Region B", "#E91E63")]):
    data = df[df["region"] == region]["salary"]
    ax.hist(data, bins=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.axvline(data.mean(),   color="black",  linestyle="--", linewidth=1.5, label=f"Mean: ${data.mean():,.0f}")
    ax.axvline(data.median(), color="orange", linestyle="-",  linewidth=1.5, label=f"Median: ${data.median():,.0f}")
    ax.set_title(region, fontsize=13, fontweight="bold")
    ax.set_xlabel("Salary ($)", fontsize=11)
    ax.set_ylabel("Frequency",  fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    print(f"{region} — Mean: ${data.mean():,.0f} | Median: ${data.median():,.0f} | Skew: {skew:.2f} | Kurt: {kurt:.2f}")

plt.tight_layout()
plt.savefig("plots/1_histogram_salary.png", dpi=150)
plt.close()
print("Saved: plots/1_histogram_salary.png")

interpretations.append(
    "HISTOGRAM — Salary:\n"
    "Region A salary follows a near-normal distribution (skew ≈ 0), with mean and median\n"
    "closely aligned around $65k. Region B shows a strong right skew due to high-value\n"
    "outliers ($180k–$300k), pulling the mean well above the median. Region B has greater\n"
    "spread and inequality in compensation."
)


# ────────────────────────────────────────────────────────
# 2. KDE — Salary & Rating
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  2. KDE — Salary & Rating")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("KDE — Density Curves: Salary & Rating", fontsize=15, fontweight="bold")

for col, ax, xlabel in zip(
    ["salary", "rating"],
    axes,
    ["Salary ($)", "Rating (1–5)"]
):
    for region, color in [("Region A", "#2196F3"), ("Region B", "#E91E63")]:
        data = df[df["region"] == region][col].dropna()
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, label=region)
        ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)

    ax.set_title(col.capitalize(), fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density",  fontsize=11)
    if col == "salary":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/2_kde_salary_rating.png", dpi=150)
plt.close()
print("Saved: plots/2_kde_salary_rating.png")

interpretations.append(
    "KDE — Salary & Rating:\n"
    "The KDE plot confirms Region A has a tight, symmetric salary distribution with a single\n"
    "peak near $65k. Region B's curve is flatter and right-tailed, indicating high variability.\n"
    "For ratings, Region A peaks higher (~3.8) with a narrower spread, while Region B is\n"
    "lower (~3.4) and more dispersed, suggesting more inconsistent performance evaluations."
)


# ────────────────────────────────────────────────────────
# 3. BOXPLOT — Compare Distributions Across Groups
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  3. BOXPLOT — Group Comparison")
print("=" * 55)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Boxplot — Distribution Comparison: Region A vs Region B",
             fontsize=15, fontweight="bold")

cols    = ["salary",    "experience",    "rating"]
labels  = ["Salary ($)", "Experience (yrs)", "Rating (1–5)"]
colors  = ["#2196F3",   "#4CAF50",       "#FF9800"]

for ax, col, label, color in zip(axes, cols, labels, colors):
    data_a = df[df["region"] == "Region A"][col].dropna()
    data_b = df[df["region"] == "Region B"][col].dropna()

    bp = ax.boxplot(
        [data_a, data_b],
        labels=["Region A", "Region B"],
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5, markerfacecolor=color)
    )
    bp["boxes"][0].set_facecolor("#2196F380")
    bp["boxes"][1].set_facecolor("#E91E6380")

    ax.set_title(col.capitalize(), fontsize=13, fontweight="bold")
    ax.set_ylabel(label, fontsize=11)
    if col == "salary":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Print outlier count
    q1_b, q3_b = data_b.quantile(0.25), data_b.quantile(0.75)
    iqr_b = q3_b - q1_b
    outliers_b = data_b[(data_b < q1_b - 1.5 * iqr_b) | (data_b > q3_b + 1.5 * iqr_b)]
    print(f"{col.upper()} — Region B outliers: {len(outliers_b)}")
    if len(outliers_b) > 0:
        print(f"  Outlier values: {sorted(outliers_b.values)[:5]} ...")

plt.tight_layout()
plt.savefig("plots/3_boxplot_comparison.png", dpi=150)
plt.close()
print("Saved: plots/3_boxplot_comparison.png")

interpretations.append(
    "BOXPLOT — Group Comparison:\n"
    "Boxplots clearly show Region B has a higher median salary but much wider IQR and\n"
    "several extreme outliers (up to $300k), visible as isolated dots above the whisker.\n"
    "Region A salary is compact with no significant outliers — a more uniform pay structure.\n"
    "Experience is higher on average in Region B, while ratings show Region A employees\n"
    "score slightly more consistently."
)


# ────────────────────────────────────────────────────────
# 4. OUTLIER DETECTION — IQR Method
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  4. OUTLIER DETECTION (IQR Method)")
print("=" * 55)

fig, ax = plt.subplots(figsize=(10, 5))

for region, color, pos in [("Region A", "#2196F3", 1), ("Region B", "#E91E63", 2)]:
    data = df[df["region"] == region]["salary"]
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)]
    normal   = data[(data >= lower) & (data <= upper)]

    ax.scatter([pos] * len(normal),   normal.values,   alpha=0.2, color=color,   s=15, label=f"{region} normal" if pos == 1 else "")
    ax.scatter([pos] * len(outliers), outliers.values, alpha=0.9, color="red",   s=50, marker="X", label=f"{region} outliers ({len(outliers)})" if pos == 2 else f"{region} outliers ({len(outliers)})")

    print(f"{region}: Q1=${q1:,.0f} | Q3=${q3:,.0f} | IQR=${iqr:,.0f} | Lower=${lower:,.0f} | Upper=${upper:,.0f} | Outliers={len(outliers)}")

ax.set_title("Outlier Detection — Salary (IQR Method)", fontsize=14, fontweight="bold")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Region A", "Region B"], fontsize=12)
ax.set_ylabel("Salary ($)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/4_outlier_detection.png", dpi=150)
plt.close()
print("Saved: plots/4_outlier_detection.png")

interpretations.append(
    "OUTLIER DETECTION — IQR Method:\n"
    "Using the IQR rule (beyond 1.5×IQR from Q1/Q3), Region B contains multiple salary\n"
    "outliers (marked as red X), all on the upper end — confirming a right-skewed,\n"
    "high-variance distribution. Region A has no meaningful outliers, reinforcing its\n"
    "normal, symmetric shape. These outliers in Region B likely represent senior executives\n"
    "or specialized roles commanding premium compensation."
)


# ────────────────────────────────────────────────────────
# 5. SKEWNESS & SPREAD SUMMARY PLOT
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  5. SKEWNESS & SPREAD SUMMARY")
print("=" * 55)

metrics = {}
for region in ["Region A", "Region B"]:
    data = df[df["region"] == region]["salary"]
    metrics[region] = {
        "Mean":     data.mean(),
        "Median":   data.median(),
        "Std Dev":  data.std(),
        "Skewness": stats.skew(data),
        "Kurtosis": stats.kurtosis(data),
    }
    print(f"\n{region}:")
    for k, v in metrics[region].items():
        print(f"  {k:<12}: {v:>10.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Skewness & Spread — Region Comparison", fontsize=14, fontweight="bold")

stat_names = ["Mean", "Median", "Std Dev", "Skewness", "Kurtosis"]
x = np.arange(len(stat_names))
width = 0.35

for ax_idx, (ax, col, scale) in enumerate(zip(
    axes,
    ["salary", "rating"],
    [1000, 1]
)):
    for i, (region, color) in enumerate([("Region A", "#2196F3"), ("Region B", "#E91E63")]):
        data = df[df["region"] == region][col].dropna()
        vals = [
            data.mean()   / scale,
            data.median() / scale,
            data.std()    / scale,
            stats.skew(data),
            stats.kurtosis(data),
        ]
        bars = ax.bar(x + i * width, vals, width, label=region, color=color, alpha=0.8)

    ax.set_title(col.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(stat_names, rotation=15, fontsize=10)
    suffix = " (÷1000)" if scale == 1000 else ""
    ax.set_ylabel(f"Value{suffix}", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("plots/5_skewness_spread.png", dpi=150)
plt.close()
print("Saved: plots/5_skewness_spread.png")


# ────────────────────────────────────────────────────────
# 6. EXPORT INTERPRETATION REPORT
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  6. EXPORTING INTERPRETATION REPORT")
print("=" * 55)

report = []
report.append("STATISTICAL PLOTS & DISTRIBUTION ANALYSIS — REPORT")
report.append("Syntecxhub Internship | Week 2 | Project 2")
report.append("=" * 55)
report.append(f"\nDataset: {len(df)} employees | 2 regions | 4 numeric features\n")

report.append("PLOTS GENERATED")
plots_info = [
    ("1_histogram_salary.png",  "Histogram  — Salary distribution per region"),
    ("2_kde_salary_rating.png", "KDE        — Density curves for salary & rating"),
    ("3_boxplot_comparison.png","Boxplot    — Multi-feature group comparison"),
    ("4_outlier_detection.png", "Scatter    — Outlier detection via IQR method"),
    ("5_skewness_spread.png",   "Bar        — Skewness & spread statistics"),
]
for fname, desc in plots_info:
    report.append(f"  {fname:<32} — {desc}")

report.append("\nSTATISTICAL SUMMARY — SALARY")
for region in ["Region A", "Region B"]:
    data = df[df["region"] == region]["salary"]
    report.append(f"\n  {region}:")
    report.append(f"    Mean     : ${data.mean():>10,.0f}")
    report.append(f"    Median   : ${data.median():>10,.0f}")
    report.append(f"    Std Dev  : ${data.std():>10,.0f}")
    report.append(f"    Skewness :  {stats.skew(data):>10.3f}")
    report.append(f"    Kurtosis :  {stats.kurtosis(data):>10.3f}")

report.append("\nINTERPRETATIONS")
for interp in interpretations:
    report.append(f"\n{interp}")

report.append("\n" + "=" * 55)
report.append("CONCLUSION")
report.append("=" * 55)
report.append(
    "Region A exhibits a normal, symmetric salary distribution with low variance and\n"
    "no significant outliers — typical of a structured pay scale. Region B shows\n"
    "right-skewed salary data with high kurtosis and multiple upper-end outliers,\n"
    "suggesting a more heterogeneous workforce with wide compensation gaps.\n"
    "Histogram + KDE are best for shape inspection; Boxplot for quick outlier\n"
    "detection and group comparison; IQR method provides a robust statistical\n"
    "framework for identifying anomalies."
)

with open("plots/interpretation_report.txt", "w") as f:
    f.write("\n".join(report))

print("Saved: plots/interpretation_report.txt")
print("\n--- CONCLUSION ---")
print(report[-1])

print("\n" + "=" * 55)
print("  Statistical Plots & Distribution Analysis — COMPLETED ✓")
print("  All files saved in: ./plots/")
print("=" * 55)