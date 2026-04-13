"""
============================================================
  Syntecxhub Internship — Data Science
  Week 2 | Project 1: Time Series & Category Charts
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ────────────────────────────────────────────────────────
# 0. SAMPLE DATA ՍՏԵՂԾՈՒՄ
# ────────────────────────────────────────────────────────
print("=" * 55)
print("  0. CREATING SAMPLE SALES DATASET")
print("=" * 55)

np.random.seed(42)

dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
categories = ["Electronics", "Clothing", "Food", "Sports", "Books"]

records = []
for date in dates:
    for cat in categories:
        base = {"Electronics": 1500, "Clothing": 800,
                "Food": 600, "Sports": 700, "Books": 300}[cat]
        sales = base + np.random.randint(-200, 300)
        records.append({"date": date, "category": cat, "sales": max(sales, 50)})

df = pd.DataFrame(records)
df["month"] = df["date"].dt.to_period("M")
df["quarter"] = df["date"].dt.to_period("Q")

print(f"Dataset: {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")
print(df.head(10).to_string())

os.makedirs("charts", exist_ok=True)

summary_lines = []

def save_summary(title, text):
    summary_lines.append(f"\n{'='*50}")
    summary_lines.append(f"  {title}")
    summary_lines.append(f"{'='*50}")
    summary_lines.append(text)


# ────────────────────────────────────────────────────────
# 1. LINE CHART — Daily Sales Over Time (per category)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  1. LINE CHART — Sales Over Time")
print("=" * 55)

fig, ax = plt.subplots(figsize=(14, 6))

colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

for i, cat in enumerate(categories):
    cat_df = df[df["category"] == cat].copy()
    # 7-day rolling average for smoother line
    cat_df = cat_df.sort_values("date")
    cat_df["rolling"] = cat_df["sales"].rolling(7).mean()
    ax.plot(cat_df["date"], cat_df["rolling"],
            label=cat, color=colors[i], linewidth=2)

ax.set_title("Daily Sales Over Time (7-Day Rolling Average)", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Sales ($)", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(title="Category", fontsize=10, title_fontsize=11,
          loc="upper left", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("charts/1_line_daily_sales.png", dpi=150)
plt.close()
print("Saved: charts/1_line_daily_sales.png")

save_summary("LINE CHART — Daily Sales Over Time",
    "Shows sales trend for each category over 2023.\n"
    "7-day rolling average smooths daily noise.\n"
    "Electronics consistently leads, Books lowest.\n"
    "Choice: Line chart best for continuous time-series data.")


# ────────────────────────────────────────────────────────
# 2. LINE CHART — Monthly Aggregation
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  2. LINE CHART — Monthly Aggregation")
print("=" * 55)

monthly = df.groupby(["month", "category"])["sales"].sum().reset_index()
monthly["month_str"] = monthly["month"].astype(str)

fig, ax = plt.subplots(figsize=(14, 6))

for i, cat in enumerate(categories):
    cat_df = monthly[monthly["category"] == cat]
    ax.plot(cat_df["month_str"], cat_df["sales"],
            marker="o", label=cat, color=colors[i], linewidth=2, markersize=5)

ax.set_title("Monthly Sales Aggregation by Category", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Total Sales ($)", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(title="Category", fontsize=10, title_fontsize=11, framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("charts/2_line_monthly_sales.png", dpi=150)
plt.close()
print("Saved: charts/2_line_monthly_sales.png")

save_summary("LINE CHART — Monthly Aggregation",
    "Aggregated daily sales into monthly totals per category.\n"
    "Markers on each month help read exact values.\n"
    "Choice: Monthly aggregation reduces noise, reveals trends.")


# ────────────────────────────────────────────────────────
# 3. BAR CHART — Quarterly Sales Aggregation
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  3. BAR CHART — Quarterly Aggregation")
print("=" * 55)

quarterly = df.groupby(["quarter", "category"])["sales"].sum().reset_index()
quarterly["quarter_str"] = quarterly["quarter"].astype(str)

quarters = quarterly["quarter_str"].unique()
x = np.arange(len(quarters))
width = 0.15

fig, ax = plt.subplots(figsize=(13, 6))

for i, cat in enumerate(categories):
    cat_df = quarterly[quarterly["category"] == cat]
    bars = ax.bar(x + i * width, cat_df["sales"], width=width,
                  label=cat, color=colors[i], alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1000,
                f"${h/1000:.0f}k", ha="center", va="bottom", fontsize=7.5)

ax.set_title("Quarterly Sales by Category", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Quarter", fontsize=12)
ax.set_ylabel("Total Sales ($)", fontsize=12)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(quarters, fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.legend(title="Category", fontsize=10, title_fontsize=11, framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("charts/3_bar_quarterly_sales.png", dpi=150)
plt.close()
print("Saved: charts/3_bar_quarterly_sales.png")

save_summary("BAR CHART — Quarterly Aggregation",
    "Grouped bar chart comparing all 5 categories per quarter.\n"
    "Value labels on top of each bar for easy reading.\n"
    "Choice: Grouped bars ideal for multi-category comparison.")


# ────────────────────────────────────────────────────────
# 4. BAR CHART — Category Comparison (Total Annual)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  4. BAR CHART — Category Comparison")
print("=" * 55)

total_by_cat = df.groupby("category")["sales"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(total_by_cat.index, total_by_cat.values,
              color=colors, alpha=0.9, edgecolor="white", linewidth=0.8)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 5000,
            f"${h:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title("Total Annual Sales by Category (2023)", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Category", fontsize=12)
ax.set_ylabel("Total Sales ($)", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("charts/4_bar_category_total.png", dpi=150)
plt.close()
print("Saved: charts/4_bar_category_total.png")

save_summary("BAR CHART — Category Comparison",
    "Simple bar chart ranking all categories by annual revenue.\n"
    "Sorted descending makes ranking immediately clear.\n"
    "Choice: Single bar chart best for direct ranking comparison.")


# ────────────────────────────────────────────────────────
# 5. PIE CHART — Market Share by Category
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  5. PIE CHART — Market Share")
print("=" * 55)

fig, ax = plt.subplots(figsize=(8, 8))

wedges, texts, autotexts = ax.pie(
    total_by_cat.values,
    labels=total_by_cat.index,
    autopct="%1.1f%%",
    colors=colors,
    startangle=140,
    pctdistance=0.82,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)

for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight("bold")
    autotext.set_color("white")

ax.set_title("Market Share by Category (2023)", fontsize=15,
             fontweight="bold", pad=20)
ax.legend(wedges, total_by_cat.index, title="Category",
          loc="lower right", fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig("charts/5_pie_market_share.png", dpi=150)
plt.close()
print("Saved: charts/5_pie_market_share.png")

save_summary("PIE CHART — Market Share",
    "Pie chart shows each category's share of total 2023 revenue.\n"
    "Electronics dominates at ~40%, Books smallest at ~8%.\n"
    "Choice: Pie chart ideal for part-to-whole proportion display.")


# ────────────────────────────────────────────────────────
# 6. SUMMARY REPORT
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  6. SAVING SUMMARY REPORT")
print("=" * 55)

report = []
report.append("TIME SERIES & CATEGORY CHARTS — SUMMARY REPORT")
report.append("Syntecxhub Internship | Week 2 | Project 1")
report.append("=" * 50)

report.append("\nDATASET OVERVIEW")
report.append(f"  Period     : 2023-01-01 to 2023-12-31")
report.append(f"  Categories : {', '.join(categories)}")
report.append(f"  Total rows : {len(df)}")

report.append("\nANNUAL SALES BY CATEGORY")
for cat, val in total_by_cat.items():
    pct = val / total_by_cat.sum() * 100
    report.append(f"  {cat:<15}: ${val:>12,.0f}  ({pct:.1f}%)")

report.append(f"\n  TOTAL      : ${total_by_cat.sum():>12,.0f}")

report.append("\nCHARTS GENERATED")
charts_info = [
    ("1_line_daily_sales.png",    "Line — Daily sales with 7-day rolling avg"),
    ("2_line_monthly_sales.png",  "Line — Monthly aggregation per category"),
    ("3_bar_quarterly_sales.png", "Bar  — Quarterly grouped comparison"),
    ("4_bar_category_total.png",  "Bar  — Annual total per category"),
    ("5_pie_market_share.png",    "Pie  — Market share by category"),
]
for fname, desc in charts_info:
    report.append(f"  {fname:<32} — {desc}")

report.append("\nCHART CHOICE DISCUSSION")
report.append("  Line chart : Best for time-series trends over continuous periods.")
report.append("  Bar chart  : Best for comparing discrete categories side by side.")
report.append("  Pie chart  : Best for showing part-to-whole proportions (≤6 slices).")
report.append("\nFORMATTING APPLIED")
report.append("  - Dollar-formatted Y-axis labels ($1,500 / $45k)")
report.append("  - Consistent color palette across all charts")
report.append("  - Value labels on bar chart tops")
report.append("  - Legend with category titles on every chart")
report.append("  - Grid lines (y-axis only) for readability")
report.append("  - Rotated x-axis labels for date readability")

for line in summary_lines:
    report.append(line)

with open("charts/summary_report.txt", "w") as f:
    f.write("\n".join(report))

print("Saved: charts/summary_report.txt")
print("\n--- SUMMARY ---")
for line in report[:25]:
    print(line)

print("\n" + "=" * 55)
print("  Time Series & Category Charts — COMPLETED ✓")
print("  All files saved in: ./charts/")
print("=" * 55)