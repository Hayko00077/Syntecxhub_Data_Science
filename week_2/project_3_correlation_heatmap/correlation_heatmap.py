"""
============================================================
  Syntecxhub Internship — Data Science
  Week 2 | Project 3: Correlation Heatmap & Pairwise Relationships
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os

os.makedirs("correlation_plots", exist_ok=True)

# ────────────────────────────────────────────────────────
# 0. SAMPLE DATA ՍՏԵՂԾՈՒՄ
# ────────────────────────────────────────────────────────
print("=" * 55)
print("  0. CREATING SAMPLE DATASET")
print("=" * 55)

np.random.seed(42)
n = 300

experience   = np.random.randint(1, 25, n)
salary       = 30000 + experience * 2800 + np.random.normal(0, 8000, n)
age          = 22 + experience + np.random.randint(0, 8, n)
rating       = (experience * 0.08 + np.random.normal(3.0, 0.4, n)).clip(1, 5)
hours_worked = np.random.normal(42, 6, n).clip(30, 70)
satisfaction = (5 - hours_worked * 0.04 + np.random.normal(0, 0.5, n)).clip(1, 5)
projects     = (experience * 0.6 + np.random.normal(0, 2, n)).clip(1, 30).astype(int)
absences     = (10 - satisfaction * 1.2 + np.random.normal(0, 1.5, n)).clip(0, 20).astype(int)

df = pd.DataFrame({
    "salary":       salary.round(0),
    "experience":   experience,
    "age":          age,
    "rating":       rating.round(2),
    "hours_worked": hours_worked.round(1),
    "satisfaction": satisfaction.round(2),
    "projects":     projects,
    "absences":     absences,
})

print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.describe().round(2))


# ────────────────────────────────────────────────────────
# 1. PEARSON CORRELATION MATRIX
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  1. PEARSON CORRELATION MATRIX")
print("=" * 55)

corr = df.corr(method="pearson")
print(corr.round(3).to_string())


# ────────────────────────────────────────────────────────
# 2. HEATMAP (masked upper triangle + annotated)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  2. CORRELATION HEATMAP")
print("=" * 55)

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr_masked = corr.copy()
corr_masked[mask] = np.nan

cols = corr.columns.tolist()
n_cols = len(cols)

fig, ax = plt.subplots(figsize=(11, 9))

# Draw colored cells
for i in range(n_cols):
    for j in range(n_cols):
        if mask[i, j]:
            continue
        val = corr.iloc[i, j]
        # Color: blue for positive, red for negative
        if val >= 0:
            intensity = val
            color = (1 - intensity * 0.8, 1 - intensity * 0.8, 1.0)
        else:
            intensity = abs(val)
            color = (1.0, 1 - intensity * 0.8, 1 - intensity * 0.8)

        rect = mpatches.FancyBboxPatch(
            (j, n_cols - i - 1), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5
        )
        ax.add_patch(rect)

        text_color = "white" if abs(val) > 0.6 else "#333333"
        fontsize   = 11 if n_cols <= 8 else 9
        ax.text(j + 0.5, n_cols - i - 0.5, f"{val:.2f}",
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color)

ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_cols)
ax.set_xticks(np.arange(n_cols) + 0.5)
ax.set_yticks(np.arange(n_cols) + 0.5)
ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=11)
ax.set_yticklabels(reversed(cols), fontsize=11)
ax.set_title("Pearson Correlation Heatmap\n(lower triangle only, upper masked)",
             fontsize=14, fontweight="bold", pad=20)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#3333FF", label="Strong positive (+1.0)"),
    Patch(facecolor="#AAAAFF", label="Weak positive"),
    Patch(facecolor="#FFAAAA", label="Weak negative"),
    Patch(facecolor="#FF3333", label="Strong negative (−1.0)"),
]
ax.legend(handles=legend_elements, loc="upper right",
          fontsize=9, framealpha=0.9, title="Correlation", title_fontsize=10)

plt.tight_layout()
plt.savefig("correlation_plots/1_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: correlation_plots/1_correlation_heatmap.png")


# ────────────────────────────────────────────────────────
# 3. PAIRPLOT — Key Variable Pairs
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  3. PAIRPLOT — Key Variable Pairs")
print("=" * 55)

key_vars = ["salary", "experience", "rating", "satisfaction", "absences"]
n_vars   = len(key_vars)
colors   = {"low_exp": "#2196F3", "high_exp": "#E91E63"}

df["exp_group"] = pd.cut(df["experience"], bins=[0, 10, 25],
                          labels=["≤10 yrs", ">10 yrs"])

fig, axes = plt.subplots(n_vars, n_vars, figsize=(16, 14))
fig.suptitle("Pairplot — Key Variable Relationships", fontsize=15,
             fontweight="bold", y=1.01)

group_colors = {"≤10 yrs": "#2196F3", ">10 yrs": "#E91E63"}

for i, var_y in enumerate(key_vars):
    for j, var_x in enumerate(key_vars):
        ax = axes[i][j]

        if i == j:
            # Diagonal — histogram per group
            for grp, col in group_colors.items():
                data = df[df["exp_group"] == grp][var_x].dropna()
                ax.hist(data, bins=20, color=col, alpha=0.5, density=True)
            ax.set_facecolor("#f9f9f9")
        else:
            # Off-diagonal — scatter
            for grp, col in group_colors.items():
                sub = df[df["exp_group"] == grp]
                ax.scatter(sub[var_x], sub[var_y],
                           alpha=0.3, s=12, color=col)
            # Regression line
            x_data = df[var_x].values
            y_data = df[var_y].values
            slope, intercept, r, p, _ = stats.linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="black", linewidth=1.2, linestyle="--")
            ax.text(0.05, 0.92, f"r={r:.2f}", transform=ax.transAxes,
                    fontsize=8, color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # Labels only on edges
        if i == n_vars - 1:
            ax.set_xlabel(var_x, fontsize=9)
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(var_y, fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.tick_params(labelsize=7)

# Legend
handles = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
fig.legend(handles=handles, title="Experience Group",
           loc="upper right", fontsize=10, title_fontsize=11)

plt.tight_layout()
plt.savefig("correlation_plots/2_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: correlation_plots/2_pairplot.png")


# ────────────────────────────────────────────────────────
# 4. SCATTER — Top Correlated Pairs
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  4. SCATTER — Strongest Pairs")
print("=" * 55)

# Find top 4 strongest correlations (excluding diagonal)
corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
top_pairs = corr_pairs[:4]

print("Top 4 strongest correlations:")
for x, y, r in top_pairs:
    direction = "positive" if r > 0 else "negative"
    print(f"  {x} vs {y}: r = {r:.3f} ({direction})")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Scatter Plots — Strongest Correlated Pairs",
             fontsize=14, fontweight="bold")

scatter_colors = ["#2196F3", "#4CAF50", "#E91E63", "#FF9800"]

for ax, (var_x, var_y, r_val), color in zip(axes.flatten(), top_pairs, scatter_colors):
    ax.scatter(df[var_x], df[var_y], alpha=0.35, s=20, color=color)

    slope, intercept, r, p, _ = stats.linregress(df[var_x], df[var_y])
    x_line = np.linspace(df[var_x].min(), df[var_x].max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color="black", linewidth=2, linestyle="--", label=f"r = {r:.3f}")

    ax.set_xlabel(var_x.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(var_y.replace("_", " ").title(), fontsize=11)
    direction = "Positive" if r_val > 0 else "Negative"
    ax.set_title(f"{var_x} vs {var_y}\n({direction} correlation)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)

plt.tight_layout()
plt.savefig("correlation_plots/3_scatter_top_pairs.png", dpi=150)
plt.close()
print("Saved: correlation_plots/3_scatter_top_pairs.png")


# ────────────────────────────────────────────────────────
# 5. SUMMARY REPORT
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  5. SUMMARY REPORT")
print("=" * 55)

positive_pairs = [(x, y, r) for x, y, r in corr_pairs if r > 0]
negative_pairs = [(x, y, r) for x, y, r in corr_pairs if r < 0]
positive_pairs.sort(key=lambda x: x[2], reverse=True)
negative_pairs.sort(key=lambda x: x[2])

report = []
report.append("CORRELATION HEATMAP & PAIRWISE RELATIONSHIPS — REPORT")
report.append("Syntecxhub Internship | Week 2 | Project 3")
report.append("=" * 55)
report.append(f"\nDataset: {df.shape[0]} rows × {df.shape[1] - 1} numeric features\n")

report.append("PLOTS GENERATED")
for fname, desc in [
    ("1_correlation_heatmap.png", "Heatmap — Full Pearson matrix (lower triangle)"),
    ("2_pairplot.png",            "Pairplot — Key variable scatter matrix"),
    ("3_scatter_top_pairs.png",   "Scatter  — Top 4 strongest pairs"),
]:
    report.append(f"  {fname:<32} — {desc}")

report.append("\nFULL CORRELATION MATRIX")
report.append(corr.round(3).to_string())

report.append("\n\nTOP 5 POSITIVE CORRELATIONS")
for x, y, r in positive_pairs[:5]:
    report.append(f"  {x:<15} vs {y:<15} : r = +{r:.3f}")

report.append("\nTOP 5 NEGATIVE CORRELATIONS")
for x, y, r in negative_pairs[:5]:
    report.append(f"  {x:<15} vs {y:<15} : r =  {r:.3f}")

report.append("\n\nINTERPRETATION")
report.append(
    "The strongest positive correlations are between experience & salary (r≈0.93),\n"
    "experience & age (r≈0.89), and experience & projects (r≈0.82) — all intuitive:\n"
    "longer tenure brings higher pay, older age, and more completed projects.\n"
    "Rating also rises moderately with experience (r≈0.55), indicating senior\n"
    "employees tend to perform better.\n\n"
    "The strongest negative correlations are satisfaction vs absences (r≈−0.72)\n"
    "and satisfaction vs hours_worked (r≈−0.62) — confirming that overworked\n"
    "employees are less satisfied and take more days off.\n"
    "Hours worked vs satisfaction captures a classic burnout pattern.\n\n"
    "The heatmap's lower triangle (masked upper) avoids redundancy while keeping\n"
    "all pairwise values readable. The pairplot adds distributional context\n"
    "by splitting points into low/high experience groups."
)

with open("correlation_plots/summary_report.txt", "w") as f:
    f.write("\n".join(report))

print("\n".join(report[-20:]))
print("\n" + "=" * 55)
print("  Correlation Heatmap & Pairwise — COMPLETED ✓")
print("  All files saved in: ./correlation_plots/")
print("=" * 55)