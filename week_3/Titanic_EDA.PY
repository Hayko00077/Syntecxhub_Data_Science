"""
Titanic EDA — Full Exploratory Data Analysis
=============================================
Covers:
  1. Data loading
  2. Missingness inspection & dtypes
  3. Survival analysis by sex, class, age bucket
  4. Visualizations — bar charts, violin/boxplots, heatmap
  5. Printed insight report (5 bullets)

Requirements:
  pip install pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0. GLOBAL STYLE
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

BLUE  = "#378ADD"
RED   = "#E24B4A"
GREEN = "#1D9E75"
AMBER = "#EF9F27"
GRAY  = "#B4B2A9"


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load the Titanic dataset.
    Primary source: seaborn built-in datasets (requires internet on first call).
    Fallback: place a 'titanic.csv' file in the same directory.
    """
    URL = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/titanic.csv"
    )
    df = pd.read_csv(URL)
    print(df.head())
    print(df.shape)
    return df

# ─────────────────────────────────────────────
# 2. INSPECT MISSINGNESS & DTYPES
# ─────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print shape, dtypes, and missing value summary to console."""
    print("\n" + "=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n--- Column dtypes ---")
    print(df.dtypes.to_string())

    print("\n--- Missing values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    miss_df = pd.DataFrame({
        "missing_count": missing,
        "missing_%": missing_pct
    }).sort_values("missing_%", ascending=False)
    # Show only columns that actually have missing values
    print(miss_df[miss_df["missing_count"] > 0].to_string())

    print("\n--- Basic statistics (numeric columns) ---")
    print(df.describe().round(2).to_string())


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create age buckets used throughout the analysis."""
    df.columns = df.columns.str.lower()

    bins   = [0, 12, 18, 35, 60, 100]
    labels = ["Child (0-12)", "Teen (13-18)", "Adult (19-35)",
              "Middle (36-60)", "Senior (61+)"]
    df = df.copy()
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df


# ─────────────────────────────────────────────
# 4. SURVIVAL ANALYSIS — CONSOLE TABLES
# ─────────────────────────────────────────────
def survival_tables(df: pd.DataFrame) -> None:
    """Print survival rate tables broken down by sex, class, and age bucket."""

    def fmt(grp_col):
        tbl = df.groupby(grp_col, observed=True)["survived"].agg(
            survived="sum",
            total="count",
            survival_rate="mean"
        )
        tbl["survival_rate"] = (tbl["survival_rate"] * 100).round(1).astype(str) + "%"
        return tbl

    print("\n" + "=" * 55)
    print("SURVIVAL ANALYSIS")
    print("=" * 55)

    print("\n--- By sex ---")
    print(fmt("sex").to_string())

    print("\n--- By passenger class ---")
    print(fmt("pclass").to_string())

    print("\n--- By age bucket ---")
    print(fmt("age_bucket").to_string())

    print("\n--- By sex x class (survival rate matrix) ---")
    matrix = df.pivot_table(
        values="survived", index="sex",
        columns="pclass", aggfunc="mean"
    ) * 100
    print(matrix.round(1).to_string())


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_survival_by_sex(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Grouped bar chart: survival count by sex."""
    counts = df.groupby(["sex", "survived"]).size().unstack()
    counts.columns = ["Perished", "Survived"]
    counts.plot(kind="bar", ax=ax, color=[GRAY, BLUE],
                edgecolor="none", width=0.6)
    ax.set_title("Survival by Sex", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Passengers")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(frameon=False)

    # Annotate bars with survival rate percentage
    female_rate = df[df["sex"] == "female"]["survived"].mean()
    male_rate   = df[df["sex"] == "male"]["survived"].mean()
    for rate, x in zip([female_rate, male_rate], [0, 1]):
        ax.text(x, ax.get_ylim()[1] * 0.97,
                f"{rate:.0%}", ha="center", fontsize=10,
                color=BLUE, fontweight="bold")


def plot_survival_by_class(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Bar chart: survival rate per passenger class."""
    rates = df.groupby("pclass")["survived"].mean() * 100
    colors = [BLUE, AMBER, RED]
    bars = ax.bar(rates.index.astype(str), rates.values,
                  color=colors, edgecolor="none", width=0.5)
    ax.set_title("Survival Rate by Class", fontweight="bold")
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim(0, 80)
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 1.5, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")


def plot_survival_by_age_bucket(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Horizontal bar chart: survival rate by age bucket."""
    rates = (
        df.groupby("age_bucket", observed=True)["survived"]
        .mean() * 100
    ).sort_values()
    # Color code: green = high, amber = medium, red = low survival
    colors = [GREEN if v >= 50 else AMBER if v >= 35 else RED
              for v in rates.values]
    bars = ax.barh(rates.index.astype(str), rates.values,
                   color=colors, edgecolor="none", height=0.55)
    ax.set_title("Survival Rate by Age Group", fontweight="bold")
    ax.set_xlabel("Survival rate (%)")
    ax.set_xlim(0, 75)
    for bar, val in zip(bars, rates.values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)


def plot_age_violin(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Violin plot: age distribution for survivors vs perished."""
    survived_map = {0: "Perished", 1: "Survived"}
    df2 = df.dropna(subset=["age"]).copy()
    df2["Outcome"] = df2["survived"].map(survived_map)
    sns.violinplot(
        data=df2, x="Outcome", y="age",
        palette={"Survived": BLUE, "Perished": RED},
        inner="quartile", linewidth=0.8, ax=ax
    )
    ax.set_title("Age Distribution — Survived vs Perished", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Age")


def plot_age_boxplot_by_class(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Boxplot: age distribution per class, split by survival outcome."""
    df2 = df.dropna(subset=["age"]).copy()
    df2["Outcome"] = df2["survived"].map({0: "Perished", 1: "Survived"})
    sns.boxplot(
        data=df2, x="pclass", y="age", hue="Outcome",
        palette={"Survived": BLUE, "Perished": RED},
        linewidth=0.8, fliersize=3, ax=ax
    )
    ax.set_title("Age by Class & Survival", fontweight="bold")
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("Age")
    ax.legend(frameon=False, title="")


def plot_heatmap(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Heatmap: survival rate (%) broken down by sex and class."""
    matrix = df.pivot_table(
        values="survived", index="sex",
        columns="pclass", aggfunc="mean"
    ) * 100
    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="Blues",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Survival rate (%)"}, ax=ax
    )
    ax.set_title("Survival Rate (%) — Sex x Class", fontweight="bold")
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("")


def plot_missing_heatmap(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Binary heatmap showing which cells are missing across a random sample."""
    sample = df.sample(min(200, len(df)), random_state=42)
    miss = sample.isnull().astype(int)
    sns.heatmap(
        miss.T, cbar=False, cmap=["#E8E8E8", RED],
        yticklabels=True, xticklabels=False, ax=ax,
        linewidths=0
    )
    ax.set_title("Missingness Pattern (sample of 200 rows)", fontweight="bold")
    ax.set_xlabel("Passengers ->")
    ax.set_ylabel("")


def build_dashboard(df: pd.DataFrame) -> None:
    """Assemble all subplots into a single figure and save as PNG."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Titanic — Exploratory Data Analysis",
                 fontsize=18, fontweight="bold", y=1.01)

    # 3x3 grid; bottom row spans all 3 columns for the missingness heatmap
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])   # survival by sex
    ax2 = fig.add_subplot(gs[0, 1])   # survival by class
    ax3 = fig.add_subplot(gs[0, 2])   # survival by age bucket
    ax4 = fig.add_subplot(gs[1, 0])   # violin — age distribution
    ax5 = fig.add_subplot(gs[1, 1])   # boxplot — age by class
    ax6 = fig.add_subplot(gs[1, 2])   # sex x class heatmap
    ax7 = fig.add_subplot(gs[2, :])   # full-width missingness heatmap

    plot_survival_by_sex(df, ax1)
    plot_survival_by_class(df, ax2)
    plot_survival_by_age_bucket(df, ax3)
    plot_age_violin(df, ax4)
    plot_age_boxplot_by_class(df, ax5)
    plot_heatmap(df, ax6)
    plot_missing_heatmap(df, ax7)

    plt.tight_layout()
    out_path = "titanic_eda_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Dashboard saved -> {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# 6. INSIGHT REPORT
# ─────────────────────────────────────────────
def print_insight_report(df: pd.DataFrame) -> None:
    """Compute key statistics and print a 5-bullet insight summary."""
    female_rate = df[df["sex"] == "female"]["survived"].mean()
    male_rate   = df[df["sex"] == "male"]["survived"].mean()

    class_rates = df.groupby("pclass")["survived"].mean()

    child_rate  = df[df["age"] <= 12]["survived"].mean()
    senior_rate = df[df["age"] >= 61]["survived"].mean()

    f1_rate = df[(df["sex"] == "female") & (df["pclass"] == 1)]["survived"].mean()
    m3_rate = df[(df["sex"] == "male")   & (df["pclass"] == 3)]["survived"].mean()

    # Handle both seaborn ('deck' col) and raw CSV ('cabin' col)
    cabin_col   = "cabin" if "cabin" in df.columns else "deck"
    cabin_miss  = df[cabin_col].isnull().mean()
    age_miss    = df["age"].isnull().mean()

    report = f"""
{'=' * 55}
INSIGHT REPORT
{'=' * 55}
1. Sex is the strongest survival predictor:
   Women survived at {female_rate:.0%} vs men at only {male_rate:.0%}.
   The 'women and children first' policy was strictly enforced.

2. Wealth (class) directly determined survival chances:
   1st class -> {class_rates[1]:.0%}  |  2nd class -> {class_rates[2]:.0%}  |  3rd class -> {class_rates[3]:.0%}.
   Higher-class passengers had better lifeboat access.

3. Children had the best odds; seniors the worst:
   Ages 0-12 survived at ~{child_rate:.0%}, while passengers 61+
   had only a ~{senior_rate:.0%} survival rate.

4. Intersecting factors produce extreme outcome gaps:
   1st-class females survived at {f1_rate:.0%},
   3rd-class males at only {m3_rate:.0%} — a {f1_rate - m3_rate:.0%} gap.

5. Data quality issues require attention before modelling:
   Cabin/deck column is {cabin_miss:.0%} missing (nearly unusable).
   Age column is {age_miss:.0%} missing — median or model-based
   imputation is recommended.
{'=' * 55}
"""
    print(report)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1 — load the dataset
    df = load_data()

    # Step 2 — inspect missingness and dtypes
    inspect_data(df)

    # Step 3 — create age bucket feature
    df = engineer_features(df)

    # Step 4 — print survival breakdown tables
    survival_tables(df)

    # Step 5 — build and save the full visual dashboard
    build_dashboard(df)

    # Step 6 — print the insight report
    print_insight_report(df)