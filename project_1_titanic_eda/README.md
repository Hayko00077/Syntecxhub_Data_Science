# 🚢 Titanic — Exploratory Data Analysis

A full exploratory data analysis (EDA) of the Titanic dataset, covering missing data inspection, survival analysis, and rich visualizations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Steps](#analysis-steps)
- [Visualizations](#visualizations)
- [Key Insights](#key-insights)
- [Data Quality Notes](#data-quality-notes)

---

## Overview

This project performs a complete EDA on the Titanic passenger dataset. It loads the data directly from a public GitHub source, inspects missing values and data types, engineers useful features, generates survival breakdown tables, and produces a multi-panel visual dashboard — all in a single script.

---

## Project Structure

```
titanic-eda/
├── titanic_eda.py               # Main analysis script
├── titanic_eda_dashboard.png    # Output dashboard (generated on run)
└── README.md                    # This file
```

---

## Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

Python 3.8 or higher is recommended.

---

## Usage

Run the script directly from the command line:

```bash
python titanic_eda.py
```

The script will:
1. Download the Titanic dataset automatically from GitHub
2. Print inspection tables and survival summaries to the console
3. Save the visual dashboard as `titanic_eda_dashboard.png`
4. Print a 5-bullet insight report

> **Note:** An internet connection is required on the first run to fetch the dataset.

---

## Analysis Steps

| Step | Function | Description |
|------|----------|-------------|
| 1 | `load_data()` | Downloads the CSV from GitHub and returns a DataFrame |
| 2 | `inspect_data()` | Prints shape, dtypes, missing value counts and percentages |
| 3 | `engineer_features()` | Creates `age_bucket` column (Child / Teen / Adult / Middle / Senior) |
| 4 | `survival_tables()` | Prints survival rates by sex, class, age bucket, and sex × class matrix |
| 5 | `build_dashboard()` | Generates and saves a 7-panel matplotlib figure |
| 6 | `print_insight_report()` | Computes and prints a 5-point summary of key findings |

---

## Visualizations

The dashboard (`titanic_eda_dashboard.png`) contains 7 subplots:

1. **Survival by Sex** — Grouped bar chart showing counts of survivors vs perished for males and females, annotated with survival rates
2. **Survival Rate by Class** — Bar chart comparing survival rates across 1st, 2nd, and 3rd class
3. **Survival Rate by Age Group** — Horizontal bar chart for each age bucket, color-coded by rate (green / amber / red)
4. **Age Distribution — Survived vs Perished** — Violin plot comparing age distributions by outcome
5. **Age by Class & Survival** — Boxplot showing age spread per class, split by survival outcome
6. **Sex × Class Heatmap** — Annotated heatmap of survival rates broken down by both sex and class
7. **Missingness Pattern** — Binary heatmap over a 200-row sample showing which fields contain missing values

---

## Key Insights

1. **Sex is the strongest predictor of survival.** Women survived at ~74% versus only ~19% for men, reflecting the "women and children first" evacuation policy.

2. **Passenger class directly determined survival chances.** 1st class: ~63% | 2nd class: ~47% | 3rd class: ~24%. Higher-class passengers had better lifeboat access.

3. **Children fared best; seniors fared worst.** Passengers aged 0–12 had the highest survival rate (~58%), while those aged 61+ had the lowest (~27%).

4. **Intersecting factors produce extreme outcome gaps.** 1st-class females survived at ~97%, while 3rd-class males survived at only ~15% — a gap of over 80 percentage points.

5. **Data quality issues must be addressed before modelling.** The cabin/deck column is ~77% missing (nearly unusable). The age column is ~20% missing — median or model-based imputation is recommended.

---

## Data Quality Notes

| Column | Missing % | Recommendation |
|--------|-----------|----------------|
| Cabin / Deck | ~77% | Drop or use as binary "known/unknown" feature |
| Age | ~20% | Impute with median or regression-based approach |
| Embarked | <1% | Fill with mode |

---

## License

This project is released for educational purposes. The Titanic dataset is publicly available and widely used for data science learning.