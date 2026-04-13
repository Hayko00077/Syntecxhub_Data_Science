# Statistical Plots & Distribution Analysis

**Syntecxhub Data Science Internship | Week 2 — Project 2**

## Overview
This project performs distribution analysis on a two-region employee dataset using histograms, KDE curves, boxplots, and outlier detection. It compares salary, rating, and experience distributions between Region A (normal distribution) and Region B (right-skewed with outliers), and exports all plots alongside a written interpretation report.

## Topics Covered
- Histograms with mean/median lines per group
- KDE (Kernel Density Estimation) curves for shape comparison
- Boxplots for multi-feature group comparison
- Outlier detection using the IQR method (1.5 × IQR rule)
- Skewness and kurtosis analysis
- Comparing distributions across two groups (Region A vs Region B)
- Exporting plots to PNG and a written interpretation report

## Requirements
```
pandas
numpy
matplotlib
scipy
```

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib scipy
```

## Usage
```bash
python3 statistical_plots.py
```

## Output — `plots/` folder
| File | Description |
|---|---|
| `1_histogram_salary.png` | Salary histogram per region |
| `2_kde_salary_rating.png` | KDE density curves |
| `3_boxplot_comparison.png` | Boxplot group comparison |
| `4_outlier_detection.png` | Outlier scatter (IQR method) |
| `5_skewness_spread.png` | Skewness & spread bar chart |
| `interpretation_report.txt` | Written analysis & conclusions |

## Key Findings
- **Region A** — near-normal salary distribution, low variance, no outliers
- **Region B** — right-skewed salary, high variance, 5+ extreme outliers ($180k–$300k)
- **Satisfaction vs Absences** — strong negative correlation confirmed visually

## Tech Stack
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- SciPy
