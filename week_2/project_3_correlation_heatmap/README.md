# Correlation Heatmap & Pairwise Relationships

**Syntecxhub Data Science Internship | Week 2 — Project 3**

## Overview
This project computes and visualizes Pearson correlations between numeric employee features. It produces a masked correlation heatmap (lower triangle only), a pairplot scatter matrix for key variables, and scatter plots of the strongest correlated pairs — all accompanied by a written summary of the most meaningful positive and negative relationships.

## Topics Covered
- Computing Pearson correlation matrix for all numeric features
- Visualizing heatmap with masked upper triangle and annotated values
- Custom color scale (blue = positive, red = negative)
- Pairplot / scatter matrix for key variable pairs
- Regression lines with r-values on scatter plots
- Identifying and ranking strongest positive/negative correlations
- Exporting plots and a written summary report

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
python3 correlation_heatmap.py
```

## Output — `correlation_plots/` folder
| File | Description |
|---|---|
| `1_correlation_heatmap.png` | Full heatmap (lower triangle, annotated) |
| `2_pairplot.png` | Scatter matrix for key variables |
| `3_scatter_top_pairs.png` | Top 4 strongest correlated pairs |
| `summary_report.txt` | Full correlation matrix + interpretation |

## Key Findings
| Pair | r value | Direction |
|---|---|---|
| experience vs salary | ~0.93 | Strong positive |
| experience vs age | ~0.89 | Strong positive |
| experience vs projects | ~0.82 | Strong positive |
| satisfaction vs absences | ~−0.72 | Strong negative |
| satisfaction vs hours_worked | ~−0.62 | Moderate negative |

## Tech Stack
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- SciPy
