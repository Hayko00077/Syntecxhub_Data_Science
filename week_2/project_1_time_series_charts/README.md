# Time Series & Category Charts

**Syntecxhub Data Science Internship | Week 2 — Project 1**

## Overview
This project focuses on data visualization using Matplotlib. It generates a synthetic sales dataset across 5 product categories over a full year, then produces a series of publication-ready charts covering time-series trends, monthly/quarterly aggregations, category comparisons, and market share — all exported as PNG files with a summary report.

## Topics Covered
- Line charts for daily sales with 7-day rolling average
- Monthly aggregation line chart with markers
- Grouped bar chart for quarterly sales comparison
- Bar chart for annual category ranking
- Pie chart for market share analysis
- Dollar-formatted axes, rotated labels, legends, grid lines
- Saving charts to PNG files
- Exporting a written summary report

## Requirements
```
pandas
numpy
matplotlib
```

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib
```

## Usage
```bash
python3 time_series_charts.py
```

## Output — `charts/` folder
| File | Description |
|---|---|
| `1_line_daily_sales.png` | Daily sales line chart (7-day rolling avg) |
| `2_line_monthly_sales.png` | Monthly aggregation per category |
| `3_bar_quarterly_sales.png` | Quarterly grouped bar chart |
| `4_bar_category_total.png` | Annual total per category |
| `5_pie_market_share.png` | Market share pie chart |
| `summary_report.txt` | Written chart interpretation |

## Chart Choice Discussion
| Chart Type | Best Used For |
|---|---|
| Line chart | Continuous time-series trends |
| Bar chart | Discrete category comparisons |
| Pie chart | Part-to-whole proportions (≤6 categories) |

## Tech Stack
- Python 3.x
- Pandas
- NumPy
- Matplotlib
