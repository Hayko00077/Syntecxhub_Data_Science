# COVID-19 Data Analysis

A Python project that loads country-wise COVID-19 time-series data and performs statistical analysis and visualization.

## Features
- Daily & weekly case computation
- 7-day rolling average (noise smoothing)
- Country comparison (absolute + per 100k population)
- Peak detection using scipy
- Effective reproduction number (Rₜ) approximation
- Exports 4 publication-ready charts

## Countries Analyzed
USA · India · Brazil · UK · Germany · Armenia

## Charts Generated
| File | Description |
|------|-------------|
| `chart1_daily_cases.png` | Daily cases + 7-day rolling average per country |
| `chart2_comparison.png` | Absolute vs per-100k population comparison |
| `chart3_weekly_rt.png` | Weekly totals + Rₜ estimate |
| `chart4_peak_detection.png` | Peak detection across all countries |

## Installation
```bash
pip install pandas matplotlib scipy numpy
```

## Usage
```bash
python covid19_analysis.py
```

## Results & Conclusions
- All countries show 2 clear waves: spring 2020 and autumn/winter 2020
- Per-100k normalization significantly changes country rankings
- All national peaks cluster in Oct–Dec 2020 (seasonal signal)
- Average Rₜ ≈ 1.02–1.03 across all countries during the period
