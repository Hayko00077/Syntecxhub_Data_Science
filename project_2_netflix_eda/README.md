# Netflix Content EDA

**Syntecxhub Data Science Internship | Week 3 — Project 2**

## Overview
Full exploratory data analysis on a Netflix titles dataset. The script downloads the dataset automatically, performs analysis, generates two pages of charts, and exports a professional PDF report.

## Topics Covered
- Loading and cleaning real-world Netflix dataset
- Content type split (Movies vs TV Shows)
- Annual and cumulative content growth trends
- Top 10 genres for Movies and TV Shows
- Content rating distribution
- Movie runtime distribution (histogram)
- TV Show seasons distribution
- Top release years analysis
- Runtime by genre (boxplot)
- Content origin by country (pie chart)
- Exporting charts to PNG and full PDF report

## Requirements
```
pandas
numpy
matplotlib
reportlab
pillow
```

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib reportlab pillow
```

## Usage
```bash
python3 Netflix_eda.py
```

## Data Source
Dataset is auto-downloaded from GitHub mirrors.
Manual download available at:
[Kaggle — Netflix Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)

After manual download set in script:
```python
DATASET_URL = None
CSV_PATH = 'netflix_titles.csv'
```

## Output
| File | Description |
|---|---|
| `page1.png` | Overview charts (donut, area, genres, ratings) |
| `page2.png` | Deep dive (runtime, seasons, countries, KPIs) |
| `Netflix_EDA_Report.pdf` | Full professional PDF report |

## Key Findings
- Movies dominate Netflix content (~70%)
- Content volumes peaked around 2019–2020
- Drama and International content lead all genres
- Most content originates from the United States
- Typical movie runtime clusters around 90–100 minutes

## Tech Stack
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- ReportLab
- Pillow