# Data Cleaning Utility

**Syntecxhub Data Science Internship | Week 1 — Project 3**

## Overview
A practical data cleaning utility that takes a deliberately messy dataset and applies systematic cleaning steps. The tool detects and handles missing values, fixes incorrect data types, parses dates in mixed formats, removes duplicates, and standardizes column names — then outputs a cleaned dataset and a cleaning log.

## Topics Covered
- Detecting and handling missing values (drop / fill / impute with median & mode)
- Fixing incorrect data types (coercing non-numeric strings to NaN)
- Parsing dates in multiple mixed formats
- Removing duplicate rows
- Standardizing column names (lowercase + underscores)
- Standardizing string fields (title case, strip whitespace)
- Exporting cleaned dataset and a step-by-step cleaning log

## Requirements
```
pandas
openpyxl
numpy
```

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas openpyxl numpy
```

## Usage
```bash
python3 data_cleaning_utility.py
```

## Output
- Terminal output with before/after comparisons
- `cleaned_employees.csv` — fully cleaned dataset
- `cleaning_log.txt` — step-by-step log of all changes made

## Tech Stack
- Python 3.x
- Pandas
- NumPy
