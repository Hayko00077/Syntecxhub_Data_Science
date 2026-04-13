# Pandas CSV Reader & Basic Analysis

**Syntecxhub Data Science Internship | Week 1 — Project 2**

## Overview
This project demonstrates how to read, explore, filter, and analyze structured data using the Pandas library. A sample employee dataset is auto-generated, loaded into a DataFrame, and analyzed with various Pandas operations.

## Topics Covered
- Reading CSV and Excel files into a DataFrame
- Inspecting data (head, tail, dtypes, shape, info)
- Summary statistics (mean, median, min, max, count)
- Filtering rows by conditions (single & multiple filters)
- Selecting columns and slicing subsets (iloc, loc)
- GroupBy aggregations by department
- Saving filtered results to CSV and Excel

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
python3 pandas_csv_reader.py
```

## Output
- Terminal output with all analysis results
- `high_salary_employees.csv` — filtered high earners
- `senior_employees.xlsx` — filtered senior staff
- `department_summary.csv` — aggregated department stats

## Tech Stack
- Python 3.x
- Pandas
- NumPy
- OpenPyXL
