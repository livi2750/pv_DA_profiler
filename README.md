# 🔎 ClearView — Data Profiler

**ClearView** is a lightweight, browser-based data profiling and cleaning tool built with Streamlit. Upload any CSV or Excel file and instantly get a quality report, exploratory charts, and a point-and-click cleaning workflow — no code required.

---

## Features

### 🔍 Profile
Automated data quality report for every column:
- Data types, null counts, and sample values
- Numerical statistics (mean, median, mode, std dev, skewness, kurtosis)
- Unique value counts for text columns
- Special characters scan — per-column summary with sample rows
- Datetime range detection
- Missing values with counts and percentages
- Duplicate row detection
- Correlation matrix

### 📈 Explore
Auto-generated EDA charts for every column:
- Histogram + box plot for numeric columns
- Top-20 value counts for text columns
- Monthly time series for datetime columns
- Correlation heatmap
- Interactive relationship explorer (scatter with OLS trendline, color-by, size-by)

### 🧹 Clean
Preview every change before applying it:
- Strip extra spaces (leading, trailing, and internal runs)
- Remove duplicate rows (keep first or last)
- Parse date columns (text → datetime)
- Fill missing values (mean / median / mode / custom value, per column)
- Drop columns
- Rename columns

Download the cleaned dataset as **CSV** or **Excel**.

### 📥 Reports
- **Excel report** — one sheet per profile section; special characters and duplicates are exported as summaries with sample rows (not raw dumps)
- **HTML report** — self-contained, shareable file with embedded Plotly charts and full profile tables

---

## Supported Formats

| Format | Notes |
|--------|-------|
| `.csv` | Configurable encoding (UTF-8, Latin-1, ISO-8859-1) and separator (`,` `;` `|` Tab) |
| `.xlsx` | Sheet picker included |
| `.xls` | Sheet picker included |

---
