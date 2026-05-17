import io
import re
import warnings
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ── Profile sections ───────────────────────────────────────────────────────────

def _data_types(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        non_na = df[col].dropna()
        samples = non_na.sample(min(3, len(non_na))).tolist() if not non_na.empty else []
        rows.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].notna().sum()),
            "Null Count": int(df[col].isnull().sum()),
            "Sample Values": str(samples),
        })
    return pd.DataFrame(rows)


def _numerical_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if s.empty:
            continue
        modes = df[col].mode()
        if modes.empty:
            mode_display = "—"
        elif len(modes) == 1:
            mode_display = modes.iloc[0]
        else:
            mode_display = f"{modes.iloc[0]} (+{len(modes) - 1} more)"
        rows.append({
            "Column": col,
            "Mean": round(float(s.mean()), 4),
            "Median": round(float(s.median()), 4),
            "Mode": mode_display,
            "Std Dev": round(float(s.std()), 4),
            "Min": float(s.min()),
            "Max": float(s.max()),
            "Skewness": round(float(s.skew()), 4),
            "Kurtosis": round(float(s.kurtosis()), 4),
        })
    return pd.DataFrame(rows)


def _unique_values(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        unique_vals = df[col].dropna().unique()
        n = len(unique_vals)
        sample = unique_vals[:5].tolist() if n > limit else unique_vals.tolist()
        suffix = f" … +{n - 5} more" if n > limit else ""
        rows.append({
            "Column": col,
            "Unique Count": n,
            "Values Sample": str(sample) + suffix,
        })
    return pd.DataFrame(rows)


def _special_characters(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised scan — uses str.contains() instead of per-cell Python loop."""
    pattern = r"[^\w\s.,!?'\"@#&()\-]"
    rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        # Keep only actual string values (object columns may contain mixed types)
        str_mask = col_data.apply(lambda x: isinstance(x, str))
        col_str = col_data[str_mask]
        if col_str.empty:
            continue
        hits = col_str[col_str.str.contains(pattern, regex=True, na=False)]
        for idx, val in hits.items():
            rows.append({"Column": col, "Row Index": idx, "Value": val})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Column", "Row Index", "Value"])


def _datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=["datetime64"]).columns:
        rows.append({
            "Column": col,
            "Min Date": df[col].min(),
            "Max Date": df[col].max(),
            "Null Count": int(df[col].isnull().sum()),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Column", "Min Date", "Max Date", "Null Count"]
    )


def _missing_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        missing_idx = df[df[col].isnull()].index.tolist()
        if missing_idx:
            rows.append({
                "Column": col,
                "Missing Count": len(missing_idx),
                "Missing %": f"{len(missing_idx) / len(df) * 100:.1f}%",
                "Row Indices (first 20)": str(missing_idx[:20]) + ("…" if len(missing_idx) > 20 else ""),
            })
    if not rows:
        return pd.DataFrame([{
            "Column": "—",
            "Missing Count": 0,
            "Missing %": "0%",
            "Row Indices (first 20)": "No missing values.",
        }])
    return pd.DataFrame(rows)


def _duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    dupes = df[df.duplicated(keep=False)]
    if not dupes.empty:
        return dupes.sort_values(by=list(df.columns)).reset_index(drop=True)
    return pd.DataFrame([{"Info": "No duplicate rows found."}])


def _correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) < 2:
        return pd.DataFrame([{"Info": "Not enough numerical columns for correlation."}])
    temp = df[num_cols].dropna()
    if temp.empty:
        return pd.DataFrame([{"Info": "All numerical values are null after dropping NaNs."}])
    corr = temp.corr().round(3)
    return corr.reset_index().rename(columns={"index": "Column"})


# ── Orchestrator ───────────────────────────────────────────────────────────────

def generate_profile(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "1_Data_Types": _data_types(df),
        "2_Numerical_Stats": _numerical_stats(df),
        "3_Unique_Values": _unique_values(df),
        "4_Special_Characters": _special_characters(df),
        "5_Datetime_Columns": _datetime_columns(df),
        "6_Missing_Values": _missing_values(df),
        "7_Duplicate_Rows": _duplicate_rows(df),
        "8_Correlation_Matrix": _correlation_matrix(df),
    }


# ── Report helpers ─────────────────────────────────────────────────────────────

def _sc_summary_df(sc: pd.DataFrame) -> pd.DataFrame:
    """Per-column summary of special characters — used in Excel and HTML exports."""
    _pat = r"[^\w\s.,!?'\"@#&()\-]"
    rows = []
    for col, grp in sc.groupby("Column"):
        chars = set()
        for v in grp["Value"]:
            chars.update(re.findall(_pat, str(v)))
        rows.append({
            "Column": col,
            "Affected Values": len(grp),
            "Special Characters Found": " ".join(sorted(chars)),
            "Sample Values (up to 5)": "  |  ".join(str(v) for v in grp["Value"].head(5).tolist()),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Column", "Affected Values", "Special Characters Found", "Sample Values (up to 5)"]
    )


# ── Excel export ───────────────────────────────────────────────────────────────

def generate_excel_report(profile: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        wb = writer.book
        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#1f77b4", "font_color": "#ffffff",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        })
        cell_fmt = wb.add_format({"border": 1, "text_wrap": True, "valign": "top"})

        for sheet_name, df in profile.items():
            safe = sheet_name[:31].translate(str.maketrans("", "", r"/\*?[]:'"))

            # Default: write full DataFrame
            df.to_excel(writer, sheet_name=safe, index=False, startrow=1, header=False)
            ws = writer.sheets[safe]
            for col_idx, col_name in enumerate(df.columns):
                ws.write(0, col_idx, col_name, header_fmt)
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    val = df.iloc[row_idx, col_idx]
                    ws.write(row_idx + 1, col_idx, str(val) if pd.isna(val) else val, cell_fmt)
            ws.set_column(0, len(df.columns) - 1, 22)

    return buffer.getvalue()


# ── HTML export ────────────────────────────────────────────────────────────────

def generate_html_report(
    df: pd.DataFrame,
    profile: Dict[str, pd.DataFrame],
    filename: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    chart_blocks = []
    first_chart = True

    def _chart_html(fig, title: str) -> str:
        nonlocal first_chart
        include_js = "cdn" if first_chart else False
        first_chart = False
        fig.update_layout(
            title=title,
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white",
        )
        return f'<div class="chart-wrap">{fig.to_html(full_html=False, include_plotlyjs=include_js)}</div>'

    # Missing values bar
    mv = profile["6_Missing_Values"]
    has_missing = not (len(mv) == 1 and mv.iloc[0]["Column"] == "—")
    if has_missing:
        fig = px.bar(
            mv, x="Column", y="Missing Count",
            color="Missing Count", color_continuous_scale="reds",
            text="Missing %",
        )
        chart_blocks.append(_chart_html(fig, "Missing Values by Column"))

    # Numeric distributions (up to 8 columns)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols[:8]:
        fig = px.histogram(df, x=col, marginal="box", nbins=40)
        chart_blocks.append(_chart_html(fig, f"Distribution — {col}"))

    # Categorical value counts (up to 8 columns)
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols[:8]:
        vc = df[col].value_counts().head(20).reset_index()
        vc.columns = [col, "Count"]
        if vc.empty:
            continue
        fig = px.bar(vc, x=col, y="Count")
        chart_blocks.append(_chart_html(fig, f"Value Counts — {col}"))

    # Datetime time series (up to 3 columns)
    dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in dt_cols[:3]:
        ts = df[[col]].dropna().set_index(col)
        try:
            ts_monthly = ts.resample("ME").size().reset_index()
        except Exception:
            ts_monthly = ts.resample("M").size().reset_index()
        ts_monthly.columns = [col, "Records"]
        fig = px.line(ts_monthly, x=col, y="Records")
        chart_blocks.append(_chart_html(fig, f"Records Over Time — {col}"))

    # Correlation heatmap
    corr_df = profile["8_Correlation_Matrix"]
    if "Info" not in corr_df.columns and "Error" not in corr_df.columns:
        corr_cols = [c for c in corr_df.columns if c != "Column"]
        z = corr_df[corr_cols].values
        fig = go.Figure(go.Heatmap(
            z=z, x=corr_cols, y=corr_df["Column"].tolist(),
            colorscale="RdBu", zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
        ))
        chart_blocks.append(_chart_html(fig, "Correlation Matrix"))

    # Profile tables
    table_blocks = []
    section_icons = {
        "1_Data_Types": "📋",
        "2_Numerical_Stats": "📐",
        "3_Unique_Values": "🔤",
        "4_Special_Characters": "⚠️",
        "5_Datetime_Columns": "📅",
        "6_Missing_Values": "❌",
        "7_Duplicate_Rows": "🔁",
        "8_Correlation_Matrix": "🔗",
    }
    for key, section_df in profile.items():
        icon = section_icons.get(key, "•")
        label = key.replace("_", " ").title()

        if key == "4_Special_Characters" and not section_df.empty and "Column" in section_df.columns:
            summary = _sc_summary_df(section_df)
            sample = section_df.groupby("Column").head(5).reset_index(drop=True)
            table_blocks.append(
                f'<div class="table-section">'
                f'<h3>{icon} {label}</h3>'
                f'<p style="margin-bottom:10px;font-size:0.84rem;color:#555">'
                f'{len(section_df):,} values with special characters across '
                f'{section_df["Column"].nunique()} column(s).</p>'
                f'{summary.to_html(index=False, classes="profile-table", border=0, na_rep="—")}'
                f'<p style="margin:12px 0 6px;font-size:0.82rem;color:#888">'
                f'<em>Sample rows (up to 5 per column):</em></p>'
                f'{sample.to_html(index=False, classes="profile-table", border=0, na_rep="—")}'
                f'</div>'
            )
        elif key == "7_Duplicate_Rows" and "Info" not in section_df.columns:
            total = len(section_df)
            unique = len(section_df.drop_duplicates())
            sample = section_df.drop_duplicates().head(5)
            table_blocks.append(
                f'<div class="table-section">'
                f'<h3>{icon} {label}</h3>'
                f'<p style="margin-bottom:10px;font-size:0.84rem;color:#555">'
                f'<strong>{total:,}</strong> duplicate rows across '
                f'<strong>{unique:,}</strong> unique patterns.</p>'
                f'<p style="margin-bottom:6px;font-size:0.82rem;color:#888">'
                f'<em>Sample patterns (up to 5):</em></p>'
                f'{sample.to_html(index=False, classes="profile-table", border=0, na_rep="—")}'
                f'</div>'
            )
        else:
            table_blocks.append(
                f'<div class="table-section">'
                f'<h3>{icon} {label}</h3>'
                f'{section_df.to_html(index=False, classes="profile-table", border=0, na_rep="—")}'
                f'</div>'
            )

    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = f"{total_missing / total_cells * 100:.1f}%" if total_cells else "0%"
    dup_count = int(df.duplicated().sum())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClearView Report — {filename}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #1a1a2e; }}
  .header {{ background: linear-gradient(135deg, #002A33, #00C9A7); color: #fff; padding: 32px 40px; }}
  .header h1 {{ font-size: 1.7rem; font-weight: 700; }}
  .meta {{ display: flex; gap: 28px; margin-top: 12px; font-size: 0.88rem; opacity: 0.85; flex-wrap: wrap; }}
  .meta span {{ background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 20px; }}
  .stats-bar {{ display: flex; gap: 16px; padding: 20px 40px; background: #fff; border-bottom: 1px solid #e8eaf0; flex-wrap: wrap; }}
  .stat-card {{ flex: 1; min-width: 120px; background: #f5f7fa; border-radius: 8px; padding: 14px 18px; text-align: center; }}
  .stat-value {{ font-size: 1.6rem; font-weight: 700; color: #00C9A7; }}
  .stat-label {{ font-size: 0.75rem; color: #666; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .content {{ max-width: 1400px; margin: 24px auto; padding: 0 24px; }}
  .section {{ background: #fff; border-radius: 10px; padding: 28px 32px; margin-bottom: 24px; box-shadow: 0 1px 6px rgba(0,0,0,.07); }}
  .section h2 {{ font-size: 1.15rem; color: #1a1a2e; border-bottom: 2px solid #e8eaf0; padding-bottom: 12px; margin-bottom: 20px; }}
  .chart-wrap {{ margin-bottom: 20px; }}
  .table-section {{ margin-bottom: 28px; }}
  .table-section h3 {{ font-size: 0.95rem; color: #444; margin-bottom: 10px; }}
  .profile-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  .profile-table th {{ background: #f0f4f8; padding: 9px 14px; text-align: left; font-weight: 600; color: #333; border-bottom: 2px solid #dde1e8; }}
  .profile-table td {{ padding: 7px 14px; border-bottom: 1px solid #eef0f4; color: #444; vertical-align: top; word-break: break-word; }}
  .profile-table tr:hover td {{ background: #fafbfc; }}
  footer {{ text-align: center; padding: 24px; color: #999; font-size: 0.78rem; }}
</style>
</head>
<body>

<div class="header">
  <h1>🔎 ClearView Report — {filename}</h1>
  <div class="meta">
    <span>Generated {now}</span>
    <span>{len(df):,} rows</span>
    <span>{len(df.columns):,} columns</span>
    <span>{missing_pct} missing</span>
    <span>{dup_count:,} duplicate rows</span>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-card"><div class="stat-value">{len(df):,}</div><div class="stat-label">Rows</div></div>
  <div class="stat-card"><div class="stat-value">{len(df.columns):,}</div><div class="stat-label">Columns</div></div>
  <div class="stat-card"><div class="stat-value">{len(num_cols)}</div><div class="stat-label">Numeric</div></div>
  <div class="stat-card"><div class="stat-value">{len(cat_cols)}</div><div class="stat-label">Text</div></div>
  <div class="stat-card"><div class="stat-value">{len(dt_cols)}</div><div class="stat-label">Datetime</div></div>
  <div class="stat-card"><div class="stat-value">{missing_pct}</div><div class="stat-label">Missing</div></div>
  <div class="stat-card"><div class="stat-value">{dup_count:,}</div><div class="stat-label">Duplicates</div></div>
</div>

<div class="content">
  <div class="section">
    <h2>📈 Visualizations</h2>
    {"".join(chart_blocks) if chart_blocks else "<p style='color:#888'>No charts generated.</p>"}
  </div>
  <div class="section">
    <h2>🔍 Data Profile</h2>
    {"".join(table_blocks)}
  </div>
</div>

<footer>Generated by ClearView &nbsp;·&nbsp; {now}</footer>
</body>
</html>"""