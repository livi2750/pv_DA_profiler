from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cleaner import (
    apply_parse_dates,
    apply_remove_duplicates,
    apply_strip_spaces,
    preview_parse_dates,
    preview_remove_duplicates,
    preview_strip_spaces,
    to_csv_bytes,
    to_excel_bytes,
)
from profiler import (
    generate_excel_report,
    generate_html_report,
    generate_profile,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClearView",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Base ── */
  .stApp { background-color: #000000; }
  section[data-testid="stSidebar"] {
    background: #001A20 !important;
    border-right: 1px solid #003D4A !important;
  }

  /* ── Metrics ── */
  [data-testid="stMetricValue"] {
    font-size: 1.3rem !important; font-weight: 700 !important; color: #00C9A7 !important;
  }
  [data-testid="stMetricLabel"] { color: #D9DADA !important; font-size: 0.75rem !important; }
  [data-testid="stMetric"] {
    background: #002A33; border: 1px solid #004D5C;
    border-radius: 10px; padding: 12px 16px;
  }
  [data-testid="stMetricDelta"] { color: #646464 !important; }

  /* ── Divider ── */
  hr { border-color: #646464 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #003D4A;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #646464 !important; font-weight: 600;
    border-radius: 8px 8px 0 0; padding: 8px 20px;
  }
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #00C9A7 !important;
    border-bottom: 3px solid #00C9A7 !important;
    background: rgba(0,201,167,0.07) !important;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #00B4D8 !important; }

  /* ── Expanders ── */
  [data-testid="stExpander"] {
    border: 1px solid #003D4A !important;
    border-radius: 8px !important;
    background: #001822 !important;
  }
  [data-testid="stExpander"] summary {
    color: #D9DADA !important; font-weight: 600;
  }
  [data-testid="stExpander"] summary:hover { color: #00C9A7 !important; }

  /* ── Buttons ── */
  .stDownloadButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00C9A7, #00B4D8) !important;
    color: #fff !important; border: none !important;
    font-weight: 600 !important; border-radius: 8px !important;
  }
  .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #00B4D8, #00C9A7) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,201,167,0.4) !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00C9A7, #00B4D8) !important;
    color: #fff !important; border: none !important;
    font-weight: 600 !important; border-radius: 8px !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00B4D8, #00C9A7) !important;
    box-shadow: 0 4px 16px rgba(0,201,167,0.4) !important;
  }
  .stButton > button {
    border: 1px solid #003D4A !important;
    color: #D9DADA !important; border-radius: 8px !important;
  }
  .stButton > button:hover {
    border-color: #00C9A7 !important; color: #00C9A7 !important;
  }

  /* ── Status box ── */
  [data-testid="stStatus"] {
    border: 1px solid #003D4A !important;
    background: #001822 !important;
    border-radius: 10px !important;
  }

  /* ── Selectbox / Radio / Multiselect ── */
  [data-testid="stSelectbox"] label,
  [data-testid="stRadio"] label,
  [data-testid="stMultiSelect"] label { color: #D9DADA !important; }

  /* ── Dataframes ── */
  [data-testid="stDataFrame"] { border: 1px solid #003D4A; border-radius: 8px; }

  /* ── Clean banner ── */
  .clean-banner {
    background: rgba(0,201,167,0.08); border: 1px solid #00C9A7;
    border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; color: #00C9A7;
  }
  .op-chip {
    display: inline-block;
    background: rgba(0,201,167,0.12); border-radius: 12px;
    padding: 2px 10px; font-size: 0.78rem; margin: 2px 4px 2px 0; color: #00C9A7;
  }

  /* ── Welcome: hero ── */
  .welcome-hero {
    background: linear-gradient(135deg, #002A33 0%, #007A8A 55%, #00C9A7 100%);
    border-radius: 14px; padding: 44px 52px; color: white;
    margin-bottom: 28px; border: 1px solid #00B4D8;
  }
  .welcome-hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 10px; letter-spacing: -0.5px; }
  .welcome-hero p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }

  /* ── Welcome: feature cards ── */
  .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 28px; }
  .feature-card {
    background: #002A33; border: 1px solid #004D5C;
    border-radius: 12px; padding: 24px 22px;
    cursor: default; user-select: none;
  }
  .feature-card .icon { font-size: 1.8rem; margin-bottom: 12px; }
  .feature-card h3  { font-size: 1rem; font-weight: 700; margin: 0 0 8px; color: #FFFFFF; }
  .feature-card p   { font-size: 0.85rem; color: #D9DADA; margin: 0 0 14px; line-height: 1.6; }
  .feature-card .tab-hint { font-size: 0.75rem; color: #646464; font-style: italic; }

  /* ── Welcome: steps ── */
  .steps-row { display: flex; align-items: center; gap: 0; margin-bottom: 8px; }
  .step {
    flex: 1; background: #002A33; border: 1px solid #004D5C;
    border-radius: 8px; padding: 14px 10px;
    text-align: center; font-size: 0.82rem; font-weight: 600; color: #FFFFFF;
  }
  .step span { display: block; font-size: 1.3rem; margin-bottom: 5px; }
  .step-arrow { color: #00C9A7; font-size: 1.4rem; padding: 0 4px; flex-shrink: 0; }

  /* ── Welcome: format badges ── */
  .formats-row { display: flex; gap: 10px; margin-top: 6px; }
  .fmt-badge {
    background: rgba(0,201,167,0.12); color: #00C9A7;
    border: 1px solid rgba(0,201,167,0.35);
    border-radius: 6px; padding: 4px 14px;
    font-size: 0.8rem; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [("file_key", None), ("df_clean", None), ("clean_ops", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:4px 0 6px 0;">
  <svg width="30" height="30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="15" cy="15" r="11" stroke="#00C9A7" stroke-width="1.6" fill="none"/>
    <circle cx="15" cy="15" r="5.5" stroke="#00C9A7" stroke-width="1.6" fill="none"/>
    <circle cx="15" cy="15" r="1.6" fill="#00C9A7"/>
    <line x1="15" y1="0" x2="15" y2="7" stroke="#00C9A7" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="15" y1="23" x2="15" y2="30" stroke="#00C9A7" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="0" y1="15" x2="7" y2="15" stroke="#00C9A7" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="23" y1="15" x2="30" y2="15" stroke="#00C9A7" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="15" y1="7" x2="15" y2="9.5" stroke="#00B4D8" stroke-width="1" stroke-linecap="round" opacity="0.6"/>
    <line x1="15" y1="20.5" x2="15" y2="23" stroke="#00B4D8" stroke-width="1" stroke-linecap="round" opacity="0.6"/>
    <line x1="7" y1="15" x2="9.5" y2="15" stroke="#00B4D8" stroke-width="1" stroke-linecap="round" opacity="0.6"/>
    <line x1="20.5" y1="15" x2="23" y2="15" stroke="#00B4D8" stroke-width="1" stroke-linecap="round" opacity="0.6"/>
  </svg>
  <span style="font-size:1.35rem;font-weight:800;color:#FFFFFF;letter-spacing:-0.3px;line-height:1;">Data Profiler</span>
</div>
""", unsafe_allow_html=True)
    st.caption("Upload a file to profile, clean, and explore your data.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files",
    )

    df_raw = None

    if uploaded_file:
        file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

        if file_ext == "csv":
            col_enc, col_sep = st.columns(2)
            with col_enc:
                encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1"],
                                        label_visibility="collapsed")
            with col_sep:
                sep_choice = st.selectbox("Separator", [",", ";", "|", "Tab"],
                                          label_visibility="collapsed")
            separator = "\t" if sep_choice == "Tab" else sep_choice
            try:
                df_raw = pd.read_csv(uploaded_file, encoding=encoding, sep=separator)
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            try:
                xls = pd.ExcelFile(uploaded_file)
                sheet = st.selectbox("Sheet", xls.sheet_names)
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)
            except Exception as e:
                st.error(f"Could not read file: {e}")

        # Reset clean state when a new file is uploaded
        if df_raw is not None:
            current_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.file_key != current_key:
                st.session_state.file_key = current_key
                st.session_state.df_clean = None
                st.session_state.clean_ops = []

    if df_raw is not None:
        is_cleaned = st.session_state.df_clean is not None
        st.success(f"✓  {uploaded_file.name}")
        st.caption(f"{len(df_raw):,} rows × {len(df_raw.columns):,} columns")
        if is_cleaned:
            st.info(f"🧹 {len(st.session_state.clean_ops)} cleaning op(s) applied")


# ── No file — welcome page ─────────────────────────────────────────────────────
if df_raw is None:
    st.markdown("""
<div class="welcome-hero">
  <h1>
    <svg width="46" height="46" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg"
         style="vertical-align:middle;margin-right:10px;margin-bottom:4px;">
      <!-- Eye body -->
      <path d="M4 26 Q26 7 48 26 Q26 45 4 26 Z"
            stroke="#00C9A7" stroke-width="2.8" fill="none" stroke-linejoin="round"/>
      <!-- Iris ring -->
      <circle cx="26" cy="26" r="9" stroke="#00C9A7" stroke-width="2.8" fill="none"/>
      <!-- Pupil -->
      <circle cx="26" cy="26" r="3.5" fill="#00C9A7"/>
      <!-- Sparkle top-right (large) -->
      <path d="M40 6 L41.4 10.6 L46 12 L41.4 13.4 L40 18 L38.6 13.4 L34 12 L38.6 10.6 Z"
            fill="#00B4D8"/>
      <!-- Sparkle top-left (medium) -->
      <path d="M12 7 L13 10.7 L16.7 11.7 L13 12.7 L12 16.4 L11 12.7 L7.3 11.7 L11 10.7 Z"
            fill="#00B4D8" opacity="0.8"/>
      <!-- Sparkle bottom-right (small) -->
      <path d="M43 37 L43.8 39.5 L46.3 40.3 L43.8 41.1 L43 43.6 L42.2 41.1 L39.7 40.3 L42.2 39.5 Z"
            fill="#00C9A7" opacity="0.7"/>
      <!-- Sparkle bottom-left (tiny) -->
      <path d="M9 37 L9.6 38.9 L11.5 39.5 L9.6 40.1 L9 42 L8.4 40.1 L6.5 39.5 L8.4 38.9 Z"
            fill="#00B4D8" opacity="0.55"/>
    </svg>ClearView
  </h1>
  <p>Upload any CSV or Excel file to instantly profile its quality, explore patterns, and clean it — all in one place.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="feature-grid">
  <div class="feature-card">
    <div class="icon">🔍</div>
    <h3>Profile</h3>
    <p>Data quality overview — types, missing values, duplicates, statistics, unique values, special characters, and correlations.</p>
    <div class="tab-hint">→ first tab after upload</div>
  </div>
  <div class="feature-card">
    <div class="icon">📈</div>
    <h3>Explore</h3>
    <p>Auto-generated EDA charts for every column — distributions, value counts, time series, and a relationship explorer.</p>
    <div class="tab-hint">→ second tab after upload</div>
  </div>
  <div class="feature-card">
    <div class="icon">🧹</div>
    <h3>Clean</h3>
    <p>Preview changes before applying them — strip spaces, remove duplicates, parse dates. Download cleaned data as CSV or Excel.</p>
    <div class="tab-hint">→ third tab after upload &nbsp;·&nbsp; optional</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("**How it works**")
    st.markdown("""
<div class="steps-row">
  <div class="step"><span>📁</span>Upload file</div>
  <div class="step-arrow">›</div>
  <div class="step"><span>🔍</span>Profile data</div>
  <div class="step-arrow">›</div>
  <div class="step"><span>📈</span>Explore patterns</div>
  <div class="step-arrow">›</div>
  <div class="step"><span>🧹</span>Clean issues</div>
  <div class="step-arrow">›</div>
  <div class="step"><span>📥</span>Download report</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>**Supported formats**", unsafe_allow_html=True)
    st.markdown("""
<div class="formats-row">
  <span class="fmt-badge">CSV</span>
  <span class="fmt-badge">Excel .xlsx</span>
  <span class="fmt-badge">Excel .xls</span>
</div>
""", unsafe_allow_html=True)

    st.stop()


# ── Active dataframe ───────────────────────────────────────────────────────────
df_active = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw

if df_active.empty:
    st.warning("The uploaded file contains no rows. Please upload a file with data.")
    st.stop()


# ── Summary metrics ────────────────────────────────────────────────────────────
total_cells = df_active.shape[0] * df_active.shape[1]
missing_cells = int(df_active.isnull().sum().sum())
missing_pct = missing_cells / total_cells * 100 if total_cells else 0
dup_count = int(df_active.duplicated().sum())
num_col_count = len(df_active.select_dtypes(include=np.number).columns)
cat_col_count = len(df_active.select_dtypes(include=["object", "string"]).columns)
dt_col_count = len(df_active.select_dtypes(include=["datetime64"]).columns)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Rows", f"{len(df_active):,}")
c2.metric("Columns", f"{len(df_active.columns):,}")
c3.metric("Numeric", num_col_count)
c4.metric("Text", cat_col_count)
c5.metric("Datetime", dt_col_count)
c6.metric("Missing", f"{missing_pct:.1f}%",
          delta=f"{missing_cells:,} cells", delta_color="inverse")
c7.metric("Duplicates", f"{dup_count:,}", delta_color="inverse")
st.divider()


# ── Cached report generators ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame) -> dict:
    return generate_profile(df)

@st.cache_data(show_spinner=False)
def cached_excel_report(profile: dict) -> bytes:
    return generate_excel_report(profile)

@st.cache_data(show_spinner=False)
def cached_html_report(df: pd.DataFrame, profile: dict, filename: str) -> bytes:
    return generate_html_report(df, profile, filename).encode("utf-8")

_is_new_profile = "profile_file_key" not in st.session_state or \
    st.session_state.profile_file_key != st.session_state.file_key or \
    st.session_state.get("profile_n_ops") != len(st.session_state.clean_ops)

if _is_new_profile:
    with st.status("Analysing your data…", expanded=True) as _status:
        st.write("Checking data types and statistics…")
        st.write("Scanning for missing values and duplicates…")
        st.write("Detecting special characters and correlations…")
        profile = cached_profile(df_active)
        _status.update(label="Analysis complete!", state="complete", expanded=False)
    st.session_state.profile_file_key = st.session_state.file_key
    st.session_state.profile_n_ops = len(st.session_state.clean_ops)
else:
    profile = cached_profile(df_active)

# ── EDA chart builder (must be defined before it is called below) ──────────────
@st.cache_data(show_spinner=False)
def generate_eda_charts(df: pd.DataFrame) -> dict:
    charts = {"numeric": {}, "categorical": {}, "datetime": {}, "correlation": None}

    for col in df.select_dtypes(include=np.number).columns:
        fig = px.histogram(df, x=col, marginal="box", nbins=40, template="plotly_white",
                           opacity=0.85)
        fig.update_layout(title=col, margin=dict(l=0, r=0, t=40, b=0), height=320)
        charts["numeric"][col] = fig

    for col in df.select_dtypes(include=["object", "string"]).columns:
        vc = df[col].value_counts().head(20).reset_index()
        vc.columns = [col, "Count"]
        fig = px.bar(vc, x=col, y="Count", template="plotly_white")
        fig.update_layout(title=col, margin=dict(l=0, r=0, t=40, b=20),
                          height=320, xaxis_tickangle=-35)
        charts["categorical"][col] = fig

    for col in df.select_dtypes(include=["datetime64"]).columns:
        ts = df[[col]].dropna().set_index(col)
        try:
            ts_monthly = ts.resample("ME").size().reset_index()
        except Exception:
            ts_monthly = ts.resample("M").size().reset_index()
        ts_monthly.columns = [col, "Records"]
        fig = px.line(ts_monthly, x=col, y="Records", markers=True, template="plotly_white")
        fig.update_layout(title=col, margin=dict(l=0, r=0, t=40, b=0), height=320)
        charts["datetime"][col] = fig

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) >= 2:
        corr = df[num_cols].dropna().corr().round(3)
        z = corr.values
        fig = go.Figure(go.Heatmap(
            z=z, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdBu", zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}", showscale=True,
        ))
        fig.update_layout(
            title="Correlation Matrix",
            margin=dict(l=0, r=0, t=50, b=0),
            height=max(350, len(num_cols) * 55),
            template="plotly_white",
        )
        charts["correlation"] = fig

    return charts


# ── EDA charts (generated before tabs so Explore is instant on click) ──────────
_is_new_eda = "eda_file_key" not in st.session_state or \
    st.session_state.eda_file_key != st.session_state.file_key or \
    st.session_state.get("eda_n_ops") != len(st.session_state.clean_ops)

if _is_new_eda:
    with st.status("Building Explore charts…", expanded=True) as _eda_status:
        st.write("Plotting numeric distributions…")
        st.write("Counting categorical values…")
        st.write("Rendering correlation matrix…")
        eda_charts = generate_eda_charts(df_active)
        _eda_status.update(label="Explore ready!", state="complete", expanded=False)
    st.session_state.eda_file_key = st.session_state.file_key
    st.session_state.eda_n_ops = len(st.session_state.clean_ops)
else:
    eda_charts = generate_eda_charts(df_active)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_profile, tab_explore, tab_clean = st.tabs(["🔍 Profile", "📈 Explore", "🧹 Clean"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:

    if st.session_state.df_clean is not None:
        ops_text = " · ".join(st.session_state.clean_ops)
        st.markdown(
            f'<div class="clean-banner">🧹 Showing <strong>cleaned data</strong> — {ops_text}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📋 Data Types", expanded=True):
        st.dataframe(profile["1_Data_Types"], use_container_width=True, hide_index=True)

    with st.expander("📐 Numerical Statistics"):
        ns = profile["2_Numerical_Stats"]
        if ns.empty:
            st.info("No numerical columns found.")
        else:
            st.dataframe(ns, use_container_width=True, hide_index=True)

    with st.expander("🔤 Unique Values (Text Columns)"):
        st.dataframe(profile["3_Unique_Values"], use_container_width=True, hide_index=True)

    with st.expander("⚠️ Special Characters"):
        sc = profile["4_Special_Characters"]
        if sc.empty:
            st.success("No special characters found.")
        else:
            # Summary per column
            summary = (
                sc.groupby("Column")
                .agg(Affected_Values=("Row Index", "count"))
                .reset_index()
                .rename(columns={"Affected_Values": "Affected Values"})
            )
            st.warning(
                f"{len(sc):,} value(s) with special characters across "
                f"**{summary['Column'].nunique()}** column(s)."
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # Sample: 3 rows per column
            st.caption("Sample (up to 3 per column):")
            sample = sc.groupby("Column").head(3).reset_index(drop=True)
            st.dataframe(sample, use_container_width=True, hide_index=True)

            # Full list on demand
            with st.expander(f"View full list ({len(sc):,} rows) + download"):
                st.dataframe(sc, use_container_width=True, hide_index=True)
                ts_sc = datetime.now().strftime("%Y%m%d_%H%M")
                fname_sc = uploaded_file.name.rsplit(".", 1)[0]
                st.download_button(
                    "📥 Download special characters list (CSV)",
                    data=sc.to_csv(index=False).encode("utf-8"),
                    file_name=f"special_chars_{fname_sc}_{ts_sc}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with st.expander("📅 Datetime Columns"):
        dt = profile["5_Datetime_Columns"]
        if dt.empty:
            st.info("No datetime columns detected. Use the 🧹 Clean tab to parse date columns.")
        else:
            st.dataframe(dt, use_container_width=True, hide_index=True)

    with st.expander("❌ Missing Values"):
        mv = profile["6_Missing_Values"]
        no_missing = len(mv) == 1 and mv.iloc[0]["Column"] == "—"
        if no_missing:
            st.success("No missing values found.")
        else:
            st.warning(f"{missing_cells:,} missing cell(s) across {len(mv)} column(s).")
            st.dataframe(mv, use_container_width=True, hide_index=True)
            fig_mv = px.bar(mv, x="Column", y="Missing Count",
                            color="Missing Count", color_continuous_scale="reds",
                            text="Missing %", height=260)
            fig_mv.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_mv, use_container_width=True)

    with st.expander("🔁 Duplicate Rows"):
        dr = profile["7_Duplicate_Rows"]
        if "Info" in dr.columns:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"{dup_count:,} duplicate row(s) found (showing all occurrences).")
            st.dataframe(dr, use_container_width=True, hide_index=True)

    with st.expander("🔗 Correlation Matrix"):
        corr_df = profile["8_Correlation_Matrix"]
        if "Info" in corr_df.columns or "Error" in corr_df.columns:
            st.info(corr_df.iloc[0, 0])
        else:
            corr_cols = [c for c in corr_df.columns if c != "Column"]
            z = corr_df[corr_cols].values
            fig_corr = go.Figure(go.Heatmap(
                z=z, x=corr_cols, y=corr_df["Column"].tolist(),
                colorscale="RdBu", zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}", showscale=True,
            ))
            fig_corr.update_layout(
                height=max(300, len(corr_cols) * 50),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    fname_stem = uploaded_file.name.rsplit(".", 1)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    excel_bytes = cached_excel_report(profile)
    st.download_button(
        "📥 Download Profile Report (Excel)",
        data=excel_bytes,
        file_name=f"profile_{fname_stem}_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLEAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_clean:

    fname_stem = uploaded_file.name.rsplit(".", 1)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Status banner ──────────────────────────────────────────────────────────
    if st.session_state.clean_ops:
        ops_chips = "".join(
            f'<span class="op-chip">✓ {op}</span>'
            for op in st.session_state.clean_ops
        )
        st.markdown(
            f'<div class="clean-banner">🧹 <strong>Cleaning applied:</strong> {ops_chips}</div>',
            unsafe_allow_html=True,
        )
        if st.button("↩ Reset to original data", key="reset_clean"):
            st.session_state.df_clean = None
            st.session_state.clean_ops = []
            st.rerun()
    else:
        st.info("No cleaning applied yet. Review the sections below, preview each change, then apply.")

    st.divider()

    # ── 1. Strip Extra Spaces ──────────────────────────────────────────────────
    st.markdown("### ✂️ Strip Extra Spaces")
    st.caption("Removes leading/trailing whitespace and reduces multiple internal spaces to one.")

    spaces_preview = preview_strip_spaces(df_active)
    spaces_already = "Strip extra spaces" in st.session_state.clean_ops

    if spaces_preview.empty:
        st.success("No extra spaces found in any text column.")
    elif spaces_already:
        st.success(f"✓ Applied — no extra spaces remaining.")
    else:
        affected_cols = spaces_preview["Column"].nunique()
        st.warning(
            f"{len(spaces_preview):,} value(s) across **{affected_cols}** column(s) have extra spaces."
        )
        with st.expander("Preview changes"):
            st.dataframe(spaces_preview, use_container_width=True, hide_index=True)

        if st.button("Apply — Strip Spaces", key="apply_spaces", type="primary"):
            st.session_state.df_clean = apply_strip_spaces(df_active)
            st.session_state.clean_ops.append("Strip extra spaces")
            st.rerun()

    st.divider()

    # ── 2. Remove Duplicates ───────────────────────────────────────────────────
    st.markdown("### 🔁 Remove Duplicate Rows")
    st.caption("Identifies and removes rows where every column value is identical.")

    keep_choice = st.radio(
        "Which occurrence to keep",
        ["First", "Last"],
        horizontal=True,
        key="keep_choice",
    )
    keep_val = keep_choice.lower()

    dupes_preview = preview_remove_duplicates(df_active, keep=keep_val)
    dupes_already = "Remove duplicates" in st.session_state.clean_ops

    if dupes_preview.empty:
        st.success("No duplicate rows found.")
    elif dupes_already:
        st.success("✓ Applied — duplicates removed.")
    else:
        remaining = len(df_active) - len(dupes_preview)
        st.warning(
            f"**{len(dupes_preview):,}** duplicate row(s) will be removed. "
            f"{remaining:,} rows will remain."
        )
        with st.expander("Preview rows to be removed"):
            st.dataframe(dupes_preview, use_container_width=True, hide_index=True)

        if st.button("Apply — Remove Duplicates", key="apply_dupes", type="primary"):
            st.session_state.df_clean = apply_remove_duplicates(df_active, keep=keep_val)
            st.session_state.clean_ops.append("Remove duplicates")
            st.rerun()

    st.divider()

    # ── 3. Parse Date Columns ──────────────────────────────────────────────────
    st.markdown("### 📅 Parse Date Columns")
    st.caption(
        "Converts text columns that contain dates into proper datetime values. "
        "This unlocks time series charts in the Explore tab. "
        "Values that can't be parsed will become blank."
    )

    obj_cols = df_active.select_dtypes(include=["object"]).columns.tolist()
    date_cols_pick = st.multiselect(
        "Select columns to parse as dates",
        options=obj_cols,
        key="date_cols_pick",
    )
    dates_already = "Parse date columns" in st.session_state.clean_ops

    if dates_already:
        st.success("✓ Applied — date columns parsed.")
    elif date_cols_pick:
        dates_preview = preview_parse_dates(df_active, date_cols_pick)
        with st.expander("Preview parsing results", expanded=True):
            st.dataframe(dates_preview, use_container_width=True, hide_index=True)

        if st.button("Apply — Parse Dates", key="apply_dates", type="primary"):
            st.session_state.df_clean = apply_parse_dates(df_active, date_cols_pick)
            st.session_state.clean_ops.append("Parse date columns")
            st.rerun()
    else:
        st.info("Select one or more text columns above to preview how they will be parsed.")

    st.divider()

    # ── Download cleaned data ──────────────────────────────────────────────────
    st.markdown("### 📥 Download Data")

    is_cleaned = st.session_state.df_clean is not None
    if is_cleaned:
        st.success(
            f"Downloading the **cleaned** version of your data "
            f"({len(df_active):,} rows × {len(df_active.columns):,} columns)."
        )
    else:
        st.info("Apply a cleaning operation above to enable the download buttons.")

    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "📥 Download as CSV",
            data=to_csv_bytes(df_active) if is_cleaned else b"",
            file_name=f"cleaned_{fname_stem}_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not is_cleaned,
        )
    with col_xlsx:
        st.download_button(
            "📥 Download as Excel",
            data=to_excel_bytes(df_active, sheet_name="Cleaned Data") if is_cleaned else b"",
            file_name=f"cleaned_{fname_stem}_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            disabled=not is_cleaned,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE  (tab_explore is the second tab in the tabs list)
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:

    if st.session_state.df_clean is not None:
        ops_text = " · ".join(st.session_state.clean_ops)
        st.markdown(
            f'<div class="clean-banner">🧹 Exploring <strong>cleaned data</strong> — {ops_text}</div>',
            unsafe_allow_html=True,
        )

    num_cols_list = df_active.select_dtypes(include=np.number).columns.tolist()

    # ── Numeric distributions ──────────────────────────────────────────────────
    if eda_charts["numeric"]:
        st.markdown("### 📊 Numeric Columns")
        st.caption("Distribution (histogram + box) for every numeric column.")
        cols_grid = st.columns(2)
        for i, (col, fig) in enumerate(eda_charts["numeric"].items()):
            with cols_grid[i % 2]:
                st.plotly_chart(fig, use_container_width=True)
        st.divider()
    else:
        st.info("No numeric columns found — numeric distributions are not available.")

    # ── Categorical value counts ───────────────────────────────────────────────
    if eda_charts["categorical"]:
        st.markdown("### 🔤 Text Columns")
        st.caption("Top 20 most frequent values for every text column.")
        cols_grid = st.columns(2)
        for i, (col, fig) in enumerate(eda_charts["categorical"].items()):
            with cols_grid[i % 2]:
                st.plotly_chart(fig, use_container_width=True)
        st.divider()
    else:
        st.info("No text columns found — categorical charts are not available.")

    # ── Datetime time series ───────────────────────────────────────────────────
    if eda_charts["datetime"]:
        st.markdown("### 📅 Datetime Columns")
        st.caption("Monthly record counts over time.")
        cols_grid = st.columns(2)
        for i, (col, fig) in enumerate(eda_charts["datetime"].items()):
            with cols_grid[i % 2]:
                st.plotly_chart(fig, use_container_width=True)
        st.divider()

    # ── Correlation heatmap ────────────────────────────────────────────────────
    if eda_charts["correlation"] is not None:
        st.markdown("### 🔗 Correlation Matrix")
        st.plotly_chart(eda_charts["correlation"], use_container_width=True)
        st.divider()

    # ── Relationship explorer (interactive) ────────────────────────────────────
    if len(num_cols_list) < 2:
        st.info("At least 2 numeric columns are needed for the Correlation Matrix and Relationship Explorer.")
    if len(num_cols_list) >= 2:
        st.markdown("### 🔀 Relationship Explorer")
        st.caption("Pick any two numeric columns to explore their relationship.")
        rx, ry, rc, rsize = st.columns(4)
        with rx:
            x_col = st.selectbox("X axis", num_cols_list, key="scatter_x")
        with ry:
            y_col = st.selectbox("Y axis", num_cols_list,
                                 index=min(1, len(num_cols_list) - 1), key="scatter_y")
        with rc:
            sc_color = st.selectbox(
                "Color by",
                ["None"] + df_active.select_dtypes(include=["object", "string"]).columns.tolist(),
                key="sc_color",
            )
        with rsize:
            sc_size = st.selectbox("Size by", ["None"] + num_cols_list, key="sc_size")

        sc_color = None if sc_color == "None" else sc_color
        sc_size = None if sc_size == "None" else sc_size

        try:
            fig_scatter = px.scatter(df_active, x=x_col, y=y_col, color=sc_color,
                                     size=sc_size, trendline="ols", opacity=0.7,
                                     template="plotly_white")
        except Exception:
            st.caption("⚠️ Trend line could not be fitted to this data.")
            fig_scatter = px.scatter(df_active, x=x_col, y=y_col, color=sc_color,
                                     size=sc_size, opacity=0.7, template="plotly_white")

        fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.divider()

    # ── HTML report download ───────────────────────────────────────────────────
    fname_stem = uploaded_file.name.rsplit(".", 1)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    html_bytes = cached_html_report(df_active, profile, uploaded_file.name)

    st.download_button(
        "📥 Download Full HTML Report",
        data=html_bytes,
        file_name=f"report_{fname_stem}_{ts}.html",
        mime="text/html",
        use_container_width=True,
    )