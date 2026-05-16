import io
import re

import pandas as pd


# ── Strip extra spaces ─────────────────────────────────────────────────────────

def preview_strip_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """Returns every value that would change, with before/after columns."""
    rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        for idx, val in df[col].items():
            if isinstance(val, str):
                cleaned = re.sub(r"\s+", " ", val.strip())
                if cleaned != val:
                    rows.append({
                        "Column": col,
                        "Row Index": idx,
                        "Before": val,
                        "After": cleaned,
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Column", "Row Index", "Before", "After"]
    )


def apply_strip_spaces(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].apply(
            lambda x: re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x
        )
    return out


# ── Remove duplicates ──────────────────────────────────────────────────────────

def preview_remove_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """Returns the rows that would be dropped."""
    return df[df.duplicated(keep=keep)].copy()


def apply_remove_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    return df.drop_duplicates(keep=keep).reset_index(drop=True)


# ── Parse date columns ─────────────────────────────────────────────────────────

def preview_parse_dates(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Returns a sample table of original vs parsed values for each column."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue

        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
        original_nulls = int(df[col].isna().sum())
        parsed_nulls = int(parsed.isna().sum())
        fail_count = max(0, parsed_nulls - original_nulls)
        success_count = int(df[col].notna().sum()) - fail_count

        # Up to 5 non-null sample rows — use .loc[] since idx is a label, not a position
        sample_idx = df[col].dropna().head(5).index
        for idx in sample_idx:
            orig = df[col].loc[idx]
            conv = parsed.loc[idx]
            rows.append({
                "Column": col,
                "Original Value": str(orig),
                "Parsed As": str(conv.date()) if pd.notna(conv) else "⚠️  Failed to parse",
                "Status": "✓" if pd.notna(conv) else "⚠️",
            })

        rows.append({
            "Column": col,
            "Original Value": "── Summary ──",
            "Parsed As": (
                f"{success_count:,} will parse successfully  ·  "
                f"{fail_count:,} will fail and become blank"
            ),
            "Status": "",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Column", "Original Value", "Parsed As", "Status"]
    )


def apply_parse_dates(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


# ── Download helpers ───────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        wb = writer.book
        header_fmt = wb.add_format({
            "bold": True,
            "bg_color": "#1f77b4",
            "font_color": "#ffffff",
            "border": 1,
        })
        df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=1, header=False)
        ws = writer.sheets[sheet_name]
        for i, col in enumerate(df.columns):
            ws.write(0, i, col, header_fmt)
        ws.set_column(0, len(df.columns) - 1, 22)
    return buffer.getvalue()