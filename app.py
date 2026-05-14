"""
LLM Vocabulary Coverage Dashboard — Gradio app for Hugging Face Spaces.

Loads pre-computed coverage JSON files from data/coverage/ and renders
interactive Plotly charts plus a searchable language table.

Pre-compute coverage files with:
  python scripts/ingest_model.py --model gpt-4o --source tiktoken --fertility
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
COVERAGE_DIR = ROOT / "data" / "coverage"

GRADE_COLORS = {
    "Excellent": "#22c55e",
    "Good":      "#84cc16",
    "Partial":   "#f59e0b",
    "Poor":      "#ef4444",
}
GRADE_ORDER = ["Excellent", "Good", "Partial", "Poor"]

TIER_PALETTE = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def coverage_grade(score: float) -> str:
    if score >= 0.95:
        return "Excellent"
    if score >= 0.80:
        return "Good"
    if score >= 0.50:
        return "Partial"
    return "Poor"


def list_models() -> list[str]:
    """Return display names of all pre-computed coverage files."""
    return sorted(
        p.stem.replace("__", "/")
        for p in COVERAGE_DIR.glob("*.json")
    )


def load_coverage(model_display: str) -> dict:
    safe = model_display.replace("/", "__").replace(":", "_").replace(" ", "_")
    path = COVERAGE_DIR / f"{safe}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No coverage file found for '{model_display}'. "
            f"Run: python scripts/ingest_model.py --model {model_display} ..."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataframe(data: dict) -> pd.DataFrame:
    """Flatten the coverage JSON into a tidy DataFrame (CLDR languages only)."""
    rows = []
    for locale, lang in data["languages"].items():
        if not lang.get("has_cldr"):
            continue
        main = lang.get("main") or {}
        if not main:
            continue

        score = main.get("weighted_score", 0.0)
        fert  = lang.get("fertility") or {}

        rows.append({
            "locale":         locale,
            "name":           lang.get("name", locale),
            "script":         lang.get("script") or "—",
            "family":         lang.get("family_name") or "Unknown",
            "macroarea":      lang.get("macroarea")   or "Unknown",
            "iso639_3":       lang.get("iso639_3")    or "",
            "glottocode":     lang.get("glottocode")  or "",
            "score":          round(score, 4),
            "grade":          coverage_grade(score),
            "tier0":          main.get("tier0_count", 0),
            "tier1":          main.get("tier1_count", 0),
            "tier2":          main.get("tier2_count", 0),
            "tier3":          main.get("tier3_count", 0),
            "total_chars":    main.get("total", 0),
            # tier2/3 codepoint lists for drill-down
            "tier2_cps":      main.get("tier2", []),
            "tier3_cps":      main.get("tier3", []),
            "tokens_per_char": float(fert["tokens_per_char"]) if fert.get("tokens_per_char") is not None else None,
            "tokens_per_word": float(fert["tokens_per_word"]) if fert.get("tokens_per_word") is not None else None,
            "sample_chars":   fert.get("sample_chars"),
            "latitude":       lang.get("latitude"),
            "longitude":      lang.get("longitude"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["grade"] = pd.Categorical(df["grade"], categories=GRADE_ORDER, ordered=True)
    return df.sort_values("score").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------

def make_summary_html(data: dict, df: pd.DataFrame) -> str:
    n = len(df)
    if n == 0:
        return "<p>No CLDR language data found.</p>"

    counts = df["grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    has_fertility = df["tokens_per_char"].notna().any()
    computed = (data.get("computed_at") or "")[:10]

    pct = lambda k: f"{100 * counts[k] // n}%"  # noqa: E731

    grade_rows = "".join(
        f"""<tr>
          <td><span style="display:inline-block;width:12px;height:12px;
              border-radius:50%;background:{GRADE_COLORS[g]};margin-right:6px"></span>{g}</td>
          <td style="text-align:right">{counts[g]}</td>
          <td style="text-align:right;color:#6b7280">{pct(g)}</td>
        </tr>"""
        for g in GRADE_ORDER
    )

    return f"""
<div style="display:flex;gap:32px;flex-wrap:wrap;font-family:sans-serif">

  <div style="min-width:220px">
    <h3 style="margin:0 0 12px">{data['model_id']}</h3>
    <table style="border-collapse:collapse;width:100%">
      <tr><td style="color:#6b7280;padding:3px 8px 3px 0">Source</td>
          <td><b>{data['source']}</b></td></tr>
      <tr><td style="color:#6b7280;padding:3px 8px 3px 0">Vocab size</td>
          <td><b>{data['vocab_size']:,}</b></td></tr>
      <tr><td style="color:#6b7280;padding:3px 8px 3px 0">Byte fallback</td>
          <td><b>{"Yes" if data['has_byte_fallback'] else "No"}</b></td></tr>
      <tr><td style="color:#6b7280;padding:3px 8px 3px 0">Computed</td>
          <td><b>{computed}</b></td></tr>
      <tr><td style="color:#6b7280;padding:3px 8px 3px 0">CLDR languages</td>
          <td><b>{n}</b></td></tr>
      <tr><td style="color:#6b7280;padding:3px 8px 3px 0">Fertility data</td>
          <td><b>{"Yes" if has_fertility else "No"}</b></td></tr>
    </table>
  </div>

  <div style="min-width:240px">
    <h3 style="margin:0 0 12px">Coverage grades</h3>
    <table style="border-collapse:collapse;width:100%">
      <tr><th style="text-align:left;color:#6b7280;font-weight:normal">Grade</th>
          <th style="text-align:right;color:#6b7280;font-weight:normal">Count</th>
          <th style="text-align:right;color:#6b7280;font-weight:normal">Share</th></tr>
      {grade_rows}
    </table>
    <p style="margin:8px 0 0;font-size:0.85em;color:#6b7280">
      Excellent ≥ 0.95 · Good ≥ 0.80 · Partial ≥ 0.50 · Poor &lt; 0.50
    </p>
  </div>

</div>
"""


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def make_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart: count of languages per grade."""
    counts = (
        df.groupby("grade", observed=True)
        .size()
        .reindex(GRADE_ORDER, fill_value=0)
        .reset_index(name="count")
    )
    counts.columns = ["grade", "count"]

    fig = px.bar(
        counts,
        x="grade", y="count",
        color="grade",
        color_discrete_map=GRADE_COLORS,
        category_orders={"grade": GRADE_ORDER},
        text="count",
        labels={"grade": "Coverage Grade", "count": "Number of Languages"},
        title="Languages by Coverage Grade",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white",
                      yaxis=dict(gridcolor="#f0f0f0"))
    return fig


def make_score_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of weighted coverage scores."""
    fig = px.histogram(
        df, x="score",
        nbins=50,
        color="grade",
        color_discrete_map=GRADE_COLORS,
        category_orders={"grade": GRADE_ORDER},
        labels={"score": "Weighted Coverage Score", "count": "Languages"},
        title="Score Distribution Across All Languages",
        opacity=0.9,
    )
    fig.update_layout(bargap=0.02, plot_bgcolor="white",
                      yaxis=dict(gridcolor="#f0f0f0"),
                      xaxis=dict(range=[0, 1]))
    return fig


def make_tier_stacked_bar(df: pd.DataFrame) -> go.Figure:
    """
    Stacked percentage bar chart of tier character counts per language,
    sorted by coverage score. Only the 60 worst languages are shown to
    keep the chart readable. Hover on Tier-2/3 bars shows the actual
    problematic characters.
    """
    worst = df.nsmallest(60, "score").copy()
    worst = worst.sort_values("score", ascending=True)

    totals = worst[["tier0", "tier1", "tier2", "tier3"]].sum(axis=1).clip(lower=1)
    labels = worst["name"] + " (" + worst["locale"] + ")"

    def _fmt_hover(cps: list, limit: int = 14) -> str:
        if not cps:
            return "none"
        chars = []
        for cp in cps[:limit]:
            try:
                chars.append(f"{chr(cp)}\u2009U+{cp:04X}")
            except (ValueError, OverflowError):
                chars.append(f"U+{cp:04X}")
        extra = f"\u2026 +{len(cps) - limit} more" if len(cps) > limit else ""
        return ",  ".join(chars) + ("  " + extra if extra else "")

    fig = go.Figure()
    for tier, color, label in zip(
        ["tier0", "tier1", "tier2", "tier3"],
        TIER_PALETTE,
        ["Tier 0 — native token", "Tier 1 — embedded", "Tier 2 — byte fallback", "Tier 3 — unreachable"],
    ):
        pct = worst[tier] / totals * 100

        if tier == "tier2":
            custom = worst["tier2_cps"].apply(_fmt_hover).tolist()
            hover = "%{x}<br>" + label + ": %{y:.1f}%<br>Chars: %{customdata}<extra></extra>"
        elif tier == "tier3":
            custom = worst["tier3_cps"].apply(_fmt_hover).tolist()
            hover = "%{x}<br>" + label + ": %{y:.1f}%<br>Chars: %{customdata}<extra></extra>"
        else:
            custom = ["" for _ in range(len(worst))]
            hover = "%{x}<br>" + label + ": %{y:.1f}%<extra></extra>"

        fig.add_trace(go.Bar(
            name=label,
            x=labels,
            y=pct,
            marker_color=color,
            customdata=custom,
            hovertemplate=hover,
        ))

    fig.update_layout(
        barmode="stack",
        title="Character-Tier Breakdown — 60 Lowest-Scoring Languages (hover Tier-2/3 bars to see missing characters)",
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="% of Characters", range=[0, 100]),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=520,
    )
    return fig


def make_tier_detail_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table of the 60 worst-scoring languages with their problematic characters listed."""
    worst = df.nsmallest(60, "score").copy()

    def _fmt(cps: list, limit: int = 20) -> str:
        if not cps:
            return "—"
        chars = []
        for cp in cps[:limit]:
            try:
                chars.append(f"{chr(cp)} (U+{cp:04X})")
            except (ValueError, OverflowError):
                chars.append(f"U+{cp:04X}")
        extra = f"  +{len(cps) - limit} more" if len(cps) > limit else ""
        return "  ".join(chars) + extra

    out = worst[["name", "locale", "score", "grade", "tier2", "tier3",
                 "tier2_cps", "tier3_cps"]].copy()
    out["Byte-Fallback Characters"] = out["tier2_cps"].apply(_fmt)
    out["Unreachable Characters"]   = out["tier3_cps"].apply(_fmt)
    out = out.drop(columns=["tier2_cps", "tier3_cps"])
    out.columns = [
        "Language", "Locale", "Score", "Grade", "T2 Count", "T3 Count",
        "Byte-Fallback Characters", "Unreachable Characters",
    ]
    return out.sort_values("Score").reset_index(drop=True)


def make_world_map(df: pd.DataFrame) -> go.Figure:
    plot_df = df.dropna(subset=["latitude", "longitude"]).copy()
    plot_df["hover_fert"] = plot_df["tokens_per_char"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"
    )

    fig = px.scatter_geo(
        plot_df,
        lat="latitude",
        lon="longitude",
        color="score",
        hover_name="name",
        hover_data={
            "locale":     True,
            "family":     True,
            "macroarea":  True,
            "score":      ":.3f",
            "grade":      True,
            "hover_fert": True,
            "latitude":   False,
            "longitude":  False,
        },
        color_continuous_scale="RdYlGn",
        range_color=[0.0, 1.0],
        size_max=9,
        projection="natural earth",
        title="Language Coverage Score by Geographic Location",
        labels={"hover_fert": "Tokens/char", "score": "Score"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    fig.update_layout(
        coloraxis_colorbar=dict(title="Score", tickformat=".0%"),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#d1d5db",
                 showland=True, landcolor="#f9fafb",
                 showocean=True, oceancolor="#eff6ff"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_macroarea_chart(df: pd.DataFrame) -> go.Figure:
    area_df = (
        df.groupby("macroarea")
        .agg(avg_score=("score", "mean"), count=("name", "count"))
        .reset_index()
        .sort_values("avg_score")
    )
    area_df.columns = ["Macroarea", "avg_score", "count"]

    fig = px.bar(
        area_df,
        x="avg_score",
        y="Macroarea",
        orientation="h",
        color="avg_score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        text=area_df["count"].apply(lambda n: f"{n} langs"),
        labels={"avg_score": "Avg Coverage Score"},
        title="Average Coverage Score by Macroarea",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white",
                      xaxis=dict(range=[0, 1.15], gridcolor="#f0f0f0"),
                      coloraxis_showscale=False)
    return fig


def make_family_chart(df: pd.DataFrame) -> go.Figure:
    family_df = (
        df.groupby("family")
        .agg(avg_score=("score", "mean"), count=("name", "count"))
        .reset_index()
        .query("count >= 2")
        .sort_values("avg_score")
    )
    family_df.columns = ["Family", "avg_score", "count"]

    fig = px.bar(
        family_df,
        x="avg_score",
        y="Family",
        orientation="h",
        color="avg_score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        hover_data={"count": True, "avg_score": ":.3f"},
        labels={"avg_score": "Avg Score", "Family": "Language Family", "count": "Languages"},
        title="Average Coverage by Language Family (≥ 2 languages)",
    )
    fig.update_layout(
        height=max(500, len(family_df) * 20),
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(range=[0, 1.05], gridcolor="#f0f0f0"),
        coloraxis_showscale=False,
    )
    return fig


def make_fertility_scatter(df: pd.DataFrame) -> go.Figure:
    fdf = df.copy()
    fdf["tokens_per_char"] = pd.to_numeric(fdf["tokens_per_char"], errors="coerce")
    fdf = fdf.dropna(subset=["tokens_per_char"]).copy()
    if fdf.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No fertility data available for this model.<br>"
                 "Re-run with: python scripts/ingest_model.py ... --fertility",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#6b7280"),
        )
        fig.update_layout(title="Fertility vs Coverage Score")
        return fig

    fig = px.scatter(
        fdf,
        x="score",
        y="tokens_per_char",
        color="macroarea",
        hover_name="name",
        hover_data={
            "locale":          True,
            "family":          True,
            "tokens_per_word": ":.2f",
            "score":           ":.3f",
            "macroarea":       False,
        },
        labels={
            "score":           "Coverage Score",
            "tokens_per_char": "Tokens per Character",
            "macroarea":       "Macroarea",
        },
        title="Fertility vs Coverage Score  (lower fertility = better tokenization)",
        opacity=0.75,
    )
    # Add a reference line at tokens/char = 1.0 (ideal)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8",
                  annotation_text="1 token / char (ideal)", annotation_position="right")
    fig.update_layout(plot_bgcolor="white",
                      xaxis=dict(range=[-0.02, 1.05], gridcolor="#f0f0f0"),
                      yaxis=dict(gridcolor="#f0f0f0"))
    return fig


def make_fertility_bar(df: pd.DataFrame) -> go.Figure:
    """Top-N worst fertility languages (highest tokens/char)."""
    fdf = df.copy()
    fdf["tokens_per_char"] = pd.to_numeric(fdf["tokens_per_char"], errors="coerce")
    fdf = fdf.dropna(subset=["tokens_per_char"]).nlargest(30, "tokens_per_char")
    if fdf.empty:
        return go.Figure()

    fig = px.bar(
        fdf.sort_values("tokens_per_char", ascending=True),
        x="tokens_per_char",
        y=fdf.sort_values("tokens_per_char", ascending=True)["name"] + " (" + fdf.sort_values("tokens_per_char", ascending=True)["locale"] + ")",
        orientation="h",
        color="tokens_per_char",
        color_continuous_scale="RdYlGn_r",
        labels={"x": "Tokens / Character", "y": "Language"},
        title="30 Worst-Fertility Languages (most token-expensive)",
        text=fdf.sort_values("tokens_per_char", ascending=True)["tokens_per_char"].apply(lambda x: f"{x:.2f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=max(400, len(fdf) * 22 + 80),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"),
        coloraxis_showscale=False,
        yaxis_title="",
    )
    return fig


# ---------------------------------------------------------------------------
# Incomplete coverage detail
# ---------------------------------------------------------------------------

def make_incomplete_table(df: pd.DataFrame) -> pd.DataFrame:
    bad = (
        df[df["grade"].isin(["Partial", "Poor"])]
        .sort_values("score")
        .copy()
    )
    if bad.empty:
        return pd.DataFrame(columns=["Language", "Locale", "Score", "Grade",
                                     "Tier0", "Tier1", "Tier2", "Tier3",
                                     "Family", "Macroarea"])

    # Decode a few tier2/tier3 codepoints for display
    def format_cps(cps: list[int], limit: int = 8) -> str:
        chars = []
        for cp in cps[:limit]:
            try:
                chars.append(f"U+{cp:04X} ({chr(cp)})")
            except (ValueError, OverflowError):
                chars.append(f"U+{cp:04X}")
        suffix = f" … +{len(cps) - limit} more" if len(cps) > limit else ""
        return ", ".join(chars) + suffix

    out = bad[[
        "name", "locale", "score", "grade",
        "tier0", "tier1", "tier2", "tier3",
        "family", "macroarea", "tier2_cps", "tier3_cps",
    ]].copy()
    out["tier2_chars"] = out["tier2_cps"].apply(lambda x: format_cps(x) if x else "")
    out["tier3_chars"] = out["tier3_cps"].apply(lambda x: format_cps(x) if x else "")
    out = out.drop(columns=["tier2_cps", "tier3_cps"])
    out.columns = [
        "Language", "Locale", "Score", "Grade",
        "T0", "T1", "T2", "T3",
        "Family", "Macroarea",
        "Tier-2 Characters", "Tier-3 Characters",
    ]
    return out.reset_index(drop=True)


def make_full_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["name", "locale", "script", "family", "macroarea",
            "score", "grade", "tier0", "tier1", "tier2", "tier3",
            "total_chars", "tokens_per_char", "tokens_per_word",
            "iso639_3", "glottocode"]
    out = df[cols].copy()
    out.columns = [
        "Language", "Locale", "Script", "Family", "Macroarea",
        "Score", "Grade", "T0", "T1", "T2", "T3",
        "Total Chars", "Tokens/Char", "Tokens/Word",
        "ISO 639-3", "Glottocode",
    ]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Language detail helpers
# ---------------------------------------------------------------------------

def _kv(label: str, value: str) -> str:
    return (
        f'<div style="background:#f9fafb;border-radius:8px;padding:10px 14px">'
        f'<div style="color:#6b7280;font-size:0.8em;margin-bottom:2px">{label}</div>'
        f'<div style="font-weight:600">{value}</div></div>'
    )


def _tier_row(tier: int, count: int, total: int) -> str:
    pct = f"{100 * count / max(total, 1):.1f}%"
    color = TIER_PALETTE[tier]
    label = ["native token", "embedded", "byte fallback", "unreachable"][tier]
    return (
        f'<tr>'
        f'<td style="padding:4px 0"><span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{color};margin-right:6px"></span>'
        f'Tier {tier} \u2014 {label}</td>'
        f'<td style="text-align:right;padding:4px 8px">{count}</td>'
        f'<td style="text-align:right;color:#6b7280">{pct}</td></tr>'
    )


def make_language_card_html(row: pd.Series, data: dict) -> str:
    locale = row["locale"]
    lang_data = data["languages"].get(locale, {})
    main = lang_data.get("main") or {}
    tier2_cps: list[int] = main.get("tier2", [])
    tier3_cps: list[int] = main.get("tier3", [])

    def fmt_cps(cps: list[int], limit: int = 28) -> str:
        items = []
        for cp in cps[:limit]:
            try:
                items.append(f"U+{cp:04X}\u00a0({chr(cp)})")
            except (ValueError, OverflowError):
                items.append(f"U+{cp:04X}")
        extra = f" \u2026 +{len(cps) - limit} more" if len(cps) > limit else ""
        return ",\u2002".join(items) + extra

    grade_color = GRADE_COLORS.get(str(row["grade"]), "#6b7280")
    tpc = row.get("tokens_per_char")
    tpw = row.get("tokens_per_word")

    tier2_section = (
        f'<div style="margin-bottom:16px">'
        f'<h4 style="margin:0 0 8px;color:#f59e0b">Tier-2 Characters '
        f'<small style="font-weight:normal;color:#6b7280">(byte-fallback only)</small></h4>'
        f'<p style="font-family:monospace;font-size:0.9em;line-height:2;margin:0">'
        f'{fmt_cps(tier2_cps)}</p></div>'
    ) if tier2_cps else ""

    tier3_section = (
        f'<div style="margin-bottom:16px">'
        f'<h4 style="margin:0 0 8px;color:#ef4444">Tier-3 Characters '
        f'<small style="font-weight:normal;color:#6b7280">(unreachable)</small></h4>'
        f'<p style="font-family:monospace;font-size:0.9em;line-height:2;margin:0">'
        f'{fmt_cps(tier3_cps)}</p></div>'
    ) if tier3_cps else ""

    no_issues = (
        '<p style="color:#22c55e;font-weight:600">\u2713 All characters are natively tokenized '
        '(Tier\u00a00) \u2014 no byte-fallback or unreachable characters.</p>'
    ) if not tier2_cps and not tier3_cps else ""

    return f"""
<div style="font-family:sans-serif;max-width:960px;padding:8px 0">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap">
    <h2 style="margin:0">{row['name']}</h2>
    <span style="background:{grade_color};color:white;padding:4px 14px;
          border-radius:999px;font-weight:600;font-size:1.05em">{row['grade']}</span>
    <span style="font-size:1.6em;font-weight:700;color:{grade_color}">{row['score']:.4f}</span>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
              gap:10px;margin-bottom:24px">
    {_kv('Locale', row['locale'])}
    {_kv('Script', row['script'])}
    {_kv('ISO 639-3', row['iso639_3'] or '\u2014')}
    {_kv('Glottocode', row['glottocode'] or '\u2014')}
    {_kv('Language Family', row['family'])}
    {_kv('Macroarea', row['macroarea'])}
    {_kv('Total Characters', str(int(row['total_chars'])))}
    {_kv('Tokens / Char', f"{tpc:.3f}" if pd.notna(tpc) else 'n/a')}
    {_kv('Tokens / Word', f"{tpw:.3f}" if pd.notna(tpw) else 'n/a')}
  </div>

  <div style="margin-bottom:24px">
    <h4 style="margin:0 0 10px;color:#374151">Character Tier Summary</h4>
    <table style="border-collapse:collapse;width:100%;max-width:500px">
      <tr><th style="text-align:left;color:#6b7280;font-weight:normal;padding:4px 0">Tier</th>
          <th style="text-align:right;color:#6b7280;font-weight:normal;padding:4px 8px">Count</th>
          <th style="text-align:right;color:#6b7280;font-weight:normal">Share</th></tr>
      {_tier_row(0, int(row['tier0']), int(row['total_chars']))}
      {_tier_row(1, int(row['tier1']), int(row['total_chars']))}
      {_tier_row(2, int(row['tier2']), int(row['total_chars']))}
      {_tier_row(3, int(row['tier3']), int(row['total_chars']))}
    </table>
  </div>

  {no_issues}
  {tier2_section}
  {tier3_section}
</div>
"""


def make_language_tier_pie(row: pd.Series) -> go.Figure:
    labels = ["Tier 0 \u2014 native", "Tier 1 \u2014 embedded",
              "Tier 2 \u2014 byte fallback", "Tier 3 \u2014 unreachable"]
    values = [int(row["tier0"]), int(row["tier1"]),
              int(row["tier2"]), int(row["tier3"])]
    if sum(values) == 0:
        return go.Figure()

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=TIER_PALETTE),
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value} characters<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Tier Distribution \u2014 {row['name']}",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_language_highlighted_map(df: pd.DataFrame, locale: str) -> go.Figure:
    fig = make_world_map(df)
    sel = df[df["locale"] == locale]
    if not sel.empty:
        r = sel.iloc[0]
        if pd.notna(r["latitude"]) and pd.notna(r["longitude"]):
            fig.add_trace(go.Scattergeo(
                lat=[r["latitude"]],
                lon=[r["longitude"]],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color="#1d4ed8",
                    line=dict(color="white", width=2),
                    symbol="star",
                ),
                text=[r["name"]],
                textposition="top center",
                textfont=dict(size=12, color="#1d4ed8"),
                hovertext=(
                    f"{r['name']} ({locale})<br>"
                    f"Score: {r['score']:.4f} \u2014 {r['grade']}<br>"
                    f"Family: {r['family']}"
                ),
                hoverinfo="text",
                showlegend=False,
            ))
    return fig


# ---------------------------------------------------------------------------
# Language quick-view & all-models comparison
# ---------------------------------------------------------------------------

def make_language_quickview_html(row: pd.Series) -> str:
    """Compact badge displayed immediately above the tabs when a language is selected."""
    grade_color = GRADE_COLORS.get(str(row["grade"]), "#6b7280")
    tpc = row.get("tokens_per_char")
    fert_str = f"{tpc:.2f} tokens/char" if pd.notna(tpc) else "n/a"
    return f"""
<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;
            padding:10px 18px;background:#f9fafb;border-radius:10px;
            border-left:5px solid {grade_color};font-family:sans-serif;margin:6px 0">
  <strong style="font-size:1.05em">{row['name']}</strong>
  <span style="background:{grade_color};color:white;padding:3px 14px;
               border-radius:999px;font-weight:600;font-size:0.9em">{row['grade']}</span>
  <span style="font-size:1.15em;font-weight:700;color:{grade_color}">{row['score']:.4f}</span>
  <span style="color:#6b7280;font-size:0.85em">
    {row['family']} &middot; {row['macroarea']} &middot; Fertility: {fert_str}
  </span>
  <span style="color:#94a3b8;font-size:0.8em;margin-left:auto;font-style:italic">
    See &ldquo;Language Detail&rdquo; tab for full breakdown &darr;
  </span>
</div>
"""


# Lazy-loaded cache: locale \u2192 {model_name: (score, grade)}
_MODEL_SCORES: dict = {}
_scores_loaded = False


def _ensure_scores_loaded() -> None:
    global _scores_loaded
    if _scores_loaded:
        return
    for model_name in list_models():
        try:
            data = load_coverage(model_name)
            for locale, lang in data.get("languages", {}).items():
                if not lang.get("has_cldr"):
                    continue
                main = lang.get("main") or {}
                if not main:
                    continue
                score = round(main.get("weighted_score", 0.0), 4)
                grade = coverage_grade(score)
                if locale not in _MODEL_SCORES:
                    _MODEL_SCORES[locale] = {}
                _MODEL_SCORES[locale][model_name] = (score, grade)
        except Exception:
            continue
    _scores_loaded = True


def make_model_comparison_chart(locale: str, lang_name: str) -> go.Figure:
    """Horizontal bar chart: every available model's coverage score for one locale."""
    _ensure_scores_loaded()
    model_data = _MODEL_SCORES.get(locale, {})
    if not model_data:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No coverage data found for locale \u2018{locale}\u2019.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#6b7280"),
        )
        fig.update_layout(title="Model Comparison")
        return fig

    rows = [
        {"model": m, "score": score, "grade": grade}
        for m, (score, grade) in model_data.items()
    ]
    sdf = pd.DataFrame(rows).sort_values("score", ascending=True)

    fig = px.bar(
        sdf,
        x="score",
        y="model",
        orientation="h",
        color="grade",
        color_discrete_map=GRADE_COLORS,
        category_orders={"grade": GRADE_ORDER},
        text=sdf["score"].apply(lambda s: f"{s:.3f}"),
        labels={"score": "Coverage Score", "model": "Model", "grade": "Grade"},
        title=f"Coverage Score Across All Models \u2014 {lang_name} ({locale})",
    )
    fig.add_vline(x=0.95, line_dash="dot", line_color=GRADE_COLORS["Excellent"],
                  annotation_text="Excellent \u22650.95", annotation_position="top right")
    fig.add_vline(x=0.80, line_dash="dot", line_color=GRADE_COLORS["Good"],
                  annotation_text="Good \u22650.80", annotation_position="top right")
    fig.add_vline(x=0.50, line_dash="dot", line_color=GRADE_COLORS["Partial"],
                  annotation_text="Partial \u22650.50", annotation_position="top right")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(420, len(sdf) * 22 + 120),
        plot_bgcolor="white",
        xaxis=dict(range=[0, 1.18], gridcolor="#f0f0f0"),
        showlegend=True,
        legend=dict(title="Grade"),
    )
    return fig


def render_language(model_name: str, locale: str):
    """Render the language detail panel for one locale."""
    if not model_name or not locale:
        return "", "", go.Figure(), go.Figure(), go.Figure()
    # When allow_custom_value=True Gradio may pass the display label
    # e.g. "Bosnian (bs)" instead of the raw locale code "bs".
    # Extract the code from parentheses if needed.
    m = re.search(r'\(([^)]+)\)\s*$', locale)
    if m:
        locale = m.group(1)
    data   = load_coverage(model_name)
    df     = build_dataframe(data)
    row_df = df[df["locale"] == locale]
    if row_df.empty:
        return (
            "",
            f"<p>Language '<b>{locale}</b>' not found in coverage data.</p>",
            go.Figure(), go.Figure(), go.Figure(),
        )
    row = row_df.iloc[0]
    return (
        make_language_quickview_html(row),
        make_language_card_html(row, data),
        make_language_tier_pie(row),
        make_language_highlighted_map(df, locale),
        make_model_comparison_chart(locale, row["name"]),
    )


# ---------------------------------------------------------------------------
# Main render function wired to every control
# ---------------------------------------------------------------------------

def render(model_name: str):
    if not model_name:
        empty_fig = go.Figure()
        empty_df  = pd.DataFrame()
        return ("", empty_fig, empty_fig, empty_fig, empty_df,
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                empty_df, empty_df,
                gr.update(choices=[], value=None))

    data = load_coverage(model_name)
    df   = build_dataframe(data)

    if df.empty:
        msg = f"<p>No CLDR language data found in coverage file for <b>{model_name}</b>.</p>"
        empty_fig = go.Figure()
        empty_df  = pd.DataFrame()
        return (msg, empty_fig, empty_fig, empty_fig, empty_df,
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                empty_df, empty_df,
                gr.update(choices=[], value=None))

    summary_html      = make_summary_html(data, df)
    grade_bar         = make_distribution_chart(df)
    score_hist        = make_score_histogram(df)
    tier_bar          = make_tier_stacked_bar(df)
    tier_detail_tbl   = make_tier_detail_table(df)
    world_map         = make_world_map(df)
    macroarea_bar     = make_macroarea_chart(df)
    family_bar        = make_family_chart(df)
    fert_scatter      = make_fertility_scatter(df)
    fert_bar          = make_fertility_bar(df)
    incomplete_tbl    = make_incomplete_table(df)
    full_tbl          = make_full_table(df)

    lang_choices = [
        (f"{r['name']} ({r['locale']})", r['locale'])
        for _, r in df.sort_values("name").iterrows()
    ]
    first_locale = lang_choices[0][1] if lang_choices else None

    return (
        summary_html,
        grade_bar,
        score_hist,
        tier_bar,
        tier_detail_tbl,
        world_map,
        macroarea_bar,
        family_bar,
        fert_scatter,
        fert_bar,
        incomplete_tbl,
        full_tbl,
        gr.update(choices=lang_choices, value=first_locale),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

models = list_models()
_PREFERRED_DEFAULTS = ["gpt-4o", "gpt-4", "meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3"]
_default_model = next((m for m in _PREFERRED_DEFAULTS if m in models), models[0] if models else None)

_OUTPUTS_COUNT = 13  # must match number of return values in render()

with gr.Blocks(title="LLM Vocabulary Coverage Dashboard") as demo:

    gr.Markdown(
        "# 🌍 LLM Vocabulary Coverage Dashboard\n"
        "How well does a model's tokenizer support the world's languages? "
        "Select a model to see tier-based Unicode coverage, fertility scoring, "
        "and per-language breakdown across ~1 000 CLDR locales."
    )

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=models,
            value=_default_model,
            label="Model",
            scale=3,
        )
        run_btn = gr.Button("Generate Report", variant="primary", scale=1)

    with gr.Row():
        language_dd = gr.Dropdown(
            choices=[],
            value=None,
            label="Language (type to search)",
            interactive=True,
            allow_custom_value=True,
            scale=4,
        )

    # ---- Language quick-view (updates on language select) ----
    lang_quickview = gr.HTML()

    # ---- Summary ----
    summary_html = gr.HTML()

    with gr.Tabs():

        # ── Overview ──────────────────────────────────────────────────────
        with gr.Tab("📊 Overview"):
            with gr.Row():
                grade_bar_plot = gr.Plot(label="Grades")
                score_hist_plot = gr.Plot(label="Score Distribution")
            tier_bar_plot = gr.Plot(label="Tier Breakdown (60 lowest-scoring)")
            gr.Markdown(
                "**Hover over the orange (Tier-2) or red (Tier-3) bars** to see the exact "
                "characters that are only reachable via byte-fallback or are completely unreachable. "
                "The table below lists them explicitly."
            )
            tier_detail_table = gr.Dataframe(
                label="Problematic Characters — 60 Lowest-Scoring Languages",
                interactive=False,
                wrap=True,
            )

        # ── World Map ─────────────────────────────────────────────────────
        with gr.Tab("🗺️ World Map"):
            world_map_plot = gr.Plot()

        # ── By Region / Family ────────────────────────────────────────────
        with gr.Tab("🌐 By Region & Family"):
            macroarea_plot = gr.Plot()
            family_plot    = gr.Plot()

        # ── Fertility ─────────────────────────────────────────────────────
        with gr.Tab("🔬 Fertility"):
            gr.Markdown(
                "**Fertility** = tokens per non-whitespace character on UDHR text. "
                "Lower is better — the model learned multi-character subwords for the language. "
                "A score > 3–4 tokens/char means the model is essentially spelling every character "
                "out in bytes. Requires re-running with `--fertility` flag."
            )
            fert_scatter_plot = gr.Plot()
            fert_bar_plot     = gr.Plot()

        # ── Incomplete Coverage ───────────────────────────────────────────
        with gr.Tab("⚠️ Incomplete Coverage"):
            gr.Markdown(
                "Languages with **Partial** (score 0.50–0.80) or **Poor** (< 0.50) coverage. "
                "The *Tier-2 Characters* column shows the specific Unicode codepoints that only "
                "reach the model via byte-fallback tokens."
            )
            incomplete_table = gr.Dataframe(
                interactive=False,
                wrap=True,
                column_widths=["14%", "6%", "6%", "7%",
                                "4%", "4%", "4%", "4%",
                                "14%", "9%", "16%", "12%"],
            )

        # ── Full Table ────────────────────────────────────────────────────
        with gr.Tab("📋 Full Language Table"):
            gr.Markdown(
                "All CLDR languages sorted by coverage score (lowest first). "
                "T0–T3 = character counts at each tier."
            )
            full_table = gr.Dataframe(interactive=False, wrap=False)
        # ── Language Detail ───────────────────────────────────────────────
        with gr.Tab("🔍 Language Detail"):
            gr.Markdown(
                "Select a language from the **Language** dropdown above. "
                "A quick-view badge appears immediately; full detail is shown here.\n\n"
                "The **Model Comparison** chart below shows how *every* available model "
                "covers the selected language — making it easy to see where coverage gaps "
                "are universal vs. model-specific."
            )
            lang_card_html = gr.HTML()
            with gr.Row():
                lang_tier_pie = gr.Plot(label="Tier Distribution")
                lang_map      = gr.Plot(label="Location on World Map")
            gr.Markdown("### 📊 Coverage Across All Models")
            gr.Markdown(
                "First selection may take a moment to load comparison data for all models."
            )
            lang_comparison_plot = gr.Plot(label="Model Comparison")
    # ── Wire events ───────────────────────────────────────────────────────

    outputs = [
        summary_html,
        grade_bar_plot,
        score_hist_plot,
        tier_bar_plot,
        tier_detail_table,
        world_map_plot,
        macroarea_plot,
        family_plot,
        fert_scatter_plot,
        fert_bar_plot,
        incomplete_table,
        full_table,
        language_dd,
    ]

    lang_outputs = [lang_quickview, lang_card_html, lang_tier_pie, lang_map, lang_comparison_plot]

    run_btn.click(fn=render, inputs=[model_dd], outputs=outputs)
    model_dd.change(fn=render, inputs=[model_dd], outputs=outputs)
    language_dd.change(
        fn=render_language,
        inputs=[model_dd, language_dd],
        outputs=lang_outputs,
    )

    # Auto-load on page open
    if models:
        demo.load(fn=render, inputs=[model_dd], outputs=outputs)


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=".plot-container { border-radius: 8px; } footer { display: none !important; }",
        ssr_mode=False,
    )
