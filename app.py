"""
LLM Vocabulary Coverage Dashboard — Gradio app for Hugging Face Spaces.

Loads pre-computed coverage JSON files from data/coverage/ and renders
interactive Plotly charts plus a searchable language table.

Pre-compute coverage files with:
  python scripts/ingest_model.py --model gpt-4o --source tiktoken --fertility
"""
from __future__ import annotations

import json
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
            "tokens_per_char": fert.get("tokens_per_char"),
            "tokens_per_word": fert.get("tokens_per_word"),
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
    keep the chart readable.
    """
    worst = df.nsmallest(60, "score").copy()
    worst = worst.sort_values("score", ascending=True)

    totals = worst[["tier0", "tier1", "tier2", "tier3"]].sum(axis=1).clip(lower=1)
    labels = worst["name"] + " (" + worst["locale"] + ")"

    fig = go.Figure()
    for tier, color, label in zip(
        ["tier0", "tier1", "tier2", "tier3"],
        TIER_PALETTE,
        ["Tier 0 — native token", "Tier 1 — embedded", "Tier 2 — byte fallback", "Tier 3 — unreachable"],
    ):
        pct = worst[tier] / totals * 100
        fig.add_trace(go.Bar(
            name=label,
            x=labels,
            y=pct,
            marker_color=color,
            hovertemplate="%{x}<br>" + label + ": %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title="Character-Tier Breakdown — 60 Lowest-Scoring Languages",
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="% of Characters", range=[0, 100]),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=520,
    )
    return fig


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
    fdf = df.dropna(subset=["tokens_per_char"]).copy()
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
    fdf = df.dropna(subset=["tokens_per_char"]).nlargest(30, "tokens_per_char")
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
# Main render function wired to every control
# ---------------------------------------------------------------------------

def render(model_name: str):
    if not model_name:
        empty_fig = go.Figure()
        empty_df  = pd.DataFrame()
        return ("", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_df, empty_df)

    data = load_coverage(model_name)
    df   = build_dataframe(data)

    if df.empty:
        msg = f"<p>No CLDR language data found in coverage file for <b>{model_name}</b>.</p>"
        empty_fig = go.Figure()
        empty_df  = pd.DataFrame()
        return (msg, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_df, empty_df)

    summary_html   = make_summary_html(data, df)
    grade_bar      = make_distribution_chart(df)
    score_hist     = make_score_histogram(df)
    tier_bar       = make_tier_stacked_bar(df)
    world_map      = make_world_map(df)
    macroarea_bar  = make_macroarea_chart(df)
    family_bar     = make_family_chart(df)
    fert_scatter   = make_fertility_scatter(df)
    fert_bar       = make_fertility_bar(df)
    incomplete_tbl = make_incomplete_table(df)
    full_tbl       = make_full_table(df)

    return (
        summary_html,
        grade_bar,
        score_hist,
        tier_bar,
        world_map,
        macroarea_bar,
        family_bar,
        fert_scatter,
        fert_bar,
        incomplete_tbl,
        full_tbl,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

models = list_models()

_OUTPUTS_COUNT = 11  # must match number of return values in render()

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
            value=models[0] if models else None,
            label="Model",
            scale=3,
        )
        run_btn = gr.Button("Generate Report", variant="primary", scale=1)

    # ---- Summary ----
    summary_html = gr.HTML()

    with gr.Tabs():

        # ── Overview ──────────────────────────────────────────────────────
        with gr.Tab("📊 Overview"):
            with gr.Row():
                grade_bar_plot = gr.Plot(label="Grades")
                score_hist_plot = gr.Plot(label="Score Distribution")
            tier_bar_plot = gr.Plot(label="Tier Breakdown (60 lowest-scoring)")

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

    # ── Wire events ───────────────────────────────────────────────────────

    outputs = [
        summary_html,
        grade_bar_plot,
        score_hist_plot,
        tier_bar_plot,
        world_map_plot,
        macroarea_plot,
        family_plot,
        fert_scatter_plot,
        fert_bar_plot,
        incomplete_table,
        full_table,
    ]

    run_btn.click(fn=render, inputs=[model_dd], outputs=outputs)
    model_dd.change(fn=render, inputs=[model_dd], outputs=outputs)

    # Auto-load on page open
    if models:
        demo.load(fn=render, inputs=[model_dd], outputs=outputs)


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=".plot-container { border-radius: 8px; } footer { display: none !important; }",
        ssr_mode=False,
    )
