from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go


def _load_results() -> tuple[dict, list[dict]]:
    results_dir = _HERE / "results"
    full = results_dir / "full_evaluation.json"
    ragas_only = results_dir / "ragas_scores.json"
    baseline_only = results_dir / "baseline_comparison.json"

    ragas: dict = {}
    baselines: list = []

    if full.exists():
        data = json.loads(full.read_text())
        ragas = data.get("ragas", {})
        baselines = data.get("baselines", [])
    else:
        if ragas_only.exists():
            ragas = json.loads(ragas_only.read_text())
        if baseline_only.exists():
            baselines = json.loads(baseline_only.read_text())

    return ragas, baselines


def ragas_bar_chart() -> go.Figure:
    ragas, _ = _load_results()

    metrics = ["faithfulness", "answer_relevance"]
    labels = ["Faithfulness", "Answer Relevance"]
    values = [ragas.get(m, 0.0) for m in metrics]
    colors = ["#6366F1", "#22D3EE"]

    fig = go.Figure()
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        fig.add_trace(
            go.Bar(
                name=label,
                x=[label],
                y=[val],
                marker_color=color,
                text=[f"{val:.3f}"],
                textposition="outside",
                width=0.4,
            )
        )

    fig.update_layout(
        title=dict(text="RAGAS Evaluation Metrics", font=dict(size=16, color="#1e293b")),
        yaxis=dict(range=[0, 1.1], title="Score", tickformat=".2f", gridcolor="#e2e8f0"),
        xaxis=dict(title=""),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(t=60, b=40, l=40, r=40),
        font=dict(family="sans-serif", color="#334155"),
    )
    fig.add_hline(y=0.7, line_dash="dot", line_color="#94a3b8", annotation_text="0.70 target")
    return fig


def baseline_comparison_chart() -> go.Figure:
    _, baselines = _load_results()

    if not baselines:
        return go.Figure()

    system_labels = {"socratOT": "socratOT", "no_rag": "Direct LLM", "no_socratic": "RAG only"}
    metrics = ["rouge_l", "bert_score_f1", "keyword_overlap"]
    metric_labels = ["ROUGE-L", "BERTScore F1", "Keyword Overlap"]
    colors = ["#6366F1", "#e2e8f0", "#cbd5e1"]

    system_names = [system_labels.get(b["system"], b["system"]) for b in baselines]

    fig = go.Figure()
    for j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        vals = [b.get(metric, 0.0) for b in baselines]
        fig.add_trace(
            go.Bar(
                name=mlabel,
                x=system_names,
                y=vals,
                marker_color=colors[j],
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
            )
        )

    fig.update_layout(
        title=dict(text="Benchmark: socratOT vs Baselines", font=dict(size=16, color="#1e293b")),
        yaxis=dict(range=[0, 0.75], title="Score", tickformat=".2f", gridcolor="#e2e8f0"),
        xaxis=dict(title="System"),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=40),
        font=dict(family="sans-serif", color="#334155"),
    )
    return fig


def ragas_radar_chart() -> go.Figure:
    ragas, _ = _load_results()

    metrics = ["faithfulness", "answer_relevance"]
    labels = ["Faithfulness", "Answer Relevance"]
    values = [ragas.get(m, 0.0) for m in metrics]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(99,102,241,0.2)",
            line=dict(color="#6366F1", width=2),
            name="socratOT",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".1f", gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0"),
            bgcolor="white",
        ),
        showlegend=False,
        title=dict(text="RAGAS Profile", font=dict(size=16, color="#1e293b")),
        paper_bgcolor="white",
        height=380,
        margin=dict(t=60, b=40, l=60, r=60),
        font=dict(family="sans-serif", color="#334155"),
    )
    return fig


def save_charts(output_dir: Path | None = None) -> None:
    out = output_dir or (_HERE / "results" / "charts")
    out.mkdir(parents=True, exist_ok=True)

    ragas_bar_chart().write_image(str(out / "ragas_metrics.png"), scale=2)
    baseline_comparison_chart().write_image(str(out / "baseline_comparison.png"), scale=2)
    ragas_radar_chart().write_image(str(out / "ragas_radar.png"), scale=2)
    print(f"Charts saved to {out}/")


if __name__ == "__main__":
    save_charts()
