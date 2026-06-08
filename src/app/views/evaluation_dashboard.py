from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation" / "results"


def _load_full() -> dict:
    path = RESULTS_DIR / "full_evaluation.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_ragas() -> dict:
    path = RESULTS_DIR / "ragas_scores.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_baselines() -> list[dict]:
    path = RESULTS_DIR / "baseline_comparison.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    return []


def render() -> None:
    st.header("Evaluation Dashboard")
    st.caption("RAGAS quality metrics and benchmark comparison results")
    st.divider()

    full = _load_full()
    ragas = full.get("ragas") if full else _load_ragas()
    baselines = full.get("baselines") if full else _load_baselines()

    if not ragas and not baselines:
        st.warning("No evaluation results found. Run `python -m src.evaluation.run_evaluation` to generate results.")
        return

    if ragas:
        st.subheader("RAGAS Evaluation")
        c1, c2, c3 = st.columns(3)
        c1.metric("Faithfulness", f"{ragas.get('faithfulness', 0):.3f}")
        c2.metric("Answer Relevance", f"{ragas.get('answer_relevance', 0):.3f}")
        c3.metric("Average", f"{ragas.get('average', 0):.3f}")

        try:
            from src.evaluation.plot_results import ragas_bar_chart, ragas_radar_chart
            col_bar, col_radar = st.columns(2)
            with col_bar:
                st.plotly_chart(ragas_bar_chart(), use_container_width=True)
            with col_radar:
                st.plotly_chart(ragas_radar_chart(), use_container_width=True)
        except Exception as e:
            st.caption(f"Chart rendering unavailable: {e}")

    if baselines:
        st.divider()
        st.subheader("Benchmark Comparison")
        st.caption(
            "socratOT is a Socratic tutor — it guides rather than directly answers, "
            "so lower ROUGE/BERTScore reflects teaching behavior, not lower quality."
        )

        try:
            from src.evaluation.plot_results import baseline_comparison_chart
            st.plotly_chart(baseline_comparison_chart(), use_container_width=True)
        except Exception as e:
            st.caption(f"Chart rendering unavailable: {e}")

        st.write("")
        system_labels = {"socratOT": "socratOT", "no_rag": "Direct LLM", "no_socratic": "RAG only"}
        rows = [
            {
                "System": system_labels.get(b["system"], b["system"]),
                "ROUGE-L": f"{b.get('rouge_l', 0):.3f}",
                "BERTScore F1": f"{b.get('bert_score_f1', 0):.3f}",
                "Keyword Overlap": f"{b.get('keyword_overlap', 0):.3f}",
                "Samples": b.get("n_samples", 0),
            }
            for b in baselines
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if full.get("metadata"):
        st.divider()
        with st.expander("Evaluation metadata"):
            st.json(full["metadata"])
