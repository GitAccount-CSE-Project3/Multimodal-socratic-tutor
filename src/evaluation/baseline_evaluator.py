from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation._metrics import bert_score_approx, keyword_overlap, rouge_l
from src.utils.logger import logger


@dataclass
class BaselineResult:

    system_name: str
    rouge_l: float = 0.0
    bert_score_f1: float = 0.0
    keyword_overlap: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "rouge_l": round(self.rouge_l, 3),
            "bert_score_f1": round(self.bert_score_f1, 3),
            "keyword_overlap": round(self.keyword_overlap, 3),
            "n_samples": self.n_samples,
        }


class BaselineEvaluator:

    def __init__(self, n_samples: int = 10) -> None:
        self._n_samples = n_samples
        self._dataset = ROOT / "evaluation" / "ground_truth.jsonl"

    def _load_samples(self) -> list[dict]:
        samples = []
        with open(self._dataset) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))
        return [s for s in samples if s.get("category", "anatomy") == "anatomy"][: self._n_samples]

    async def _get_socratot_answer(self, question: str) -> str:
        try:
            from src.core.rag.pipeline import RAGPipeline
            result = await RAGPipeline().query(question)
            return result.answer
        except Exception as e:
            logger.warning("socratOT query failed: {e}", e=str(e))
            return ""

    async def _get_no_rag_answer(self, question: str) -> str:
        try:
            from src.models.llm_factory import get_llm
            llm = get_llm()
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: llm.invoke(f"Answer this anatomy question directly: {question}"))
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.warning("No-RAG query failed: {e}", e=str(e))
            return ""

    async def _get_no_socratic_answer(self, question: str) -> str:
        try:
            from src.core.rag.pipeline import RAGPipeline
            result = await RAGPipeline(system_prompt="You are a helpful anatomy tutor. Answer questions directly.").query(question)
            return result.answer
        except Exception as e:
            logger.warning("No-Socratic query failed: {e}", e=str(e))
            return ""

    async def run(self) -> list[BaselineResult]:
        samples = self._load_samples()
        print(f"  Running benchmark on {len(samples)} samples...")
        systems = {"socratOT": self._get_socratot_answer, "no_rag": self._get_no_rag_answer, "no_socratic": self._get_no_socratic_answer}
        scores: dict[str, dict] = {name: {"rouge": [], "bert": [], "overlap": []} for name in systems}

        for i, sample in enumerate(samples):
            q, ref = sample["question"], sample["reference_answer"]
            print(f"  Sample {i + 1}/{len(samples)}: {q[:50]}...", end="\r")
            for name, fn in systems.items():
                try:
                    answer = await fn(q)
                    if answer:
                        scores[name]["rouge"].append(rouge_l(answer, ref))
                        scores[name]["bert"].append(bert_score_approx(answer, ref))
                        scores[name]["overlap"].append(keyword_overlap(answer, ref))
                except Exception as e:
                    logger.warning("System {n} failed: {e}", n=name, e=str(e))

        print()
        return [
            BaselineResult(
                system_name=name,
                rouge_l=sum(s["rouge"]) / max(len(s["rouge"]), 1),
                bert_score_f1=sum(s["bert"]) / max(len(s["bert"]), 1),
                keyword_overlap=sum(s["overlap"]) / max(len(s["overlap"]), 1),
                n_samples=len(s["rouge"]),
            )
            for name, s in scores.items()
        ]


async def main() -> None:
    print("\n" + "=" * 60)
    print("  socratOT — Benchmark Comparison")
    print("=" * 60 + "\n")
    evaluator = BaselineEvaluator(n_samples=10)
    results = await evaluator.run()
    print(f"\n  {'System':<15} {'ROUGE-L':>8} {'BERTScore':>10} {'Overlap':>8}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 10} {'-' * 8}")
    for r in results:
        print(f"  {r.system_name:<15} {r.rouge_l:>8.3f} {r.bert_score_f1:>10.3f} {r.keyword_overlap:>8.3f}")
    out = ROOT / "evaluation" / "results" / "baseline_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
