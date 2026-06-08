from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import logger


@dataclass
class RagasResult:

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    n_samples: int = 0
    failed_samples: int = 0
    per_sample: list = field(default_factory=list)

    @property
    def average(self) -> float:
        return round((self.faithfulness + self.answer_relevance) / 2, 3)

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "average": self.average,
            "n_samples": self.n_samples,
            "failed_samples": self.failed_samples,
        }


class RagasEvaluator:

    def __init__(
        self,
        dataset_path: Path | None = None,
        n_samples: int | None = None,
    ) -> None:
        self._dataset_path = dataset_path or (ROOT / "evaluation" / "ground_truth.jsonl")
        self._n_samples = n_samples

    def _load_dataset(self) -> list[dict]:
        all_samples = []
        with open(self._dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    if s.get("category") != "clinical_application":
                        all_samples.append(s)

        excluded = sum(
            1
            for line in open(self._dataset_path)
            if line.strip() and json.loads(line.strip()).get("category") == "clinical_application"
        )
        if excluded:
            logger.info(
                "Excluded {n} clinical-application questions (not in corpus)",
                n=excluded,
            )

        if not self._n_samples:
            logger.info("Loaded {n} evaluation samples", n=len(all_samples))
            return all_samples

        from collections import defaultdict

        by_topic: dict = defaultdict(list)
        for s in all_samples:
            by_topic[s.get("topic", "other")].append(s)

        topics = list(by_topic.keys())
        per_topic = max(1, self._n_samples // len(topics))
        sampled: list = []
        for topic in topics:
            sampled.extend(by_topic[topic][:per_topic])

        remaining = [s for s in all_samples if s not in sampled]
        sampled.extend(remaining[: self._n_samples - len(sampled)])
        sampled = sampled[: self._n_samples]

        logger.info(
            "Loaded {n} samples across {t} topics",
            n=len(sampled),
            t=len(topics),
        )
        return sampled

    async def _get_rag_response(self, question: str) -> tuple[str, str, list[str]]:
        try:
            from src.core.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline()
            result = await pipeline.query(question)
            return (
                result.answer,
                result.retrieval.assembled_context,
                result.citations,
            )
        except Exception as e:
            logger.warning("RAG failed for question: {e}", e=str(e))
            return "", "", []

    async def run(self) -> RagasResult:
        samples = self._load_dataset()

        try:
            return await self._run_ragas_library(samples)
        except ImportError:
            logger.warning("ragas library unavailable — using manual metrics")
            return await self._run_manual_metrics(samples)
        except Exception as e:
            logger.warning("RAGAS failed: {e} — using manual metrics", e=str(e))
            return await self._run_manual_metrics(samples)

    async def _run_ragas_library(self, samples: list[dict]) -> RagasResult:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        print(f"  Running RAGAS on {len(samples)} samples...")

        rows = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for i, s in enumerate(samples):
            print(f"  Getting RAG response {i + 1}/{len(samples)}...", end="\r")
            answer, context, _ = await self._get_rag_response(s["question"])
            rows["question"].append(s["question"])
            rows["answer"].append(answer or "No answer retrieved.")
            rows["contexts"].append([context] if context else ["No context retrieved."])
            rows["ground_truth"].append(s["reference_answer"])

        print()
        dataset = Dataset.from_dict(rows)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
            ),
        )

        df = result.to_pandas()
        return RagasResult(
            faithfulness=float(df["faithfulness"].mean()),
            answer_relevance=float(df["answer_relevancy"].mean()),
            n_samples=len(samples),
            failed_samples=int(df.isnull().any(axis=1).sum()),
            per_sample=df.to_dict(orient="records"),
        )

    async def _run_manual_metrics(self, samples: list[dict]) -> RagasResult:
        import re

        def keywords(text: str) -> set:
            stops = {"the", "a", "an", "is", "are", "to", "of", "in", "and", "or", "it", "its"}
            return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()) if w not in stops}

        faith_scores, rel_scores = [], []
        per_sample = []
        failed = 0

        for i, s in enumerate(samples):
            print(f"  Evaluating sample {i + 1}/{len(samples)}...", end="\r")
            try:
                answer, context, _ = await self._get_rag_response(s["question"])

                if not answer or not context:
                    failed += 1
                    continue

                ans_kw = keywords(answer)
                ctx_kw = keywords(context)
                q_kw = keywords(s["question"])

                faith = len(ans_kw & ctx_kw) / max(len(ans_kw), 1)
                relevance = len(ans_kw & q_kw) / max(len(q_kw), 1)

                faith_scores.append(faith)
                rel_scores.append(relevance)

                per_sample.append(
                    {
                        "question": s["question"][:60],
                        "faithfulness": round(faith, 3),
                        "answer_relevancy": round(relevance, 3),
                    }
                )
            except Exception as e:
                logger.warning("Sample {i} failed: {e}", i=i, e=str(e))
                failed += 1

        print()
        n = len(faith_scores)
        return RagasResult(
            faithfulness=sum(faith_scores) / max(n, 1),
            answer_relevance=sum(rel_scores) / max(n, 1),
            n_samples=len(samples),
            failed_samples=failed,
            per_sample=per_sample,
        )


async def main() -> None:
    print("\n" + "=" * 60)
    print("  socratOT — RAGAS Evaluation")
    print("=" * 60 + "\n")

    evaluator = RagasEvaluator(n_samples=20)
    result = await evaluator.run()

    print("\n" + "=" * 60)
    print("  RAGAS Results")
    print("=" * 60)
    print(f"  Faithfulness:     {result.faithfulness:.3f}")
    print(f"  Answer relevance: {result.answer_relevance:.3f}")
    print(f"  Average:          {result.average:.3f}")
    print(f"  Samples:          {result.n_samples} ({result.failed_samples} failed)")
    print("=" * 60 + "\n")

    out = ROOT / "evaluation" / "results" / "ragas_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"  Saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
