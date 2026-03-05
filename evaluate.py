"""
RAG-Aya :: Evaluation

RAGAS evaluation for the RAG pipeline.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EvalSample:
    question: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str] = None


def evaluate_ragas(samples: List[EvalSample]) -> dict:
    """Evaluate RAG pipeline using RAGAS metrics."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    data = {
        "question": [s.question for s in samples],
        "contexts": [s.contexts for s in samples],
        "answer": [s.answer for s in samples],
    }
    # Add ground_truth if available (needed for context_recall)
    has_gt = all(s.ground_truth is not None for s in samples)
    if has_gt:
        data["ground_truth"] = [s.ground_truth for s in samples]

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_precision]
    results = evaluate(dataset=dataset, metrics=metrics)

    return dict(results)


def evaluate_simple(samples: List[EvalSample]) -> dict:
    """Simple evaluation without RAGAS (fallback)."""
    stats = {
        "total": len(samples),
        "has_answer": sum(1 for s in samples if s.answer.strip()),
        "avg_answer_len": 0,
        "avg_context_count": 0,
    }
    if samples:
        stats["avg_answer_len"] = sum(len(s.answer) for s in samples) / len(samples)
        stats["avg_context_count"] = sum(len(s.contexts) for s in samples) / len(samples)
    return stats
