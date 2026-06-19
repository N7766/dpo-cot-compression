"""Utilities for preparing GSM8K examples."""

from __future__ import annotations

from datasets import load_dataset

from src.evaluation.answer_extraction import extract_gold_answer


def load_gsm8k_split(
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "test",
):
    """Load a GSM8K split from Hugging Face datasets."""
    return load_dataset(dataset_name, dataset_config, split=split)


def process_gsm8k_examples(dataset, split: str = "test", max_samples: int | None = None) -> list[dict]:
    """Convert raw GSM8K rows into the project JSONL format."""
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    processed = []
    for idx, row in enumerate(dataset):
        answer = row["answer"]
        processed.append(
            {
                "id": f"{split}_{idx}",
                "question": row["question"],
                "answer": answer,
                "gold_answer": extract_gold_answer(answer),
            }
        )
    return processed
