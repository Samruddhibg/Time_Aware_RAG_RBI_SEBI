"""
training/finetune.py — Fine-tune the embedding model with MultipleNegativesRankingLoss.

Usage:
    python -m training.finetune \
        --data data/training/pairs_with_negatives.jsonl \
        --base-model BAAI/bge-small-en-v1.5 \
        --output models/bge-small-ft \
        --epochs 3 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from loguru import logger
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


def load_dataset_from_jsonl(path: Path) -> Dataset:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    anchors = [r["anchor"] for r in records]
    positives = [r["positive"] for r in records]
    negatives = [r.get("negative") for r in records]

    data: dict = {"anchor": anchors, "positive": positives}
    if any(n is not None for n in negatives):
        # Filter out any None negatives
        valid = [(a, p, n) for a, p, n in zip(anchors, positives, negatives) if n]
        if valid:
            data["anchor"] = [v[0] for v in valid]
            data["positive"] = [v[1] for v in valid]
            data["negative"] = [v[2] for v in valid]

    return Dataset.from_dict(data)


def finetune(
    data_path: Path,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir.parent / "runs"

    logger.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)

    logger.info(f"Loading training data: {data_path}")
    train_ds = load_dataset_from_jsonl(data_path)
    logger.info(f"Training samples: {len(train_ds)}")

    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(runs_dir / "ft1"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=True,  # set to False if no GPU
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=20,
        save_steps=500,
        eval_strategy="no",
        seed=seed,
        report_to="none",  # change to "wandb" for tracking
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(str(output_dir))
    logger.success(f"Model saved to {output_dir}")
    logger.info(
        "Next step: update EMBED_MODEL in .env to point to this model, "
        "then re-run build_index.py with --rebuild to re-embed all chunks."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL with (anchor, positive[, negative])")
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output", default="models/bge-small-ft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    finetune(
        Path(args.data),
        args.base_model,
        Path(args.output),
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
    )
