#!/usr/bin/env python3
"""
SP-205 — LoRA Fine-Tuning Script for Qwen3.6 35B A3B
======================================================

Fine-tunes the Qwen3.6 35B A3B model on ShieldPoint's historical claims
data using QLoRA (Quantized Low-Rank Adaptation):

- Base model: Qwen3.6-35B-A3B (4-bit AWQ quantized)
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, v_proj (query and value projections)
- Training data: 3,500 historical claims (from prepare_training_data.py)
- Optimizer: AdamW with cosine learning rate schedule
- Batch size: 4 (with gradient accumulation = 4, effective batch = 16)
- Epochs: 3
- Max sequence length: 2048

The QLoRA approach leaves the base model weights untouched — only the
LoRA adapter weights are updated. This:

1. Preserves the model's general reasoning capabilities (no catastrophic forgetting).
2. Reduces GPU memory from ~140GB (full fine-tune) to ~24GB (QLoRA).
3. Allows the adapter to be iteratively refined weekly without retraining the base.

Usage
-----
    python lora_train.py \\
        --base-model Qwen/Qwen3.6-35B-A3B \\
        --train-data training/data/train.jsonl \\
        --val-data training/data/val.jsonl \\
        --output-dir training/adapters/v1 \\
        --rank 16 --alpha 32

Dependencies
------------
    pip install torch transformers peft trl bitsandbytes accelerate datasets

The script auto-detects GPU availability and falls back to CPU (very slow)
if no GPU is present. For production training, use an A100 80GB GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shieldpoint.training.lora")


def check_dependencies() -> bool:
    """Check that the required ML dependencies are available."""
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(
            "Missing dependencies: %s. Install with: pip install %s",
            missing, " ".join(missing),
        )
        return False
    return True


def load_dataset(path: Path) -> list[dict[str, str]]:
    """Load an instruction-format JSONL dataset."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def setup_model_and_tokenizer(
    base_model: str,
    *,
    load_in_4bit: bool = True,
):
    """Load the base model and tokenizer with 4-bit quantization."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.config.use_cache = False  # Required for gradient checkpointing
    return model, tokenizer


def setup_lora_config(rank: int, alpha: int, dropout: float = 0.05):
    """Configure the LoRA adapter."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],  # Query and value projections
        bias="none",
        modules_to_save=None,
    )


def train(
    *,
    base_model: str,
    train_data_path: Path,
    val_data_path: Path,
    output_dir: Path,
    rank: int = 16,
    alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    warmup_ratio: float = 0.03,
    use_4bit: bool = True,
) -> dict[str, Any]:
    """Run the LoRA fine-tuning training loop.

    Returns a summary dict with training metrics.
    """
    if not check_dependencies():
        return {"error": "Missing dependencies", "success": False}

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading training data from %s", train_data_path)
    train_data = load_dataset(train_data_path)
    val_data = load_dataset(val_data_path)
    logger.info("Train: %d examples, Val: %d examples", len(train_data), len(val_data))

    # Setup model + tokenizer
    logger.info("Loading base model: %s (4-bit: %s)", base_model, use_4bit)
    model, tokenizer = setup_model_and_tokenizer(base_model, load_in_4bit=use_4bit)

    # Setup LoRA
    lora_config = setup_lora_config(rank=rank, alpha=alpha)
    logger.info(
        "LoRA config: rank=%d, alpha=%d, target_modules=%s",
        rank, alpha, lora_config.target_modules,
    )

    # Prepare dataset in the format expected by TRL's SFTTrainer
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    def format_example(example: dict) -> str:
        """Format an instruction example into a single text string."""
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Output:\n{example['output']}"
        )

    train_dataset = train_dataset.map(
        lambda x: {"text": format_example(x)}, remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: {"text": format_example(x)}, remove_columns=val_dataset.column_names
    )

    # Setup trainer
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Set to "wandb" for experiment tracking
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save the adapter
    adapter_path = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Saved LoRA adapter to %s", adapter_path)

    # Evaluate
    eval_result = trainer.evaluate()
    logger.info("Final eval loss: %.4f", eval_result.get("eval_loss", 0))

    # Save training summary
    summary = {
        "success": True,
        "base_model": base_model,
        "rank": rank,
        "alpha": alpha,
        "target_modules": ["q_proj", "v_proj"],
        "epochs": epochs,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "adapter_path": str(adapter_path),
        "lora_config": {
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to %s", summary_path)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Qwen3.6 35B A3B on claims data"
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--train-data", type=Path,
        default=Path("/home/z/my-project/sheildpoint/training/data/train.jsonl"),
    )
    parser.add_argument(
        "--val-data", type=Path,
        default=Path("/home/z/my-project/sheildpoint/training/data/val.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/z/my-project/sheildpoint/training/adapters/v1"),
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    summary = train(
        base_model=args.base_model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        use_4bit=not args.no_4bit,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
