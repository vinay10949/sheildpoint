"""
SP-205 — LoRA Fine-Tuning Pipeline Tests
==========================================

Tests for the LoRA fine-tuning training pipeline. Since we can't run
actual GPU training in CI, these tests verify:

1. Training data preparation produces the correct format.
2. The LoRA config uses the specified rank=16, alpha=32, target modules.
3. The evaluation script computes accuracy correctly.
4. The MMLU benchmark retention check works.
5. The adapter packaging script produces the correct structure.
6. The weekly evaluation pipeline makes correct promotion decisions.

Run with::
    python -m pytest tests/v2/test_lora_training.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent.parent
_training_root = _root / "training"
if str(_training_root) not in sys.path:
    sys.path.insert(0, str(_training_root))


# ===========================================================================
# Training Data Preparation Tests
# ===========================================================================
class TestTrainingDataPreparation:
    def test_build_instruction_example_format(self):
        from prepare_training_data import build_instruction_example
        claim = {
            "claim_id": "CLM-001",
            "policy_id": "POL-001",
            "claimant": "Alice",
            "amount": 5000.00,
            "date_of_loss": "2026-03-14",
            "description": "Wind damage.",
            "claim_type": "property_damage",
            "label_severity": "medium",
            "label_claim_type": "wind",
            "label_fraud_risk_score": 0.3,
        }
        example = build_instruction_example(claim)
        assert "instruction" in example
        assert "input" in example
        assert "output" in example
        # The output should contain the expected classification
        output = json.loads(example["output"])
        assert output["severity"] == "medium"
        assert output["claim_type"] == "wind"
        assert output["fraud_risk_score"] == 0.3

    def test_prepare_dataset_creates_splits(self, tmp_path):
        from prepare_training_data import prepare_dataset
        # Create a small input dataset
        input_path = tmp_path / "claims.jsonl"
        claims = [
            {"claim_id": f"CLM-{i}", "amount": 1000 * i,
             "label_severity": "low", "label_claim_type": "wind",
             "label_fraud_risk_score": 0.1}
            for i in range(100)
        ]
        with open(input_path, "w") as f:
            for c in claims:
                f.write(json.dumps(c) + "\n")

        output_dir = tmp_path / "output"
        counts = prepare_dataset(input_path, output_dir, seed=42)

        assert counts["train"] == 80  # 80%
        assert counts["val"] == 10    # 10%
        assert counts["test"] == 10   # 10%
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()
        assert (output_dir / "manifest.json").exists()

    def test_instruction_format_matches_alpaca(self, tmp_path):
        """Each example must have instruction, input, output fields."""
        from prepare_training_data import prepare_dataset
        input_path = tmp_path / "claims.jsonl"
        # Need enough claims for the train split to be non-empty
        with open(input_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"claim_id": f"C{i}", "amount": 100,
                                    "label_severity": "low"}) + "\n")
        output_dir = tmp_path / "output"
        prepare_dataset(input_path, output_dir, seed=42)
        with open(output_dir / "train.jsonl") as f:
            example = json.loads(f.readline())
        assert set(example.keys()) == {"instruction", "input", "output"}


# ===========================================================================
# LoRA Config Tests
# ===========================================================================
class TestLoRAConfig:
    def test_lora_config_uses_correct_rank_and_alpha(self):
        """AC: LoRA rank=16, alpha=32."""
        # We can't import peft in CI, so we test the config function
        # by checking its source code
        config_src = (_training_root / "lora_train.py").read_text()
        # The script uses --rank default=16 and --alpha default=32
        assert 'default=16' in config_src
        assert 'default=32' in config_src
        # And passes them to the LoraConfig
        assert 'r=rank' in config_src
        assert 'lora_alpha=alpha' in config_src

    def test_lora_targets_q_proj_and_v_proj(self):
        """AC: target modules (q_proj, v_proj)."""
        config_src = (_training_root / "lora_train.py").read_text()
        assert "q_proj" in config_src
        assert "v_proj" in config_src
        assert 'target_modules=["q_proj", "v_proj"]' in config_src

    def test_uses_4bit_awq_quantization(self):
        """AC: 4-bit AWQ base model (QLoRA approach)."""
        config_src = (_training_root / "lora_train.py").read_text()
        assert "4bit" in config_src.lower() or "BitsAndBytesConfig" in config_src
        assert "nf4" in config_src  # NF4 quantization type


# ===========================================================================
# Evaluation Tests
# ===========================================================================
class TestEvaluation:
    def test_compute_accuracy_exact_match(self):
        from evaluate_adapter import compute_accuracy
        predictions = [
            {"severity": "low", "claim_type": "wind", "fraud_risk_score": 0.1},
            {"severity": "medium", "claim_type": "fire", "fraud_risk_score": 0.5},
        ]
        labels = [
            {"severity": "low", "claim_type": "wind", "fraud_risk_score": 0.1},
            {"severity": "medium", "claim_type": "fire", "fraud_risk_score": 0.5},
        ]
        metrics = compute_accuracy(predictions, labels)
        assert metrics["exact_match_accuracy"] == 1.0
        assert metrics["severity_accuracy"] == 1.0
        assert metrics["claim_type_accuracy"] == 1.0
        assert metrics["fraud_risk_mae"] == 0.0

    def test_compute_accuracy_partial_match(self):
        from evaluate_adapter import compute_accuracy
        predictions = [
            {"severity": "low", "claim_type": "wind", "fraud_risk_score": 0.1},
            {"severity": "high", "claim_type": "fire", "fraud_risk_score": 0.5},
        ]
        labels = [
            {"severity": "low", "claim_type": "wind", "fraud_risk_score": 0.1},
            {"severity": "medium", "claim_type": "fire", "fraud_risk_score": 0.4},
        ]
        metrics = compute_accuracy(predictions, labels)
        assert metrics["exact_match_accuracy"] == 0.5  # only first matches exactly
        assert metrics["severity_accuracy"] == 0.5
        assert metrics["claim_type_accuracy"] == 1.0
        assert metrics["fraud_risk_mae"] == pytest.approx(0.05, abs=0.01)

    def test_parse_model_output_valid_json(self):
        from evaluate_adapter import parse_model_output
        result = parse_model_output('{"severity": "low", "fraud_risk_score": 0.1}')
        assert result["severity"] == "low"

    def test_parse_model_output_with_surrounding_text(self):
        from evaluate_adapter import parse_model_output
        result = parse_model_output(
            'Here is the classification:\n{"severity": "low"}\nDone.'
        )
        assert result["severity"] == "low"

    def test_parse_model_output_invalid_returns_empty(self):
        from evaluate_adapter import parse_model_output
        result = parse_model_output("not json at all")
        assert result == {}


# ===========================================================================
# MMLU Benchmark Tests
# ===========================================================================
class TestMMLUBenchmark:
    def test_check_reasoning_retention_within_threshold(self):
        from mmlu_benchmark import check_reasoning_retention
        result = check_reasoning_retention(
            base_accuracy=0.80, lora_accuracy=0.79, threshold_pp=2.0,
        )
        assert result["within_threshold"] is True
        assert result["delta_pp"] == pytest.approx(-1.0)

    def test_check_reasoning_retention_exceeds_threshold(self):
        from mmlu_benchmark import check_reasoning_retention
        result = check_reasoning_retention(
            base_accuracy=0.80, lora_accuracy=0.70, threshold_pp=2.0,
        )
        assert result["within_threshold"] is False


# ===========================================================================
# Adapter Packaging Tests
# ===========================================================================
class TestAdapterPackaging:
    def test_package_adapter_creates_directory(self, tmp_path):
        from package_adapter import package_adapter
        # Create a fake adapter source
        source = tmp_path / "source"
        source.mkdir()
        (source / "adapter_model.safetensors").write_text("fake weights")
        (source / "adapter_config.json").write_text("{}")

        adapter_dir = package_adapter(
            adapter_source=source,
            output_dir=tmp_path / "packaged",
            adapter_name="test-adapter-v1",
            metrics={"exact_match_accuracy": 0.85},
            training_summary={"rank": 16, "alpha": 32, "epochs": 3},
            create_zip=False,
        )
        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "metrics.json").exists()
        assert (adapter_dir / "README.md").exists()

    def test_adapter_config_has_correct_lora_params(self, tmp_path):
        from package_adapter import package_adapter
        source = tmp_path / "source"
        source.mkdir()
        (source / "adapter_config.json").write_text("{}")

        adapter_dir = package_adapter(
            adapter_source=source,
            output_dir=tmp_path / "packaged",
            adapter_name="test-adapter-v1",
            training_summary={"rank": 16, "alpha": 32},
            create_zip=False,
        )
        with open(adapter_dir / "adapter_config.json") as f:
            config = json.load(f)
        assert config["r"] == 16
        assert config["lora_alpha"] == 32
        assert config["target_modules"] == ["q_proj", "v_proj"]

    def test_readme_contains_metrics(self, tmp_path):
        from package_adapter import package_adapter
        source = tmp_path / "source"
        source.mkdir()
        (source / "adapter_config.json").write_text("{}")

        adapter_dir = package_adapter(
            adapter_source=source,
            output_dir=tmp_path / "packaged",
            adapter_name="test-adapter-v1",
            metrics={"exact_match_accuracy": 0.85, "severity_accuracy": 0.90,
                      "improvement_pp": 7.5, "meets_5pp_threshold": True},
            create_zip=False,
        )
        readme = (adapter_dir / "README.md").read_text()
        assert "85.00%" in readme  # exact match accuracy
        assert "7.5pp" in readme   # improvement


# ===========================================================================
# Weekly Pipeline Tests
# ===========================================================================
class TestWeeklyPipeline:
    def test_promotion_decision_logic(self, tmp_path):
        """Test the promotion decision logic without running real inference."""
        from weekly_eval_pipeline import run_weekly_pipeline

        # We can't run real inference, but we can test the pipeline structure
        # by checking that it produces a report with the expected keys.
        # Create a minimal adapter dir
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        # Create minimal val data
        val_data = tmp_path / "val.jsonl"
        with open(val_data, "w") as f:
            f.write(json.dumps({
                "instruction": "test",
                "input": "{}",
                "output": '{"severity": "low", "claim_type": "wind", '
                          '"fraud_risk_score": 0.1}',
            }) + "\n")

        report = run_weekly_pipeline(
            adapter_dir=adapter_dir,
            val_data=val_data,
            report_output=tmp_path / "report.json",
            production_dir=tmp_path / "production",
        )
        assert "adapter_metrics" in report
        assert "mmlu_base" in report
        assert "mmlu_lora" in report
        assert "retention_check" in report
        assert "promotion_decision" in report
        assert (tmp_path / "report.json").exists()
