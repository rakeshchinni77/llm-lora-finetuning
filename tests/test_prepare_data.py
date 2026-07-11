"""Unit tests for scripts.prepare_data."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import prepare_data


def test_format_instruction_with_context() -> None:
	"""Format output includes context section when context is provided."""
	sample = {
		"instruction": "Summarize this",
		"context": "A short paragraph.",
		"response": "A short summary.",
	}

	formatted = prepare_data.format_instruction(sample)

	assert "### Instruction:" in formatted
	assert "### Context:" in formatted
	assert "### Response:" in formatted
	assert "A short paragraph." in formatted


def test_format_instruction_without_context() -> None:
	"""Format output excludes context section when context is empty."""
	sample = {
		"instruction": "Explain LoRA",
		"context": "   ",
		"response": "LoRA is parameter-efficient fine-tuning.",
	}

	formatted = prepare_data.format_instruction(sample)

	assert "### Instruction:" in formatted
	assert "### Context:" not in formatted
	assert "### Response:" in formatted


def test_filter_samples_rejects_invalid_records() -> None:
	"""Samples with missing instruction/response should be filtered out."""
	dataset = [
		{"instruction": "Valid", "response": "Valid response", "context": "ctx"},
		{"instruction": "", "response": "Has response", "context": "ctx"},
		{"instruction": "Has instruction", "response": "   ", "context": "ctx"},
		{"instruction": None, "response": "x", "context": "ctx"},
		{"instruction": "x", "response": None, "context": "ctx"},
	]

	records = prepare_data.filter_samples(dataset)

	assert len(records) == 1
	assert isinstance(records[0], dict)
	assert set(records[0].keys()) == {"text"}
	assert "### Instruction:" in records[0]["text"]


def test_filter_samples_rejects_whitespace_instruction() -> None:
	"""Whitespace-only instruction should be treated as invalid."""
	dataset = [{"instruction": "   ", "response": "ok", "context": "x"}]

	records = prepare_data.filter_samples(dataset)

	assert records == []


def test_filter_samples_rejects_oversized_instruction_plus_response() -> None:
	"""Records above max character length are excluded."""
	oversized_instruction = "i" * (prepare_data.MAX_CHAR_LENGTH // 2 + 10)
	oversized_response = "r" * (prepare_data.MAX_CHAR_LENGTH // 2 + 10)
	dataset = [
		{
			"instruction": oversized_instruction,
			"response": oversized_response,
			"context": "",
		}
	]

	records = prepare_data.filter_samples(dataset)

	assert records == []


def test_filter_samples_handles_malformed_non_string_fields() -> None:
	"""Non-string fields should be normalized and formatted safely."""
	dataset = [{"instruction": 123, "response": 456, "context": None}]

	records = prepare_data.filter_samples(dataset)

	assert len(records) == 1
	assert "123" in records[0]["text"]
	assert "456" in records[0]["text"]


def test_write_json_records_writes_expected_schema(tmp_path: Path) -> None:
	"""JSON writer should output list[dict[text=str]] structure."""
	output_file = tmp_path / "records.json"
	records = [{"text": "sample one"}, {"text": "sample two"}]

	prepare_data.write_json_records(output_file, records)

	written = json.loads(output_file.read_text(encoding="utf-8"))
	assert isinstance(written, list)
	assert written == records
	assert all(isinstance(item.get("text"), str) for item in written)


def test_percentile_is_deterministic_for_known_input() -> None:
	"""Percentile helper should return stable deterministic values."""
	values = [1, 2, 3, 4, 5]

	p50 = prepare_data._percentile(values, 50)
	p95 = prepare_data._percentile(values, 95)

	assert p50 == 3.0
	assert p95 == 4.8
