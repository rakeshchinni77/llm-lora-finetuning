"""Prompt formatting utilities for instruction fine-tuned inference.

This module is intentionally limited to string template construction.
"""

from __future__ import annotations


def _normalize_text(value: str, field_name: str) -> str:
	"""Validate and normalize user-provided text fields.

	Args:
		value: Input text.
		field_name: Name of the field for error messages.

	Returns:
		Stripped text.

	Raises:
		TypeError: If value is not a string.
		ValueError: If the stripped text is empty.
	"""
	if not isinstance(value, str):
		raise TypeError(f"{field_name} must be a string")

	normalized = value.strip()
	if not normalized:
		raise ValueError(f"{field_name} must not be empty")

	return normalized


def build_prompt(instruction: str, context: str | None = None) -> str:
	"""Build a training-template-compatible instruction prompt.

	Format:
		### Instruction:
		{instruction}

		### Context:
		{context}

		### Response:

	The context section is included only when context is provided and non-empty.

	Args:
		instruction: Instruction text for the model.
		context: Optional supporting context.

	Returns:
		Formatted prompt string.
	"""
	normalized_instruction = _normalize_text(instruction, "instruction")

	prompt = f"### Instruction:\n{normalized_instruction}\n\n"
	if context is not None and context.strip():
		normalized_context = _normalize_text(context, "context")
		prompt += f"### Context:\n{normalized_context}\n\n"

	prompt += "### Response:\n"
	return prompt
