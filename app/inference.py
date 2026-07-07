"""Inference engine for the fine-tuned LoRA model.

This module is intentionally focused on inference concerns only.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from app.loader import load_model_and_tokenizer
from app.prompt_template import build_prompt
from scripts.utils import get_logger, load_json_config

LOGGER = get_logger("app.inference")
GENERATION_CONFIG_PATH = ROOT_DIR / "config" / "generation_config.json"


def _load_generation_config() -> dict[str, Any]:
	"""Load generation settings from config file."""
	config = load_json_config(GENERATION_CONFIG_PATH)
	LOGGER.info("Generation config loaded")
	return config


def _prepare_generation_kwargs(
	generation_config: dict[str, Any],
	tokenizer: Any,
	max_new_tokens: int,
) -> dict[str, Any]:
	"""Build generation kwargs using config defaults with runtime override."""
	if max_new_tokens <= 0:
		raise ValueError("max_new_tokens must be greater than 0")

	kwargs = dict(generation_config)
	kwargs["max_new_tokens"] = max_new_tokens
	kwargs["pad_token_id"] = tokenizer.pad_token_id
	return kwargs


@lru_cache(maxsize=1)
def _get_default_engine() -> "InferenceEngine":
	"""Return a cached inference engine instance for one-shot API calls."""
	return InferenceEngine()


class InferenceEngine:
	"""Encapsulates model loading and text generation."""

	def __init__(self) -> None:
		"""Initialize engine by loading model, tokenizer, and generation config."""
		self.model, self.tokenizer = load_model_and_tokenizer()
		self.generation_config = _load_generation_config()

	def generate_response(
		self,
		instruction: str,
		context: str | None = None,
		max_new_tokens: int = 256,
	) -> str:
		"""Generate a response for an instruction/context pair.

		Args:
			instruction: Instruction text.
			context: Optional context text.
			max_new_tokens: Runtime cap for generated token count.

		Returns:
			Generated response text (prompt removed).
		"""
		import torch

		prompt = build_prompt(instruction=instruction, context=context)
		inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
		input_device = self.model.get_input_embeddings().weight.device
		inputs = {key: value.to(input_device) for key, value in inputs.items()}

		generation_kwargs = _prepare_generation_kwargs(
			generation_config=self.generation_config,
			tokenizer=self.tokenizer,
			max_new_tokens=max_new_tokens,
		)

		with torch.no_grad():
			output_ids = self.model.generate(**inputs, **generation_kwargs)

		decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
		if decoded.startswith(prompt):
			return decoded[len(prompt) :].strip()

		# Fallback if whitespace/tokenization differences alter exact prefix matching
		return decoded.replace(prompt, "", 1).strip()


def generate_response(
	instruction: str,
	context: str | None = None,
	max_new_tokens: int = 256,
) -> str:
	"""Convenience function for one-shot inference calls."""
	engine = _get_default_engine()
	return engine.generate_response(
		instruction=instruction,
		context=context,
		max_new_tokens=max_new_tokens,
	)


def _build_arg_parser() -> argparse.ArgumentParser:
	"""Create CLI argument parser.

	Important:
		This function performs no model loading so `--help` stays fast.
	"""
	parser = argparse.ArgumentParser(
		description="Run the local LLM inference demo.",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=256,
		help="Maximum number of new tokens to generate per response.",
	)
	return parser


def _run_interactive_demo(max_new_tokens: int) -> None:
	"""Run a console-based interactive inference demo."""
	print("=========================")
	print("LLM Inference Demo")
	print("=========================")
	print("Loading model and tokenizer. This may take some time...")
	engine = InferenceEngine()
	print("Inference engine ready. Type 'exit' or 'quit' to stop.\n")

	while True:
		instruction = input("Instruction: ").strip()
		if instruction.lower() in {"exit", "quit"}:
			print("Exiting interactive demo.")
			break
		if not instruction:
			print("Please enter a non-empty instruction.\n")
			continue

		context = input("Context (optional): ").strip()
		context_value = context if context else None

		try:
			answer = engine.generate_response(
				instruction=instruction,
				context=context_value,
				max_new_tokens=max_new_tokens,
			)
			print(f"Answer:\n{answer}\n")
		except Exception as exc:  # pragma: no cover - runtime diagnostics
			LOGGER.exception("Generation failed: %s", exc)
			print(f"Error: {exc}\n")


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()
	_run_interactive_demo(max_new_tokens=args.max_new_tokens)
