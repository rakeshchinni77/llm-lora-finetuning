"""Model and tokenizer loading for inference.

This module is intentionally limited to loading concerns:
- read and validate environment
- authenticate with Hugging Face
- load tokenizer
- load base model with environment-aware strategy
- load and merge LoRA adapter

No prompt building or text generation happens here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from scripts.utils import ensure_dir, get_logger

LOGGER = get_logger("app.loader")
ADAPTER_DIR = ROOT_DIR / "models" / "fine_tuned_adapter"
OFFLOAD_DIR = ROOT_DIR / "offload"


def _load_environment() -> dict[str, str]:
	"""Load and validate required environment variables from `.env`.

	Returns:
		Dict containing required environment values.

	Raises:
		ValueError: If one or more required variables are missing.
	"""
	load_dotenv(ROOT_DIR / ".env", override=False)

	required = {
		"HF_TOKEN": os.getenv("HF_TOKEN"),
		"BASE_MODEL_ID": os.getenv("BASE_MODEL_ID"),
	}
	missing = [key for key, value in required.items() if not value]
	if missing:
		raise ValueError("Missing required environment variables: " + ", ".join(missing))

	return {key: value for key, value in required.items() if value is not None}


def _build_quantization_config() -> Any:
	"""Build the 4-bit quantization config used during successful evaluation."""
	import torch
	from transformers import BitsAndBytesConfig

	return BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_use_double_quant=True,
	)


def _can_use_4bit_gpu() -> bool:
	"""Return True when CUDA and bitsandbytes are both available."""
	import torch

	if not torch.cuda.is_available():
		LOGGER.info("Using CPU inference")
		return False

	try:
		import bitsandbytes  # noqa: F401
		LOGGER.info("Using 4-bit GPU inference")
		return True
	except ImportError:
		LOGGER.warning("bitsandbytes is unavailable; falling back to CPU inference")
		LOGGER.info("Using CPU inference")
		return False


def _load_tokenizer(model_id: str) -> Any:
	"""Load and normalize tokenizer settings for causal LM inference."""
	from transformers import AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"
	return tokenizer


def load_model_and_tokenizer() -> tuple[Any, Any]:
	"""Load merged inference model and tokenizer.

	Returns:
		Tuple of `(model, tokenizer)` with the LoRA adapter merged into the base model.
	"""
	import torch
	from huggingface_hub import login as hf_login
	from peft import PeftModel
	from transformers import AutoModelForCausalLM

	env = _load_environment()
	model_id = env["BASE_MODEL_ID"]

	ensure_dir(OFFLOAD_DIR)

	hf_login(token=env["HF_TOKEN"], add_to_git_credential=False)
	LOGGER.info("Authenticated with Hugging Face Hub")

	tokenizer = _load_tokenizer(model_id)
	LOGGER.info("Tokenizer loaded")

	use_4bit_gpu = _can_use_4bit_gpu()
	if use_4bit_gpu:
		try:
			quantization_config = _build_quantization_config()
			base_model = AutoModelForCausalLM.from_pretrained(
				model_id,
				quantization_config=quantization_config,
				device_map="auto",
				torch_dtype=torch.float16,
				offload_folder=str(OFFLOAD_DIR),
			)
			adapter_model = PeftModel.from_pretrained(
				base_model,
				str(ADAPTER_DIR),
				offload_folder=str(OFFLOAD_DIR),
			)
		except (ImportError, RuntimeError) as exc:
			LOGGER.warning(
				"4-bit GPU loading failed (%s). Falling back to CPU inference.",
				exc,
			)
			LOGGER.info("Using CPU inference")
			base_model = AutoModelForCausalLM.from_pretrained(
				model_id,
				torch_dtype=torch.float32,
			)
			base_model.to("cpu")
			adapter_model = PeftModel.from_pretrained(
				base_model,
				str(ADAPTER_DIR),
			)
	else:
		base_model = AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype=torch.float32,
		)
		base_model.to("cpu")
		adapter_model = PeftModel.from_pretrained(
			base_model,
			str(ADAPTER_DIR),
		)

	LOGGER.info("Base model loaded")
	merged_model = adapter_model.merge_and_unload()
	merged_model.eval()
	LOGGER.info("LoRA adapter loaded and merged")

	return merged_model, tokenizer
