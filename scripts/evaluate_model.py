"""Production-ready evaluation script for the LLM pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.utils import ensure_dir, get_logger, load_json_config, save_json

LOGGER = get_logger("evaluate_model")
GENERATION_CONFIG_PATH = ROOT_DIR / "config" / "generation_config.json"
VALIDATION_DATA_PATH = ROOT_DIR / "data" / "processed" / "validation.json"
ADAPTER_PATH = ROOT_DIR / "models" / "fine_tuned_adapter"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
REPORT_PATH = RESULTS_DIR / "comparison.md"


def load_environment() -> dict[str, str]:
    """Load required environment variables from the repository .env file."""
    load_dotenv(ROOT_DIR / ".env", override=False)

    required_variables = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "BASE_MODEL_ID": os.getenv("BASE_MODEL_ID"),
    }
    missing = [name for name, value in required_variables.items() if not value]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    LOGGER.info("Environment loaded")
    return {key: value for key, value in required_variables.items() if value is not None}


def load_generation_config() -> dict[str, Any]:
    """Load the inference generation configuration."""
    generation_config = load_json_config(GENERATION_CONFIG_PATH)
    LOGGER.info("Configs loaded")
    return generation_config


def load_validation_dataset(development_mode: bool) -> Any:
    """Load the validation dataset and optionally reduce its size."""
    from datasets import load_dataset

    validation_dataset = load_dataset("json", data_files={"validation": str(VALIDATION_DATA_PATH)})[
        "validation"
    ]
    if development_mode:
        validation_dataset = validation_dataset.select(range(min(100, len(validation_dataset))))
        LOGGER.info("Development mode enabled: reduced validation dataset to %s", len(validation_dataset))
    else:
        LOGGER.info("Validation dataset loaded with full size %s", len(validation_dataset))

    return validation_dataset


def build_tokenizer(model_id: str) -> Any:
    """Load and configure the tokenizer for inference."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    LOGGER.info("Tokenizer loaded")
    return tokenizer


def build_model(model_id: str) -> Any:
    """Load the base causal language model for inference."""
    from transformers import AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    import torch

    offload_dir = ROOT_DIR / "offload"
    ensure_dir(offload_dir)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        offload_folder=str(offload_dir),
    )
    LOGGER.info("Base model loaded")
    return model


def load_adapter(model: Any) -> Any:
    """Load the fine-tuned LoRA adapter and merge it for inference."""
    from peft import PeftModel

    offload_dir = ROOT_DIR / "offload"
    ensure_dir(offload_dir)
    peft_model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        offload_folder=str(offload_dir),
    )
    merged_model = peft_model.merge_and_unload()
    LOGGER.info("Adapter loaded and merged")
    return merged_model


def parse_sample_text(sample_text: str) -> tuple[str, str]:
    """Extract instruction and reference response from a formatted prompt."""
    instruction = ""
    reference = ""
    if "### Instruction:" in sample_text and "### Response:" in sample_text:
        instruction_part = sample_text.split("### Instruction:", 1)[1]
        response_part = instruction_part.split("### Response:", 1)
        instruction_text = response_part[0]
        reference = response_part[1].strip()
        if "### Context:" in instruction_text:
            instruction_text = instruction_text.split("### Context:", 1)[0]
        instruction = instruction_text.strip()
    elif "### Response:" in sample_text:
        parts = sample_text.split("### Response:", 1)
        instruction = parts[0].strip()
        reference = parts[1].strip()
    else:
        instruction = sample_text.strip()
        reference = ""
    return instruction, reference


def generate_prediction(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_config: dict[str, Any],
) -> str:
    """Generate a prediction from the model for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_kwargs = {
        "pad_token_id": tokenizer.pad_token_id,
        **generation_config,
    }
    generated = model.generate(**inputs, **generation_kwargs)
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    prediction = decoded[len(prompt) :].strip()
    return prediction


def compute_metrics(
    references: list[str],
    predictions: list[str],
    validation_dataset: Any,
    tokenizer: Any,
    model: Any,
) -> dict[str, float]:
    """Compute BLEU, ROUGE-L, and perplexity for the validation data."""
    import evaluate
    import torch

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=[[reference] for reference in references],
    )
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    rouge_l = float(rouge_result["rougeL"])

    total_loss = 0.0
    input_device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        for sample in validation_dataset:
            sample_text = sample["text"]
            tokenized = tokenizer(sample_text, return_tensors="pt", truncation=True)
            input_ids = tokenized["input_ids"].to(input_device)
            attention_mask = tokenized["attention_mask"].to(input_device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()

    average_loss = total_loss / len(validation_dataset)
    perplexity = float(torch.exp(torch.tensor(average_loss)).item())

    LOGGER.info("Metrics computed")
    return {
        "bleu": float(bleu_result["bleu"]),
        "rougeL": rouge_l,
        "perplexity": perplexity,
    }


def save_comparison_report(
    records: list[dict[str, str]],
    path: Path,
) -> None:
    """Save a markdown comparison report of predictions against references."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Evaluation Results\n\n")
        for index, record in enumerate(records, start=1):
            handle.write(f"## Sample {index}\n\n")
            handle.write("**Instruction**\n\n")
            handle.write(record["instruction"].strip() + "\n\n")
            handle.write("**Reference**\n\n")
            handle.write(record["reference"].strip() + "\n\n")
            handle.write("**Prediction**\n\n")
            handle.write(record["prediction"].strip() + "\n\n")
            handle.write("---\n\n")


def main() -> None:
    """Load assets, run evaluation, and persist results."""
    DEVELOPMENT_MODE = True
    try:
        env = load_environment()
        generation_config = load_generation_config()
        validation_dataset = load_validation_dataset(DEVELOPMENT_MODE)
        if DEVELOPMENT_MODE:
            # Temporary debug slice; revert before the final evaluation.
            validation_dataset = validation_dataset.select(range(min(20, len(validation_dataset))))

        tokenizer = build_tokenizer(env["BASE_MODEL_ID"])
        model = build_model(env["BASE_MODEL_ID"])
        model = load_adapter(model)

        LOGGER.info("Evaluation started")

        records: list[dict[str, str]] = []
        references: list[str] = []
        predictions: list[str] = []

        for sample in validation_dataset:
            prompt = sample["text"]
            instruction, reference = parse_sample_text(prompt)
            prediction = generate_prediction(model, tokenizer, prompt, generation_config)
            records.append(
                {
                    "instruction": instruction,
                    "reference": reference,
                    "prediction": prediction,
                }
            )
            references.append(reference)
            predictions.append(prediction)

        metrics = compute_metrics(references, predictions, validation_dataset, tokenizer, model)
        ensure_dir(RESULTS_DIR)
        save_json(METRICS_PATH, metrics)
        save_comparison_report(records, REPORT_PATH)

        LOGGER.info("Files saved")
        LOGGER.info("Evaluation completed")
    except Exception as exc:  # pragma: no cover - exercised during runtime diagnostics
        LOGGER.exception("Evaluation pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
