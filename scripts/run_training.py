"""Production-ready QLoRA fine-tuning entry point for the LLM pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.utils import ensure_dir, get_logger, load_json_config, set_seed
TRAIN_CONFIG_PATH = ROOT_DIR / "config" / "train_config.json"
LORA_CONFIG_PATH = ROOT_DIR / "config" / "lora_config.json"
TRAIN_DATA_PATH = ROOT_DIR / "data" / "processed" / "train.json"
VALIDATION_DATA_PATH = ROOT_DIR / "data" / "processed" / "validation.json"
LOGGER = get_logger("run_training")


def load_environment() -> dict[str, str]:
    """Load required runtime environment variables.

    Returns:
        A dictionary containing the required environment values.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    load_dotenv(ROOT_DIR / ".env", override=False)

    required_variables = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "BASE_MODEL_ID": os.getenv("BASE_MODEL_ID"),
    }

    missing = [name for name, value in required_variables.items() if not value]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return {key: value for key, value in required_variables.items() if value is not None}


def configure_huggingface_and_wandb(env: dict[str, str]) -> None:
    """Authenticate to Hugging Face Hub and Weights & Biases.

    Args:
        env: Environment variables containing authentication tokens.
    """
    from huggingface_hub import login as hf_login

    hf_login(token=env["HF_TOKEN"], add_to_git_credential=False)
    LOGGER.info("Authenticated to Hugging Face Hub.")

    import wandb

    # Modern W&B uses short-lived API keys (e.g. wandb_v1_...).
    # Preserve loading from the environment file but rely on the
    # runtime environment variable for the login call so wandb can
    # handle the newer formats automatically.
    if env.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = env["WANDB_API_KEY"]

    # Call plain login() so wandb uses the environment or interactive
    # auth flow as appropriate for the installed client version.
    wandb.login()
    LOGGER.info("Initialized Weights & Biases.")


def load_training_assets() -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Load training configuration and the processed datasets.

    Returns:
        A tuple containing the training config, LoRA config, training dataset,
        and validation dataset.
    """
    train_config = load_json_config(TRAIN_CONFIG_PATH)
    lora_config = load_json_config(LORA_CONFIG_PATH)

    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_DATA_PATH),
            "validation": str(VALIDATION_DATA_PATH),
        },
    )
    return train_config, lora_config, dataset["train"], dataset["validation"]


def build_tokenizer(model_id: str) -> Any:
    """Load and configure the tokenizer for causal language modeling.

    Args:
        model_id: The base model identifier.

    Returns:
        A configured tokenizer instance.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    LOGGER.info("Loaded tokenizer for model %s", model_id)
    return tokenizer


def build_quantization_config() -> Any:
    """Create the BitsAndBytes configuration for QLoRA.

    Returns:
        A BitsAndBytesConfig instance.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_model(model_id: str, quantization_config: Any) -> Any:
    """Load the quantized causal language model.

    Args:
        model_id: The base model identifier.
        quantization_config: The QLoRA quantization configuration.

    Returns:
        The loaded model.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    LOGGER.info("Loaded model %s with 4-bit quantization.", model_id)
    return model


def prepare_model_for_training(model: Any) -> Any:
    """Prepare the model for k-bit training.

    Args:
        model: The loaded model instance.

    Returns:
        The prepared model instance.
    """
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    LOGGER.info("Prepared model for k-bit training.")
    return model


def apply_lora(model: Any, lora_config: dict[str, Any]) -> Any:
    """Apply LoRA adapters to the prepared model.

    Args:
        model: The prepared model instance.
        lora_config: The LoRA configuration dictionary.

    Returns:
        The PEFT-enabled model instance.
    """
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["lora_alpha"]),
        lora_dropout=float(lora_config["lora_dropout"]),
        bias=str(lora_config["bias"]),
        task_type=str(lora_config["task_type"]),
        target_modules=[str(module) for module in lora_config["target_modules"]],
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    LOGGER.info("Applied LoRA adapters to the model.")
    return peft_model


def build_training_arguments(train_config: dict[str, Any]) -> Any:
    """Create the Hugging Face training arguments from config values.

    Args:
        train_config: The training configuration dictionary.

    Returns:
        A TrainingArguments instance.
    """
    from transformers import TrainingArguments

    output_dir = ROOT_DIR / str(train_config["output_dir"])
    logging_dir = ROOT_DIR / str(train_config["logging_dir"])
    ensure_dir(output_dir)
    ensure_dir(logging_dir)

    return TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        num_train_epochs=int(train_config["num_train_epochs"]),
        learning_rate=float(train_config["learning_rate"]),
        per_device_train_batch_size=int(train_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        logging_steps=int(train_config["logging_steps"]),
        save_steps=int(train_config["save_steps"]),
        eval_steps=int(train_config["eval_steps"]),
        weight_decay=float(train_config["weight_decay"]),
        evaluation_strategy=str(train_config["evaluation_strategy"]),
        save_strategy=str(train_config["save_strategy"]),
        optim=str(train_config["optimizer"]),
        report_to=["wandb"],
        fp16=True,
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    train_config: dict[str, Any],
    lora_config: dict[str, Any],
) -> Any:
    """Create the SFTTrainer instance.

    Args:
        model: The PEFT-enabled model.
        tokenizer: The tokenizer instance.
        train_dataset: The processed training dataset.
        eval_dataset: The processed validation dataset.
        train_config: The training configuration dictionary.
        lora_config: The LoRA configuration dictionary.

    Returns:
        An SFTTrainer instance.
    """
    from trl import SFTTrainer

    training_args = build_training_arguments(train_config)
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=int(train_config["max_seq_length"]),
        peft_config=None,
        packing=False,
        remove_unused_columns=False,
    )


def train_model() -> None:
    """Run the full QLoRA training workflow."""
    env = load_environment()
    configure_huggingface_and_wandb(env)

    train_config, lora_config, train_dataset, eval_dataset = load_training_assets()
    set_seed(int(train_config["seed"]))

    model_id = env["BASE_MODEL_ID"]
    tokenizer = build_tokenizer(model_id)
    quantization_config = build_quantization_config()
    model = build_model(model_id, quantization_config)
    model = prepare_model_for_training(model)
    model = apply_lora(model, lora_config)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_config=train_config,
        lora_config=lora_config,
    )

    LOGGER.info("Starting training run.")
    trainer.train()

    output_dir = ROOT_DIR / str(train_config["output_dir"])
    ensure_dir(output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Saved trained adapters to %s", output_dir)


def main() -> None:
    """Execute the training pipeline and log any failures."""
    try:
        train_model()
    except Exception as exc:  # pragma: no cover - exercised during runtime diagnostics
        LOGGER.exception("Training pipeline failed: %s", exc)
        raise
    finally:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
