"""Prepare the Dolly instruction dataset for future fine-tuning workflows.

This script downloads the public databricks/databricks-dolly-15k dataset,
formats each sample into the required instruction-following template, filters
invalid rows, splits the data into train and validation partitions, and writes
JSON files in the repository's processed-data directory.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from datasets import load_dataset

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import ensure_dir, get_logger, set_seed


LOGGER = get_logger("prepare_data")
SEED = 42
MAX_CHAR_LENGTH = 4000
LONG_SAMPLE_WARNING_THRESHOLD = 8000
PREVIEW_SAMPLE_COUNT = 3
ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.json"
VALIDATION_PATH = PROCESSED_DIR / "validation.json"


def _normalize_text(value: Any) -> str:
    """Normalize a dataset field to a stripped string.

    Args:
        value: The value to normalize.

    Returns:
        A stripped string, or an empty string when the input is missing.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def format_instruction(sample: dict[str, Any]) -> str:
    """Format a raw dataset sample using the required instruction template.

    Args:
        sample: The raw dataset sample.

    Returns:
        A fully formatted prompt string.
    """
    instruction = _normalize_text(sample.get("instruction"))
    response = _normalize_text(sample.get("response"))
    context = _normalize_text(sample.get("context"))

    if context:
        return (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Response:\n"
            f"{response}"
        )

    return (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{response}"
    )


def filter_samples(dataset: Any) -> list[dict[str, str]]:
    """Filter out invalid samples and return formatted records.

    Args:
        dataset: The dataset object to process.

    Returns:
        A list of dictionaries containing formatted text records.
    """
    records: list[dict[str, str]] = []
    removed_samples = 0

    for sample in dataset:
        instruction = _normalize_text(sample.get("instruction"))
        response = _normalize_text(sample.get("response"))

        if not instruction or not response:
            removed_samples += 1
            continue

        if len(instruction) + len(response) > MAX_CHAR_LENGTH:
            removed_samples += 1
            continue

        formatted_text = format_instruction(sample)
        if len(formatted_text) > LONG_SAMPLE_WARNING_THRESHOLD:
            LOGGER.warning(
                "Very long sample detected (length=%s characters). The sample will be kept. It may be truncated later during tokenization in Phase 4.",
                len(formatted_text),
            )

        records.append({"text": formatted_text})

    LOGGER.info("Filtering complete: removed %s invalid samples", removed_samples)
    return records


def preview_samples(samples: list[dict[str, str]], num_samples: int = 3) -> None:
    """Print a preview of the first formatted prompts.

    Args:
        samples: The processed samples to preview.
        num_samples: The number of samples to display.
    """
    preview_items = samples[: max(0, num_samples)]
    for index, sample in enumerate(preview_items, start=1):
        print("=" * 80)
        print(f"\nSample {index}\n")
        print(sample["text"])
        print(f"\n{'=' * 80}")


def _percentile(values: list[int], percentile: float) -> float:
    """Calculate a percentile using linear interpolation.

    Args:
        values: Numeric values to summarize.
        percentile: Percentile to calculate (for example 95 for the 95th percentile).

    Returns:
        The interpolated percentile value.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)

    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    weight = rank - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def write_json_records(path: Path, records: list[dict[str, str]]) -> None:
    """Write a list of records to a JSON file.

    Args:
        path: Destination path for the JSON output.
        records: The records to serialize.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    """Download the dataset, prepare records, split them, and save outputs."""
    set_seed(SEED)
    ensure_dir(PROCESSED_DIR)

    LOGGER.info("Downloading dataset databricks/databricks-dolly-15k")
    raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    original_size = len(raw_dataset)
    LOGGER.info("Original dataset size: %s", original_size)

    formatted_records = filter_samples(raw_dataset)
    filtered_size = len(formatted_records)
    LOGGER.info("Filtered dataset size: %s", filtered_size)

    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    train_records = filter_samples(train_dataset)
    validation_records = filter_samples(validation_dataset)

    write_json_records(TRAIN_PATH, train_records)
    write_json_records(VALIDATION_PATH, validation_records)

    LOGGER.info("Train output path: %s", TRAIN_PATH)
    LOGGER.info("Validation output path: %s", VALIDATION_PATH)

    combined_records = train_records + validation_records
    lengths = [len(record["text"]) for record in combined_records]
    total_samples = len(combined_records)
    removed_samples = original_size - total_samples
    minimum_length = min(lengths, default=0)
    average_length = statistics.mean(lengths) if lengths else 0.0
    median_length = statistics.median(lengths) if lengths else 0.0
    percentile_95 = _percentile(lengths, 95)
    maximum_length = max(lengths, default=0)

    preview_samples(combined_records, num_samples=PREVIEW_SAMPLE_COUNT)

    LOGGER.info("Dataset preparation complete")
    LOGGER.info("Total samples: %s", total_samples)
    LOGGER.info("Train samples: %s", len(train_records))
    LOGGER.info("Validation samples: %s", len(validation_records))
    LOGGER.info("Removed samples: %s", removed_samples)
    LOGGER.info("Minimum character length: %s", minimum_length)
    LOGGER.info("Median character length: %.2f", median_length)
    LOGGER.info("95th percentile character length: %.2f", percentile_95)
    LOGGER.info("Maximum character length: %s", maximum_length)
    LOGGER.info("Average character length: %.2f", average_length)


if __name__ == "__main__":
    main()
