# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module containing utilities to create HuggingFace compatible databases for the factual recall task.

@ 2025, Ambroise Odonnat
"""

import csv
import json
import logging
import random
import re
from itertools import product
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from jinja2 import Template
from tqdm import tqdm

HF_DATASET_PATH = Path(__file__).parents[1] / "HF_datasets"
MEMORY_DATASET_PATH = Path(__file__).parents[3] / "apps/memory/dataset"


def generate_hf_dataset(
    n: int,
    atom_dir: str,
    template_dir: str,
    seed: int = 42,
) -> Dataset:
    """
    Directly generate a HuggingFace-compatible dataset for QA and QATool-style chat dialogues.

    Returns a Dataset with columns:
        - qa: list of {"role": ..., "content": ...}
        - qatool: list of {"role": ..., "content": ...} with 'database' → 'ipython'
        - name: name of that person
        - attribute: attribute of named
        - value: value of attribute
    """
    atom_dir = Path(atom_dir)
    template_dir = Path(template_dir)

    # Load atoms
    with open(atom_dir / "first_names_extended.txt") as f:
        first_names = f.read().splitlines()
    with open(atom_dir / "last_names.txt") as f:
        last_names = f.read().splitlines()
    with open(atom_dir / "cities.txt") as f:
        cities = f.read().splitlines()
    with open(atom_dir / "countries.txt") as f:
        countries = f.read().splitlines()
    with open(atom_dir / "occupations.txt") as f:
        occupations = f.read().splitlines()

    # Shuffle people
    random.seed(seed)
    all_pairs = list(product(first_names, last_names))
    random.shuffle(all_pairs)
    selected_people = all_pairs[:n]

    # Load templates
    def load_templates(suffix: str) -> list:
        pattern = re.compile(r"^<\|(\w+)\|>(.*)$")
        templates = []
        for file in sorted(template_dir.glob(f"{suffix}?.j2")):
            with open(file) as f:
                lines = f.read().splitlines()
            out, source, buffer = [], None, []
            for line in lines:
                match = pattern.match(line)
                if match:
                    if source is not None:
                        out.append({"source": source, "content": Template("\n".join(buffer))})
                    source, msg = match.groups()
                    buffer = [msg.strip()]
                else:
                    buffer.append(line.strip())
            if source is not None:
                out.append({"source": source, "content": Template("\n".join(buffer))})
            templates.append(out)
        return templates

    qa_templates = load_templates("qa")
    qatool_templates = load_templates("qatool")

    assert qa_templates, "No QA templates found!"
    assert qatool_templates, "No QATool templates found!"

    data = []
    for _, (first, last) in enumerate(tqdm(selected_people, desc="Generating data")):
        person = {
            "name": f"{first} {last}",
            "birth_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(1950, 2000)}",
            "birth_place": f"{random.choice(countries)}",
            "current_address": f"{random.choice(cities)}",
            "occupation": random.choice(occupations),
        }

        for qa_dialog, qatool_dialog in zip(qa_templates, qatool_templates, strict=False):

            def render_chat(dialog: list, attribute: dict = person):
                rendered = []
                for msg in dialog:
                    try:
                        content = msg["content"].render(**attribute)
                    except Exception as e:
                        print(f"Template render failed: {e}")
                        continue
                    role = msg["source"].lower()
                    if role == "answer":
                        continue  # exclude 'answer' role from chat
                    if role == "database":
                        role = "ipython"
                    rendered.append({"role": role, "content": content})
                return rendered

            qa_chat = render_chat(qa_dialog)
            qatool_chat = render_chat(qatool_dialog)

            # Extract attribute from SQL in assistant message
            sql_line = next(
                (m["content"] for m in qatool_chat if m["role"] == "assistant" and "```sql" in m["content"]), ""
            )
            sql_match = re.search(r"FIND\s+(\w+)\s+FOR", sql_line)
            attribute = sql_match.group(1).lower() if sql_match else "unknown"

            # Extract value from ipython/tool output
            value = next((m["content"] for m in qatool_chat if m["role"] == "ipython"), "").strip()

            data.append(
                {
                    "qa": qa_chat,
                    "qatool": qatool_chat,
                    "name": person["name"],
                    "attribute": attribute,
                    "value": value,
                }
            )

    return Dataset.from_list(data)


def print_dataset_structure(dataset: Any) -> None:
    if isinstance(dataset, DatasetDict):
        print("DatasetDict Structure:")
        for split in dataset.keys():
            print(f"\nSplit: {split}")
            print(f"Number of examples: {len(dataset[split])}")
            print(f"Features: {dataset[split].features}")
            print("Sample Data:")
            print(dataset[split][0])  # Print the first example
    elif isinstance(dataset, Dataset):
        print("Dataset Structure (Single Split):")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {dataset.features}")
        print("Sample Data:")
        print(dataset[0])  # Print the first example
    else:
        print("Unknown dataset format!")


def export_dataset(dataset: Dataset, output_path: str, filetype: str = "jsonl", limit: int = None) -> None:
    """
    Export a Hugging Face dataset saved to disk into a .jsonl or .csv file.

    Args:
        dataset_path (str): Path to the saved dataset directory.
        output_path (str): Path to save the exported file.
        filetype (str): One of "jsonl" or "csv".
        limit (int): Maximum number of examples to export (optional).
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        dataset = dataset.select(range(min(len(dataset), limit)))

    if filetype == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Exported {len(dataset)} examples to {output_path}")

    elif filetype == "csv":
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dataset.column_names)
            writer.writeheader()
            for example in dataset:
                writer.writerow(example)
        print(f"Exported {len(dataset)} examples to {output_path}")

    else:
        raise ValueError("Unsupported filetype. Use 'jsonl' or 'csv'.")


def format_with_underscores(number: int) -> str:
    return f"{number:_}"


def build_database(n_people: int = 50000) -> None:
    """
    Build database for factual recall task and save it to disk. The dataset size
    is 4 times the number of people, as each person has 4 attributes.

    Usage:
    Build a database with 200_000 facts (50_000 people) by running
    ```bash
    python -m apps.large_scale.data.HF_dataset_generation build --n_people 50000
    ```
    """

    # Paths and configs
    atom_dir = MEMORY_DATASET_PATH / "atoms"
    template_dir = MEMORY_DATASET_PATH / "templates"
    dataset_name = f"HF_dataset_{format_with_underscores(4 * n_people)}"
    save_dir = HF_DATASET_PATH / f"{dataset_name}"

    # Generate dataset
    dataset = generate_hf_dataset(n_people, atom_dir, template_dir)
    print(f"Generated dataset with {len(dataset)} examples.")
    print_dataset_structure(dataset)

    # Save dataset
    dataset.save_to_disk(save_dir)
    print(f"Dataset saved to {save_dir}")
    dataset = Dataset.load_from_disk(save_dir)
    print(f"Reloaded dataset from {save_dir} for verification.")

    # Export dataset for inspection
    export_dataset(
        dataset=dataset,
        output_path=HF_DATASET_PATH / f"{dataset_name}_preview.jsonl",
        filetype="jsonl",
        limit=10,
    )


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.INFO)

    fire.Fire(
        {
            "build": build_database,
        }
    )
