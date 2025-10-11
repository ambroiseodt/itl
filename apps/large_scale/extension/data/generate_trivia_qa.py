# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module containing utilities to create HuggingFace compatible databases for the trivia QA task.

Copyright (c) 2025 by the authors
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value, load_dataset

TRIVIA_DATASET_PATH = Path(__file__).parents[1] / "trivia_datasets"


# ---------- Features ----------
_MSG = Features({"role": Value("string"), "content": Value("string")})
_FEATS = Features(
    {
        "qa": Sequence(_MSG),
        "qatool": Sequence(_MSG),
        "id": Value("string"),
        "question": Value("string"),
        "answer_value": Value("string"),
        "answer_aliases": Sequence(Value("string")),
    }
)


# ---------- Helpers ----------
def _pick_answer(ans_obj: Any) -> tuple[str, list[str]]:
    """
    TriviaQA 'answer' can be:
      - {"value": str, "aliases": [str, ...]}
      - str
      - [str, ...]
    Returns (canonical_value, aliases_list).
    """
    if isinstance(ans_obj, dict):
        val = (ans_obj.get("value") or "").strip()
        aliases = [a.strip() for a in (ans_obj.get("aliases") or []) if isinstance(a, str) and a.strip()]
        return val, aliases
    if isinstance(ans_obj, str):
        return ans_obj.strip(), []
    if isinstance(ans_obj, list) and ans_obj:
        return (ans_obj[0] or "").strip(), [str(x).strip() for x in ans_obj[1:] if str(x).strip()]
    return "", []


def _qa_messages(q: str, a: str) -> list[dict[str, str]]:
    # Closed-book supervision
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]


def _python_tool_prompt(q: str) -> str:
    # Clear Python-flavored “database access” template (no SQL).
    return (
        "To answer this question I will query the database using Python:\n"
        "```python\n"
        f'answer = query_database("{q}")\n'
        "print(answer)\n"
        "```"
    )


def _qatool_messages(q: str, a: str) -> list[dict[str, str]]:
    # Tool-use supervision with an ipython tool return (like your old dataset)
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": _python_tool_prompt(q)},
        {"role": "ipython", "content": a},  # simulated tool stdout
        {"role": "assistant", "content": f"The answer is {a}."},  # finalization step
    ]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s).lower().strip()
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


# ---------- Builder ----------
def build_triviaqa_chat(split="validation", config="unfiltered", limit=None) -> Dataset:
    raw = load_dataset("trivia_qa", config, split=split)
    qa_rows, qatool_rows, ids, qs, avals, aliases = [], [], [], [], [], []

    n = len(raw) if limit is None else min(limit, len(raw))
    for i in range(n):
        ex = raw[i]
        qid = str(ex.get("question_id", i))
        q = (ex.get("question") or "").strip()
        a_val, a_aliases = _pick_answer(ex.get("answer"))

        if not q or not a_val:
            continue

        qa_rows.append(_qa_messages(q, a_val))
        qatool_rows.append(_qatool_messages(q, a_val))
        ids.append(qid)
        qs.append(q)
        avals.append(a_val)
        aliases.append(a_aliases)

    ds = Dataset.from_dict(
        {
            "qa": qa_rows,
            "qatool": qatool_rows,
            "id": ids,
            "question": qs,
            "answer_value": avals,
            "answer_aliases": aliases,
        },
        # features=_FEATS,
    )
    return ds


# ---------- I/O ----------
def export_preview(dataset: Dataset, out_path: Path, k: int = 10):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(min(k, len(dataset))):
            f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")


def print_dataset_structure(dataset: Dataset):
    print("Dataset Structure:")
    print(f"- num_rows: {len(dataset)}")
    print(f"- features: {dataset.features}")
    if len(dataset):
        print("- sample row:", json.dumps(dataset[0], ensure_ascii=False)[:400], "...")


def build_database(n_facts: int = None):
    """
    # triviaqa_to_chat.py
    Build a HuggingFace dataset with:
      - qa:     [{"role":"user","content":...}, {"role":"assistant","content":...}]
      - qatool: [{"role":"user","content":...},
                 {"role":"assistant","content": "python tool call"},
                 {"role":"ipython","content": "<answer>"},
                 {"role":"assistant","content": "...final..."}]

    Usage:
      python triviaqa_to_chat.py \
          --out_dir /path/to/TriviaQA_HF_chat \
          --split validation --config unfiltered --limit 30000 \
          --preview /tmp/trivia_preview.jsonl
    """
    dataset_name = "trivia_dataset"
    ds = build_triviaqa_chat(split="train", limit=n_facts)
    print_dataset_structure(ds)
    if n_facts is None:
        n_facts = "all"
    out_dir = TRIVIA_DATASET_PATH / f"{dataset_name}_{n_facts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Saved dataset to {out_dir}")


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.INFO)

    fire.Fire(
        {
            "build": build_database,
        }
    )
