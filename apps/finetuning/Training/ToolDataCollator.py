from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForToolOnlyLM:
    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100

    def __post_init__(self):
        # Detect model type from tokenizer name or vocab
        tokenizer_name = self.tokenizer.name_or_path.lower()
        if "llama" in tokenizer_name:
            self.start_header_token = "<|start_header_id|>"
            self.end_header_token = "<|end_header_id|>"
            self.assistant_token = "assistant"
            self.eot_token = "<|eot_id|>"
            self.eos_token = "<|end_of_text|>"
            self.pad_token = "<|finetune_right_pad_id|>"
            self.masked_label_token = "M"

            self.assistant_token_id = self.tokenizer.convert_tokens_to_ids(self.assistant_token)
        elif "smol" in tokenizer_name or "smollm" in tokenizer_name:
            self.start_header_token = "<|im_start|>"
            self.end_header_token = ""  # SmolLM may not use an end_header
            self.assistant_token = "assistant"
            self.eot_token = "<|im_end|>"
            self.eos_token = "<|endoftext|>"
            self.pad_token = "<|pad|>"  # Update to actual token
            self.masked_label_token = "M"
            self.assistant_token_ids = self.tokenizer.encode(self.assistant_token, add_special_tokens=False)
        else:
            raise ValueError("Unknown model type for tokenizer. Expected 'llama' or 'smol' in tokenizer name.")

        # Convert tokens to IDs
        self.start_header_id = self.tokenizer.convert_tokens_to_ids(self.start_header_token)
        self.end_header_id = (
            self.tokenizer.convert_tokens_to_ids(self.end_header_token) if self.end_header_token else None
        )
        self.eot_id = self.tokenizer.convert_tokens_to_ids(self.eot_token)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.masked_label_token_id = self.tokenizer.convert_tokens_to_ids(self.masked_label_token)
        self.description_of_masked_label_token_id = self.tokenizer.convert_tokens_to_ids("M")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # input_ids = [f["input_ids"] for f in features]
        # attention_mask = [f["attention_mask"] for f in features]
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        # attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        attention_mask = None

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        # attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # Attention mask is 1 for tokens not equal to pad_token_id, else 0
        attention_mask_padded = (input_ids_padded != self.pad_token_id).long()

        labels = torch.full_like(input_ids_padded, fill_value=self.ignore_index)

        is_llama = self.end_header_id is not None
        for i, input_ids_seq in enumerate(input_ids):
            seq = input_ids_seq.tolist()
            j = 0
            while j < len(seq) - 1:
                if is_llama:
                    if (
                        j + 2 < len(seq)
                        and seq[j] == self.start_header_id
                        and seq[j + 1] == self.assistant_token_id
                        and seq[j + 2] == self.end_header_id
                    ):
                        start_idx = j + 3
                    else:
                        j += 1
                        continue
                else:
                    if (
                        j + len(self.assistant_token_ids) < len(seq)
                        and seq[j] == self.start_header_id
                        and seq[j + 1 : j + 1 + len(self.assistant_token_ids)] == self.assistant_token_ids
                    ):
                        start_idx = j + 1 + len(self.assistant_token_ids)
                    else:
                        j += 1
                        continue

                try:
                    end_idx = seq.index(self.eot_id, start_idx)

                    # Extend to include eos token if present
                    if end_idx + 1 < len(seq) and seq[end_idx + 1] == self.eos_id:
                        end_idx += 2
                    else:
                        end_idx += 1

                    labels[i, start_idx:end_idx] = input_ids_padded[i, start_idx:end_idx]
                    j = end_idx  # <-- important to advance past the current assistant block
                except ValueError:
                    break

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels,
        }


def inspect_collator_outputs(tokenized_dataset, collator, tokenizer, n=3):
    """
    Inspects the output of DataCollatorForToolOnlyLM on the first `n` samples.
    Prints input_ids, attention_mask, and labels with decoded text.
    Also verifies that all items are padded to the same length.
    """
    from copy import deepcopy
    import torch

    print(f"\n===============================================================")
    print(f"Inspecting first {n} examples with DataCollatorForToolOnlyLM...")
    print(f"================================================================")
    features = [deepcopy(tokenized_dataset[i]) for i in range(n)]

    batch = collator(features)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print("\n[Batch shapes]:")
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"labels: {labels.shape}")

    max_len_input_ids = max(len(f["input_ids"]) for f in features)
    print(f"\n[Expected max sequence length in batch before padding]: {max_len_input_ids}")
    print(f"[Actual padded sequence length]: {input_ids.shape[1]}")

    assert all(len(seq) == input_ids.shape[1] for seq in input_ids), "Not all input_ids are padded to the same length!"
    assert all(len(seq) == attention_mask.shape[1] for seq in attention_mask), (
        "Not all attention_masks are padded to the same length!"
    )
    assert all(len(seq) == labels.shape[1] for seq in labels), "Not all labels are padded to the same length!"

    for i in range(n):
        print(f"\n\n\n--- Example {i} ---")
        print("\n[Input IDs]:")
        print(input_ids[i].tolist())
        print("\n[Decoded Input]:")
        print(tokenizer.decode(input_ids[i], skip_special_tokens=False))

        print("\n[Attention Mask]:")
        print(attention_mask[i].tolist())

        print("\n[Labels]:")
        print(labels[i].tolist())

        # Decode labels (masking -100s)
        masked_labels = [
            tid if tid != -100 else collator.description_of_masked_label_token_id for tid in labels[i].tolist()
        ]
        print("\n[Decoded Labels] (assistant-only text, masked labels 'M'):")
        print(tokenizer.decode(masked_labels, skip_special_tokens=False))

    print(f"\n================================================================")
