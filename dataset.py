from transformers import GPT2TokenizerFast, BatchEncoding
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, TypedDict
from functools import partial
import torch

class DataloaderOutput(TypedDict):
    chosen_input_ids: torch.Tensor
    rejected_input_ids : torch.Tensor
    chosen_attention_mask : torch.Tensor
    rejected_attention_mask : torch.Tensor
    chosen_loss_mask : torch.Tensor
    rejected_loss_mask : torch.Tensor
    prompt_lengths:List[int]

class DPO_Dataset(Dataset):
    def __init__(self, is_train: bool = True):
        super().__init__()
        ds_to_dpo = load_dataset(
            "Anthropic/hh-rlhf"
        )
        self.samples = ds_to_dpo["train" if is_train else "test"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, str, str]:
        chosen_text = self.samples["chosen"][index]
        rejected_text = self.samples["rejected"][index]
        return self.get_prompt(chosen_text, rejected_text)

    def get_prompt(self, chosen: str, rejected: str) -> Tuple[str, str, str]:
        prompt = []
        chosen_only = []
        rejected_only = []

        chosen_parts = chosen.split("\n\n")
        rejected_parts = rejected.split("\n\n")

        for p_chosen, p_rejected in zip(chosen_parts, rejected_parts):

            if p_chosen == '' and p_rejected == '':
                continue

            if p_chosen == p_rejected:
                prompt.append(p_chosen)

            else:
                chosen_only.append(p_chosen)
                rejected_only.append(p_rejected)

        return (
            "\n\n".join(prompt),
            "\n\n".join(chosen_only),
            "\n\n".join(rejected_only),
        )


tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def concat_encodings(p_encoding: BatchEncoding, c_encoding: BatchEncoding, truncate = False, max_length=1024) -> BatchEncoding:
    concatted: BatchEncoding = {"input_ids": []}

    for p_enc, c_enc in zip(p_encoding["input_ids"], c_encoding["input_ids"]):
        if len(p_enc)+len(c_enc)>max_length and truncate:
            tokens_to_remove = len(p_enc)+len(c_enc) - max_length
        else:
            tokens_to_remove = 0
        concatted["input_ids"].append((p_enc+c_enc)[tokens_to_remove:])
    return concatted


def add_token_to_seq(seq: List[List[int]], token: int, to_end: bool = False):
    for idx in range(len(seq)):
        if to_end:
            seq[idx] = seq[idx] + [token]
        else:
            seq[idx] = [token] + seq[idx]

def collate_fn(batch, add_bos=False, truncation = True) -> DataloaderOutput:
    prompt = [i[0] for i in batch]
    chosen = [i[1] for i in batch]
    rejected = [i[2] for i in batch]
    output = {}
    prompt_out = tokenizer(
        prompt, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    chosen_out = tokenizer(
        chosen, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    rejected_out = tokenizer(
        rejected, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    if tokenizer.bos_token is not None and add_bos:
        add_token_to_seq(prompt_out["input_ids"], token=tokenizer.bos_token_id)
    prompt_lengths = [len(tokens) for tokens in prompt_out["input_ids"]]
    if tokenizer.eos_token is not None:
        add_token_to_seq(chosen_out["input_ids"],
                         token=tokenizer.eos_token_id, to_end=True)
        add_token_to_seq(rejected_out["input_ids"],
                         token=tokenizer.eos_token_id, to_end=True)
    if truncation:
        chosen_lengths = [len(tokens) for tokens in chosen_out["input_ids"]]
        rejected_lengths = [len(tokens) for tokens in rejected_out["input_ids"]]

    preferred_completions = concat_encodings(prompt_out, chosen_out, truncate=truncation)
    dispreferred_completions = concat_encodings(prompt_out, rejected_out, truncate=truncation)

    

    preferred_completions = tokenizer.pad(
        preferred_completions, return_attention_mask=True, padding="max_length", max_length=1024, return_tensors="pt")
    dispreferred_completions = tokenizer.pad(
        dispreferred_completions, return_attention_mask=True, padding="max_length", max_length=1024, return_tensors="pt")
    
    chosen_loss_mask = torch.clone(preferred_completions["attention_mask"])
    rejected_loss_mask = torch.clone(dispreferred_completions["attention_mask"])

    if truncation:
        for idx, (c_length, r_length) in enumerate(zip(chosen_lengths,rejected_lengths)):
            comp_size = chosen_loss_mask[idx].sum(dim=-1).item()
            discomp_size = rejected_loss_mask[idx].sum(dim=-1).item()
            prompt_size_c = comp_size - c_length
            prompt_size_r = discomp_size - r_length
            chosen_loss_mask[idx, :prompt_size_c] = 0
            rejected_loss_mask[idx, :prompt_size_r] = 0
    else:
        for idx,length in enumerate(prompt_lengths):
            chosen_loss_mask[idx, :length] = 0
            rejected_loss_mask[idx, :length] = 0

    output["chosen_input_ids"] = preferred_completions["input_ids"]
    output["rejected_input_ids"] = dispreferred_completions["input_ids"]
    output["chosen_attention_mask"] = preferred_completions["attention_mask"]
    output["rejected_attention_mask"] = dispreferred_completions["attention_mask"]
    output["chosen_loss_mask"] = chosen_loss_mask
    output["rejected_loss_mask"] = rejected_loss_mask
    output["prompt_lengths"] = prompt_lengths

    return output


dataset_tr = DPO_Dataset()
dataloader_tr = DataLoader(
    dataset=dataset_tr, batch_size=8, collate_fn=partial(collate_fn,truncation=True, add_bos = False))