from dataset import dataloader_tr, DataloaderOutput
from utils import selective_log_softmax, generate_weight
from hyperparams import *
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
import tqdm
accelerator = Accelerator(gradient_accumulation_steps=ACC_STEPS,mixed_precision="bf16",)

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)
model_tr = AutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    device_map="auto",
    quantization_config=quant_config,
    use_cache=False
)

model_tr = prepare_model_for_kbit_training(model_tr)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model_tr = get_peft_model(model_tr, lora_config)
model_tr.train()

model_ref = AutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    device_map="auto",
    quantization_config=quant_config,
    use_cache=True,
)
model_ref.eval()


def dpo_forward(batch: DataloaderOutput, perform_rpo: bool = True):
    with accelerator.autocast():
      ch_outputs_tr = model_tr(
          input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
      re_outputs_tr = model_tr(
          input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
      with torch.no_grad():
          ch_outputs_ref = model_ref(
              batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
          re_outputs_ref = model_ref(
              batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits

    ch_labels = torch.roll(batch["chosen_input_ids"], shifts=-1, dims=1)
    re_labels = torch.roll(batch["rejected_input_ids"], shifts=-1, dims=1)

    ch_loss_mask = torch.roll(
        batch["chosen_loss_mask"], shifts=-1, dims=1).bool()
    re_loss_mask = torch.roll(
        batch["rejected_loss_mask"], shifts=-1, dims=1).bool()

    chosen_log_ps_tr = selective_log_softmax(
        ch_outputs_tr, ch_labels, ch_loss_mask)
    rejected_log_ps_tr = selective_log_softmax(
        re_outputs_tr, re_labels, re_loss_mask)
    chosen_log_ps_ref = selective_log_softmax(
        ch_outputs_ref, ch_labels, ch_loss_mask)
    rejected_log_ps_ref = selective_log_softmax(
        re_outputs_ref, re_labels, re_loss_mask)

    if perform_rpo:
        logits = ch_outputs_tr[:, :-1, :]
        labels = batch["chosen_input_ids"][:, 1:]
        loss_mask = batch["chosen_loss_mask"][:, 1:]

        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        loss_mask = loss_mask.reshape(-1).bool()

        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        ce_loss = ce_loss.masked_fill(~loss_mask, 0.0)
        ce_loss = ce_loss.sum() / loss_mask.sum()


        return chosen_log_ps_tr.sum(dim=-1), chosen_log_ps_ref.sum(dim=-1), rejected_log_ps_tr.sum(dim=-1), rejected_log_ps_ref.sum(dim=-1), ce_loss

    return chosen_log_ps_tr.sum(dim=-1), chosen_log_ps_ref.sum(dim=-1), rejected_log_ps_tr.sum(dim=-1), rejected_log_ps_ref.sum(dim=-1)

def dpo_loss(chosen_log_ps_tr: torch.FloatTensor, chosen_log_ps_ref: torch.FloatTensor, rejected_log_ps_tr: torch.FloatTensor, rejected_log_ps_ref: torch.FloatTensor, beta:float, rpo_loss: Optional[torch.FloatTensor] = None, alpha: Optional[float] = 0):
    preferred_logits = chosen_log_ps_tr - chosen_log_ps_ref
    dispreferred_logits = rejected_log_ps_tr - rejected_log_ps_ref

    loss = -F.logsigmoid(beta*preferred_logits - beta*dispreferred_logits)
    loss_reduced = loss.mean()

    if alpha>0 and rpo_loss is not None:
        loss_reduced = loss_reduced + alpha * rpo_loss
    reward_acc = (preferred_logits.detach()>dispreferred_logits.detach()).float().mean()
    return loss_reduced, (preferred_logits - dispreferred_logits).detach().mean(), reward_acc

def train_epoch(optimizer: torch.optim.Optimizer, dataloader, scheduler, loop: Optional[tqdm.tqdm] = None):
    global ALL_LOSS, ALL_REWARDS, total_steps, ALL_ACCURACIES
    average_reward_margin = []
    losses = []
    step_counter = 1
    optimizer.zero_grad()
    for batch in dataloader:
        with accelerator.accumulate(model_tr):
            chosen_log_probs_train, chosen_log_probs_reference, rejected_log_probs_train, rejected_log_probs_reference, rpo_loss = dpo_forward(batch, perform_rpo = True)
            batch_loss, reward_margin, acc = dpo_loss(chosen_log_probs_train, chosen_log_probs_reference, rejected_log_probs_train, rejected_log_probs_reference, BETA, rpo_loss, ALPHA)
            accelerator.backward(batch_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model_tr.parameters(), max_norm=1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        losses.append(
            batch_loss.detach().item()
        )
        ALL_LOSS.append(
            batch_loss.detach().item()
        )
        average_reward_margin.append(
            reward_margin.item()
        )
        ALL_REWARDS.append(
            reward_margin.item()
        )
        ALL_ACCURACIES.append(
            acc.item()
        )
        step_counter += 1

        if loop:
            loop.set_postfix(loss = losses[-1], rewards = average_reward_margin[-1], step_count = f"{step_counter}/{total_steps//NUM_EPOCHS}", accuracy = acc.item())
    return np.mean(losses), np.mean(average_reward_margin)


optimizer = bnb.optim.AdamW(model_tr.parameters(), weight_decay = .005, lr = 2e-5, is_paged = True)
TRAIN_LOSS = []
ALL_LOSS = []
ALL_REWARDS = []
ALL_ACCURACIES = []
TRAIN_REWARD = []
loop = tqdm.tqdm(range(1, NUM_EPOCHS+1), desc=f"Epoch 1/{NUM_EPOCHS}")

model_tr, optimizer, dataloader_tr = accelerator.prepare(model_tr, optimizer, dataloader_tr)
total_steps = len(dataloader_tr) * NUM_EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps,
)

for epoch in loop:
    loop.set_description(f"EPOCH {epoch}/{NUM_EPOCHS}")
    losses, rewards = train_epoch(optimizer, dataloader_tr, scheduler, loop)
    if TRAIN_LOSS==[]:
        best_model = True
    elif np.min(TRAIN_LOSS)>losses and np.max(TRAIN_REWARD)<rewards:
        best_model = True
    else:
        best_model = False

    TRAIN_LOSS.append(losses)
    TRAIN_REWARD.append(rewards)

    if best_model:
        accelerator.unwrap_model(model_tr).save_pretrained("./best_model/dpo_model")