from dataset import dataloader_tr, DataloaderOutput
from utils import selective_log_softmax
from hyperparams import *
from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
import tqdm

model_tr = GPT2LMHeadModel.from_pretrained("gpt2-large").cuda()
model_ref = GPT2LMHeadModel.from_pretrained("gpt2-large").cuda()

model_tr.train()
model_ref.eval()


def dpo_forward(batch: DataloaderOutput, perform_rpo: bool = True):

    ch_outputs_tr = model_tr(
        input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits

    re_outputs_tr = model_tr(
        batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
    with torch.no_grad():
        ch_outputs_ref = model_tr(
            batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
        re_outputs_ref = model_tr(
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
        mask = ch_loss_mask[:,:-1].reshape(-1)
        shifted_logits = ch_outputs_tr[:,:-1,:].flatten(end_dim=1)
        shifted_labels = ch_labels[:,:-1].flatten(end_dim=1)
        cross_entropy = F.cross_entropy(shifted_logits,shifted_labels, reduction="none")
        cross_entropy = cross_entropy.masked_fill(~mask, .0)
        cross_entropy = cross_entropy.sum()/mask.sum()

        return chosen_log_ps_tr.sum(dim=-1), chosen_log_ps_ref.sum(dim=-1), rejected_log_ps_tr.sum(dim=-1), rejected_log_ps_ref.sum(dim=-1), cross_entropy

    return chosen_log_ps_tr.sum(dim=-1), chosen_log_ps_ref.sum(dim=-1), rejected_log_ps_tr.sum(dim=-1), rejected_log_ps_ref.sum(dim=-1)

def dpo_loss(chosen_log_ps_tr: torch.FloatTensor, chosen_log_ps_ref: torch.FloatTensor, rejected_log_ps_tr: torch.FloatTensor, rejected_log_ps_ref: torch.FloatTensor, beta:float, rpo_loss: Optional[torch.FloatTensor] = None, alpha: Optional[float] = 0):
    preferred_logits = chosen_log_ps_tr - chosen_log_ps_ref
    dispreferred_logits = rejected_log_ps_tr - rejected_log_ps_ref

    loss = -F.logsigmoid(beta*preferred_logits - beta*dispreferred_logits)
    loss_reduced = loss.mean()

    if alpha>0 and rpo_loss is not None:
        loss_reduced = loss_reduced + alpha * rpo_loss

    return loss_reduced, (preferred_logits - dispreferred_logits).detach().mean()

def train_epoch(optimizer: torch.optim.Optimizer, scaler:torch.amp.GradScaler, dataloader, loop: Optional[tqdm.tqdm] = None):
    average_reward_margin = []
    losses = []
    step_counter = 1
    optimizer.zero_grad()
    for batch in dataloader:
        with torch.amp.autocast():
          chosen_log_probs_train, chosen_log_probs_reference, rejected_log_probs_train, rejected_log_probs_reference, rpo_loss = dpo_forward(batch, perform_rpo = True)
          batch_loss, reward_margin = dpo_loss(chosen_log_probs_train, chosen_log_probs_reference, rejected_log_probs_train, rejected_log_probs_reference, BETA, rpo_loss, ALPHA)
        scaler.scale(batch_loss/ACC_STEPS).backward()
        scaler.step(optimizer)
        scaler.update()
        if step_counter % ACC_STEPS ==0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        losses.append(
            scaler.scale(batch_loss).detach().item()
        )
        average_reward_margin.append(
            reward_margin.item()
        )
        step_counter += 1
        if loop:
            loop.set_postfix(loss = losses[-1], rewards = average_reward_margin[-1])
    return np.mean(losses), np.mean(average_reward_margin)


optimizer = torch.optim.AdamW(model_tr.parameters(), weight_decay = .05, lr = 1e-5)
scaler = torch.amp.GradScaler()
TRAIN_LOSS = []
TRAIN_REWARD = []
loop = tqdm.tqdm(range(1, NUM_EPOCHS+1), desc=f"Epoch 1/{NUM_EPOCHS}")

for epoch in loop:
    loop.set_description(f"{epoch}/{NUM_EPOCHS}")
    losses, rewards = train_epoch(optimizer, scaler, dataloader_tr, loop)

    if TRAIN_LOSS==[]:
        best_model = True
    else:
        if np.min(TRAIN_LOSS)>losses and np.max(TRAIN_REWARD)<rewards:
            best_model = True
        else:
            best_model = False

    TRAIN_LOSS.append(losses)
    TRAIN_REWARD.append(rewards)

    if best_model:
        model_tr.save_pretrained("./best_model/dpo_model")