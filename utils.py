import torch
import torch.nn.functional as F

def selective_log_softmax(logits, index, loss_mask):
    """
    Main algorithm taken from huggingface trl
    """
    selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    per_token_logps = selected_logits - logsumexp_values
    per_token_logps = per_token_logps.masked_fill(~loss_mask,0.)
    adjusted_logps = torch.roll(per_token_logps, shifts=1, dims=1)
    return adjusted_logps

def generate_weight(logits:torch.FloatTensor, loss_mask: torch.BoolTensor, strategy:str|None = "sample"):
    logits = logits.detach()
    log_probs = F.log_softmax(logits, dim=-1)
    match strategy:
        case "greedy":
            alignment = torch.amax(log_probs, dim = -1)
        case "sample":
            alignment = torch.logsumexp(2 * log_probs, dim=-1)
        case _:
            alignment = 0
    log_probs = log_probs - alignment
    log_probs = log_probs.masked_fill(~loss_mask, .0).sum(dim=-1)
    num_probs = loss_mask.int().sum(dim=1)
    weight = torch.exp(log_probs / num_probs)
    return weight