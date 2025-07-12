import torch

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