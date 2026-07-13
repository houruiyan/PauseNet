"""Prediction helpers for PauseNet."""

from __future__ import annotations

import torch


@torch.no_grad()
def predict_batch(model, sequence_codes, device="cuda:0"):
    """Return profile probabilities and predicted log1p counts for one batch."""

    model.eval()
    sequence_codes = sequence_codes.to(device)
    profile_logits, predicted_log1p_counts = model(sequence_codes)
    profile_probabilities = torch.softmax(profile_logits.float(), dim=-1)
    return profile_probabilities.cpu(), predicted_log1p_counts.float().cpu()
