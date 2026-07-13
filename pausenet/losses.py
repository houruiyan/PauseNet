"""Training losses for PauseNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPSILON = 1e-8


def multinomial_profile_nll(
    profile_logits: torch.Tensor,
    observed_profiles: torch.Tensor,
    profile_masks: torch.Tensor,
) -> torch.Tensor:
    """Multinomial negative log likelihood for profile shape."""

    totals = observed_profiles.sum(dim=-1)
    valid = profile_masks.bool() & (totals > 0)
    if not valid.any():
        return profile_logits.sum() * 0.0
    log_probs = F.log_softmax(profile_logits[valid], dim=-1)
    per_sample = -torch.sum(observed_profiles[valid] * log_probs, dim=-1)
    return per_sample.mean()


def log1p_count_mse(
    predicted_log1p_counts: torch.Tensor,
    observed_counts: torch.Tensor,
) -> torch.Tensor:
    target = torch.log1p(observed_counts.float())
    return F.mse_loss(predicted_log1p_counts.float(), target)


def differentiable_js_divergence(
    target_prob: torch.Tensor,
    predicted_prob: torch.Tensor,
) -> torch.Tensor:
    midpoint = 0.5 * (target_prob + predicted_prob)
    left = torch.sum(
        torch.where(
            target_prob > 0,
            target_prob * torch.log2((target_prob + EPSILON) / (midpoint + EPSILON)),
            torch.zeros_like(target_prob),
        ),
        dim=-1,
    )
    right = torch.sum(
        torch.where(
            predicted_prob > 0,
            predicted_prob
            * torch.log2((predicted_prob + EPSILON) / (midpoint + EPSILON)),
            torch.zeros_like(predicted_prob),
        ),
        dim=-1,
    )
    return 0.5 * (left + right)


def binned_probabilities(values: torch.Tensor, resolution: int) -> torch.Tensor:
    if resolution == 1:
        return values
    usable = values.shape[1] // resolution * resolution
    return values[:, :usable].reshape(values.shape[0], usable // resolution, resolution).sum(dim=-1)


def multiresolution_profile_jsd_loss(
    profile_logits: torch.Tensor,
    observed_profiles: torch.Tensor,
    profile_masks: torch.Tensor,
    resolutions: tuple[int, ...] = (1, 5, 10, 20),
    weights: tuple[float, ...] = (0.55, 0.15, 0.15, 0.15),
) -> torch.Tensor:
    """Differentiable profile-shape loss used by earlier PauseNet experiments."""

    totals = observed_profiles.sum(dim=-1)
    valid = profile_masks.bool() & (totals > 0)
    if not valid.any():
        return profile_logits.sum() * 0.0
    target = observed_profiles[valid].float() / totals[valid].float().unsqueeze(1).clamp_min(EPSILON)
    predicted = torch.softmax(profile_logits[valid].float(), dim=-1)
    loss = profile_logits.sum() * 0.0
    for resolution, weight in zip(resolutions, weights):
        loss = loss + weight * differentiable_js_divergence(
            binned_probabilities(target, resolution),
            binned_probabilities(predicted, resolution),
        ).mean()
    return loss


def pausenet_loss(
    profile_logits: torch.Tensor,
    predicted_log1p_counts: torch.Tensor,
    observed_profiles: torch.Tensor,
    observed_counts: torch.Tensor,
    profile_masks: torch.Tensor,
    counts_weight: float = 100.0,
    profile_loss: str = "mnll",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined profile/count loss.

    The ProCapNet-style default is:

        L = MNLL(profile) + counts_weight * MSE(log1p counts)
    """

    if profile_loss == "mnll":
        profile = multinomial_profile_nll(profile_logits, observed_profiles, profile_masks)
    elif profile_loss == "multiresolution_jsd":
        profile = multiresolution_profile_jsd_loss(profile_logits, observed_profiles, profile_masks)
    else:
        raise ValueError(f"Unknown profile_loss: {profile_loss}")
    count = log1p_count_mse(predicted_log1p_counts, observed_counts)
    total = profile + counts_weight * count
    return total, profile, count
