"""Density-based OOD reward penalty shared by mbpo_based/ and mobile_based/.

Both world models (mbpo_based/transition_model.py: TransitionModel and
mobile_based/dynamics/ensemble_dynamics.py: EnsembleDynamics) score imagined
(state, action) pairs with a pre-trained density model and subtract a penalty
from the predicted reward. The penalty computation lives here so the two
algorithms use exactly the same code.
"""
import numpy as np
import torch

PENALTY_TYPES = ("linear", "tanh", "tanh_reg", "tanh_shuffled", "flat")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class DensityPenalty:
    """Computes the density (OOD) penalty for imagined transitions.

    Args:
        classifier: dict returned by the density model loaders, with keys
            'model' (exposes score_samples(x, device)) and 'thr', and optional
            'name', 'mean', 'std'.
        device: device passed to the density model's score_samples.
    """

    def __init__(self, classifier, device):
        self.model = classifier['model']
        self.thr = classifier['thr']
        self.name = classifier['name'] if 'name' in classifier else None
        self.mean = classifier['mean'] if 'mean' in classifier else None
        self.std = classifier['std'] if 'std' in classifier else None
        self.device = device

    def __call__(self, state, action, reward=None, type="linear", alpha=0.1):
        """Return (weight, raw_log_probs) as numpy arrays.

        state/action/reward may be numpy arrays or torch tensors. weight is the
        per-sample penalty; raw_log_probs is the density model's log-likelihood
        before normalization by the stored training mean/std.
        """
        if self.name is None:
            model_input = np.concatenate([_to_numpy(state), _to_numpy(action)], axis=1)
        else:
            action_t = _to_tensor(action, self.device)
            reward_t = _to_tensor(reward, action_t.device)
            model_input = torch.cat([action_t.squeeze(1), reward_t.view(-1, 1)], dim=1)

        log_probs = _to_numpy(self.model.score_samples(model_input, self.device))
        raw_log_probs = log_probs.copy()
        if self.mean is not None and self.std is not None:
            log_probs = (log_probs - self.mean) / self.std

        if type == "linear":
            log_weight = self.thr - log_probs  # high means more likely to be OOD
            q1, q3 = np.percentile(log_weight, [25, 75])
            upper_bound = q3 + 1.5 * (q3 - q1)
            weight = np.clip(log_weight, a_min=0, a_max=upper_bound)
        elif type == "tanh":
            log_weight = np.tanh(0.1 * (-log_probs + self.thr))
            weight = np.clip(log_weight, a_min=0, a_max=None)
        elif type == "tanh_reg":
            weight = np.tanh(0.1 * (-log_probs + self.thr))
        elif type == "tanh_shuffled":
            log_weight = np.tanh(0.1 * (-log_probs + self.thr))
            weight = np.clip(log_weight, a_min=0, a_max=None)
            weight = np.random.permutation(weight)  # shuffle penalties
        elif type == "flat":
            weight = np.ones_like(log_probs)
        else:
            raise ValueError(f"Unknown penalty type '{type}'. Expected one of {PENALTY_TYPES}.")

        return weight, raw_log_probs
