"""
ppo_network.py  (CORRECTED — Patch C)
======================================
Actor-Critic network with per-dimension log_std and tanh squashing correction.

CHANGES vs original
--------------------
1. log_std is a VECTOR (n_actions,) with ASYMMETRIC initialisation:
     Hc (dim 0): log(0.8) = -0.22  — wider exploration on heater
     u1 (dim 1): log(0.5) = -0.69  — moderate exploration on coolant
   Original used scalar -0.693 for both → collapsed identically

2. log_std clamp widened from [-3, 0] to [-4, +1]:
     Upper bound +1 allows std to expand if needed (std=e^1≈2.7)
     Lower bound -4 prevents total collapse (std=e^-4≈0.018, still usable)
     Original [-3, 0] caused log_std[0] to hit -3 floor and freeze

3. Actor head outputs PRE-TANH values; tanh applied explicitly in forward().
   This is required for the correct log_prob calculation (item 4).

4. Tanh squashing log_prob correction applied:
     log π(a|s) = Σ_i [ log N(atanh(a_i); μ_i, σ_i) - log(1 - a_i²) ]
   Without this correction, the policy believes boundary actions (a=±1)
   have finite probability → drives log_std → -∞ → dim 0 saturates at Hc=8mA.

5. Per-dimension entropy accessible for Patch D (entropy floor enforcement).
"""

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone Actor-Critic for PPO with continuous squashed actions.
    Drop-in replacement — same external interface as original.
    """

    def __init__(self, input_dims: int, n_actions: int,
                 n_hidden: int = 256, n_layers: int = 2,
                 chkpt_dir: str = '', name: str = 'ppo_ac'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '.pt')
        self.n_actions = n_actions

        # ── Shared backbone (unchanged) ───────────────────────────────────
        layers = []
        in_dim = input_dims
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, n_hidden),
                       nn.LayerNorm(n_hidden),
                       nn.Tanh()]
            in_dim = n_hidden
        self.backbone = nn.Sequential(*layers)

        # ── Actor head: outputs PRE-TANH values (no activation) ──────────
        self.actor_head = nn.Linear(n_hidden, n_actions)

        # ── Critic head ───────────────────────────────────────────────────
        self.critic_head = nn.Linear(n_hidden, 1)

        # ── Per-dimension log_std with asymmetric initialisation ──────────
        # Hc (dim 0): log(0.8) ≈ -0.22  → std ≈ 0.80  (was collapsed to e^-3≈0.05)
        # u1 (dim 1): log(0.5) ≈ -0.69  → std ≈ 0.50  (was same, now independent)
        _init = T.tensor([-0.22, -0.69] if n_actions == 2
                         else [-0.22] * n_actions)
        self.log_std = nn.Parameter(_init)

        self._init_weights()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller gain for output heads (standard PPO)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: T.Tensor):
        """
        Returns
        -------
        mu_squashed  : action mean in (-1,1)  shape (B, n_actions)
        value        : critic output           shape (B, 1)
        std          : per-dim std (positive)  shape (n_actions,)
        pre_tanh_mu  : pre-tanh mean           shape (B, n_actions)
        """
        feat        = self.backbone(state)
        pre_tanh_mu = self.actor_head(feat)
        mu_squashed = T.tanh(pre_tanh_mu)
        value       = self.critic_head(feat)
        # Wider clamp: [-4, +1] instead of original [-3, 0]
        std         = T.exp(T.clamp(self.log_std, -4.0, 1.0))
        return mu_squashed, value, std, pre_tanh_mu

    def get_action_and_value(self, state: T.Tensor,
                             action: T.Tensor = None):
        """
        Sample or evaluate actions with squashing correction.

        Parameters
        ----------
        state  : (B, input_dims) or (input_dims,)
        action : if provided, must be in (-1, 1)  (post-tanh space)

        Returns
        -------
        action_squashed : (B, n_actions)  in (-1, 1)
        log_prob        : (B,)            corrected log π(a|s)
        entropy         : scalar          mean Gaussian entropy (pre-squash)
        value           : (B, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        mu_sq, value, std, pre_tanh_mu = self.forward(state)

        # Distribution is over pre-tanh space
        dist = T.distributions.Normal(pre_tanh_mu, std)

        if action is None:
            pre_tanh_sample = dist.rsample()
            action_squashed = T.tanh(pre_tanh_sample)
        else:
            # Recover pre-tanh action via atanh; clamp avoids ±∞
            a_clamped       = T.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh_sample = T.atanh(a_clamped)
            action_squashed = a_clamped

        # ── Tanh-corrected log probability ────────────────────────────────
        # log π(a|s) = log N(atanh(a); μ, σ) - Σ log(1 - a²)
        log_prob_gaussian = dist.log_prob(pre_tanh_sample)             # (B, n_actions)
        squash_corr       = T.log(1.0 - action_squashed.pow(2) + 1e-6) # (B, n_actions)
        log_prob          = (log_prob_gaussian - squash_corr).sum(dim=-1)  # (B,)

        # Entropy: Gaussian entropy (pre-squash, per PPO convention)
        # Mean over batch and actions → scalar
        entropy = dist.entropy().mean()

        return action_squashed, log_prob, entropy, value

    def get_per_dim_entropy(self, state: T.Tensor) -> T.Tensor:
        """
        Returns per-dimension entropy (n_actions,) for monitoring.
        Used by entropy floor in Patch D.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        _, _, std, pre_tanh_mu = self.forward(state)
        dist = T.distributions.Normal(pre_tanh_mu, std)
        return dist.entropy().mean(dim=0)  # (n_actions,)

    def get_value(self, state: T.Tensor) -> T.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        feat = self.backbone(state)
        return self.critic_head(feat)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print(f'...saved checkpoint: {self.checkpoint_file}')

    def load_checkpoint(self):
        self.load_state_dict(
            T.load(self.checkpoint_file, map_location=self.device)
        )
        print(f'...loaded checkpoint: {self.checkpoint_file}')