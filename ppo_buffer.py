
import numpy as np
import torch as T


class PPORolloutBuffer:

    def __init__(
        self,
        n_steps:    int,
        obs_dim:    int,
        act_dim:    int,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        device:     str   = 'cpu',
    ):
        self.n_steps    = n_steps
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device
        self._ptr       = 0
        self._full      = False

        self.states    = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(n_steps,             dtype=np.float32)
        self.rewards   = np.zeros(n_steps,             dtype=np.float32)
        self.values    = np.zeros(n_steps,             dtype=np.float32)
        self.dones     = np.zeros(n_steps,             dtype=np.float32)

        self.advantages = None
        self.returns    = None

    # ── Collection ────────────────────────────────────────────────────────────
    def store(self, state, action, log_prob, reward, value, done):
        idx = self._ptr
        self.states[idx]    = state
        self.actions[idx]   = action
        self.log_probs[idx] = log_prob
        self.rewards[idx]   = reward
        self.values[idx]    = value
        self.dones[idx]     = float(done)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def reset(self):
        self._ptr       = 0
        self._full      = False
        self.advantages = None
        self.returns    = None

    # ── GAE computation ───────────────────────────────────────────────────────
    def compute_advantages(self, last_value: float):
        """
        Generalised Advantage Estimate (Schulman et al. 2015, Eq.11).

        The shaped reward (tracking + cooperation + anti-sat + smoothness
        + participation) is stored in self.rewards. GAE is computed on
        these shaped rewards — the cooperation signal is automatically
        carried through into the advantages.
        """
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta    = (self.rewards[t]
                        + self.gamma * next_value * next_non_terminal
                        - self.values[t])
            last_gae = (delta
                        + self.gamma * self.gae_lambda
                        * next_non_terminal * last_gae)
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns    = advantages + self.values

    # ── Mini-batch generator ──────────────────────────────────────────────────
    def get_minibatches(self, batch_size: int):
        """
        Yield random mini-batches with normalised advantages.
        Cooperation signal is preserved in relative advantage ordering.
        """
        assert self.advantages is not None, "Call compute_advantages() first"

        adv  = self.advantages.copy()
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                T.tensor(self.states[idx],    dtype=T.float32).to(self.device),
                T.tensor(self.actions[idx],   dtype=T.float32).to(self.device),
                T.tensor(self.log_probs[idx], dtype=T.float32).to(self.device),
                T.tensor(adv[idx],            dtype=T.float32).to(self.device),
                T.tensor(self.returns[idx],   dtype=T.float32).to(self.device),
            )