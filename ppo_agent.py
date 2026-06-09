
import numpy as np
import torch as T
import torch.nn.functional as F

from ppo_network import ActorCriticNetwork
from ppo_buffer  import PPORolloutBuffer


# ── Physical actuator bounds ──────────────────────────────────────────────────
HC_LOW  =  8.0;   HC_HIGH = 20.0    # mA
U1_LOW  =  0.0;   U1_HIGH =  0.7    # L/min
HC_RANGE = HC_HIGH - HC_LOW
U1_RANGE = U1_HIGH - U1_LOW


def raw_to_physical(raw_action: np.ndarray) -> tuple[float, float]:
    """
    Map raw action in (−1, 1)² → physical actuator values.

    Returns
    -------
    (Hc [mA], u1 [L/min])
    """
    a  = np.clip(raw_action, -1.0, 1.0)
    Hc = HC_LOW  + (a[0] + 1.0) / 2.0 * HC_RANGE
    u1 = U1_LOW  + (a[1] + 1.0) / 2.0 * U1_RANGE
    Hc = float(np.clip(Hc, HC_LOW,  HC_HIGH))
    u1 = float(np.clip(u1, U1_LOW,  U1_HIGH))
    return Hc, u1


class PPOAgent:
    """
    PPO agent for continuous-action control of the batch reactor.

    Parameters
    ----------
    obs_dim        : state dimension (5 for UKF version)
    act_dim        : action dimension (2: Hc, u1)
    n_steps        : rollout length before each update (2048)
    n_epochs       : PPO update epochs per rollout (10)
    batch_size     : mini-batch size (64)
    gamma          : discount factor (0.99)
    gae_lambda     : GAE lambda (0.95)
    clip_eps       : PPO clipping epsilon (0.2)
    lr             : learning rate (3e-4)
    ent_coef       : entropy bonus coefficient — RAISED to 0.05 (RC3 fix)
    vf_coef        : value-function loss coefficient (0.5)
    max_grad_norm  : gradient clipping norm (0.5)
    n_hidden       : hidden units per layer (256)
    n_layers       : number of hidden layers (2)
    chkpt_dir      : directory for checkpoint files
    name           : checkpoint file prefix
    """

    def __init__(
        self,
        obs_dim:       int   = 5,
        act_dim:       int   = 2,
        n_steps:       int   = 2048,
        n_epochs:      int   = 10,
        batch_size:    int   = 64,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_eps:      float = 0.2,
        lr:            float = 3e-4,
        ent_coef:      float = 0.05,   # RC3 fix: raised from 0.01
        vf_coef:       float = 0.5,
        max_grad_norm: float = 0.5,
        n_hidden:      int   = 256,
        n_layers:      int   = 2,
        chkpt_dir:     str   = '',
        name:          str   = 'ppo_reactor',
    ):
        self.obs_dim       = obs_dim
        self.act_dim       = act_dim
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm

        self.network = ActorCriticNetwork(
            input_dims=obs_dim,
            n_actions=act_dim,
            n_hidden=n_hidden,
            n_layers=n_layers,
            chkpt_dir=chkpt_dir,
            name=name,
        )
        self.device = self.network.device

        self.optimizer = T.optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5
        )

        self.buffer = PPORolloutBuffer(
            n_steps=n_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=str(self.device),
        )

        self.train_step   = 0
        self.loss_history = []

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def log_std_value(self) -> np.ndarray:
        """Current log_std parameter value (for entropy monitoring)."""
        return self.network.log_std.detach().cpu().numpy().copy()

    @property
    def std_value(self) -> np.ndarray:
        """Current std value (exp of clamped log_std)."""
        import torch as T_
        from ppo_network import LOG_STD_MIN, LOG_STD_MAX
        clamped = T_.clamp(self.network.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return T_.exp(clamped).detach().cpu().numpy().copy()

    # ── Interaction ───────────────────────────────────────────────────────────
    @T.no_grad()
    def choose_action(self, obs: np.ndarray):
        """
        Sample action from the current stochastic policy.

        Returns
        -------
        raw_action : np.ndarray (act_dim,)  in (−1, 1)
        log_prob   : float
        value      : float  V(s)
        Hc         : float  physical [mA]
        u1         : float  physical [L/min]
        """
        self.network.eval()
        state_t = T.tensor(obs, dtype=T.float32).to(self.device)
        action_t, log_prob_t, _, value_t = \
            self.network.get_action_and_value(state_t)
        self.network.train()

        raw_action = action_t.squeeze(0).cpu().numpy()
        log_prob   = log_prob_t.squeeze(0).item()
        value      = value_t.squeeze(0).item()
        Hc, u1     = raw_to_physical(raw_action)
        return raw_action, log_prob, value, Hc, u1

    @T.no_grad()
    def deterministic_action(self, obs: np.ndarray):
        """
        Return the deterministic mean action (no sampling).
        Used during validation / plotting.

        Returns
        -------
        Hc, u1 : float  physical actuator values
        """
        self.network.eval()
        state_t = T.tensor(obs, dtype=T.float32).to(self.device)
        mu, _, _, _= self.network(
            state_t.unsqueeze(0) if state_t.dim() == 1 else state_t
        )
        raw = mu.squeeze(0).cpu().numpy()
        return raw_to_physical(raw)

    # ── Storage ───────────────────────────────────────────────────────────────
    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.store(state, action, log_prob, reward, value, done)

    # ── Learning ──────────────────────────────────────────────────────────────
    def learn(self, last_obs: np.ndarray, last_done: bool) -> dict:
        """
        Run PPO update on the collected rollout.

        Parameters
        ----------
        last_obs  : final observation after the rollout (for bootstrap)
        last_done : whether the episode ended at the last rollout step

        Returns
        -------
        metrics : dict with keys policy_loss, value_loss, entropy, approx_kl
        """
        if last_done:
            last_value = 0.0
        else:
            with T.no_grad():
                state_t    = T.tensor(last_obs, dtype=T.float32).to(self.device)
                last_value = self.network.get_value(state_t).squeeze().item()

        self.buffer.compute_advantages(last_value)

        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy  = 0.0
        total_kl       = 0.0
        n_updates      = 0

        for _ in range(self.n_epochs):
            for (states_b, actions_b, old_log_probs_b,
                 advantages_b, returns_b) in \
                    self.buffer.get_minibatches(self.batch_size):

                _, new_log_probs, entropy, values_new = \
                    self.network.get_action_and_value(states_b, actions_b)
                values_new = values_new.squeeze(-1)

                log_ratio  = new_log_probs - old_log_probs_b
                ratio      = T.exp(log_ratio)

                with T.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                pg_loss1 = -advantages_b * ratio
                pg_loss2 = -advantages_b * T.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                )
                pol_loss = T.max(pg_loss1, pg_loss2).mean()
                val_loss = F.mse_loss(values_new, returns_b)

                loss = (pol_loss
                        + self.vf_coef  * val_loss
                        - self.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_pol_loss += pol_loss.item()
                total_val_loss += val_loss.item()
                total_entropy  += entropy.item()
                total_kl       += approx_kl
                n_updates      += 1

        self.buffer.reset()
        self.train_step += 1

        metrics = dict(
            policy_loss = total_pol_loss / max(n_updates, 1),
            value_loss  = total_val_loss / max(n_updates, 1),
            entropy     = total_entropy  / max(n_updates, 1),
            approx_kl   = total_kl       / max(n_updates, 1),
        )
        self.loss_history.append(metrics)
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        self.network.save_checkpoint()

    def load(self):
        self.network.load_checkpoint()