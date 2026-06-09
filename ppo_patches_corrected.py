
import numpy as np

# Physical constants needed for reward (copy from env or import)
_MAT      = 80.0    # Maximum Allowable Temperature [°C]
_T_SCALE  = 5.0     # tracking performance scale [°C]
_HC_LOW   = 8.0     # heater lower bound [mA]
_HC_HIGH  = 20.0    # heater upper bound [mA]
_U1_LOW   = 0.0     # coolant lower bound [L/min]
_U1_HIGH  = 0.7     # coolant upper bound [L/min]

# MAT soft barrier parameters
_MAT_MARGIN      = 10.0    # °C — barrier activates below MAT by this margin
_MAT_BARRIER_K   = 2.0     # steepness of log-barrier
_MAT_TERMINAL    = -500.0  # terminal penalty on actual MAT violation (not -5000)

# Cooperation reward parameters
_COOP_ALPHA   = 0.3   # weight of cooperation term in total reward
_COOP_GAMMA   = 0.2   # weight of UKF confidence term
_SMOOTH_BETA  = 0.15  # weight of smoothness penalty

# UKF confidence scale — P_trace at episode start ≈ 5.2e-6+5.2e-4+0.52+0.52 ≈ 1.04
_UKF_P_SCALE  = 2.0   # normalisation for trace(P)


def compute_reward(
    e: float,           # tracking error: Tr - Tr_ref  [°C]
    Tr: float,          # current reactor temperature  [°C]
    Hc: float,          # heater action applied        [mA]
    u1: float,          # coolant action applied       [L/min]
    Hc_prev: float,     # heater action previous step  [mA]
    u1_prev: float,     # coolant action previous step [L/min]
    ukf_P_trace: float, # trace of UKF covariance P    [mixed units]
    M_hat: float,       # UKF monomer estimate         [mol/L]
    M_MAX: float = 0.7034,
    done_mat: bool = False,
) -> tuple:
    """
    Corrected reward function with 5 components:

      r_track    : primary tracking objective — exponential proximity
      r_coop     : actuator cooperation — heater and coolant both active
      r_smooth   : smoothness — penalises large action deltas
      r_barrier  : MAT soft log-barrier — activates in [MAT-margin, MAT]
      r_ukf      : UKF confidence bonus — rewards low-uncertainty states
      r_terminal : hard terminal penalty if MAT actually violated

    Returns
    -------
    total_reward : float
    components   : dict  (for logging/debugging)
    done         : bool  (True if MAT violated)
    """

    # ── 1. Tracking reward (primary objective) ─────────────────────────────
    r_track = -abs(e)   # keep sign-consistent with existing learning curve

    # ── 2. Actuator cooperation reward ─────────────────────────────────────
    # Normalise each actuator to [0, 1] based on its physical range
    hc_norm = (Hc - _HC_LOW) / (_HC_HIGH - _HC_LOW)   # 0=min, 1=max
    u1_norm = u1 / _U1_HIGH                            # 0=off, 1=max

    # Cooperation: reward simultaneous moderate use of BOTH actuators
    # Maximum when hc_norm ≈ 0.5 AND u1_norm ≈ 0.5 (both at midpoint)
    # Zero when either actuator is at its extreme (0 or 1)
    coop_hc = 4.0 * hc_norm * (1.0 - hc_norm)   # parabola peak=1 at hc_norm=0.5
    coop_u1 = 4.0 * u1_norm * (1.0 - u1_norm)   # parabola peak=1 at u1_norm=0.5
    r_coop  = _COOP_ALPHA * (coop_hc * coop_u1)  # product → both must be active

    # ── 3. Smoothness penalty ───────────────────────────────────────────────
    delta_hc = (Hc - Hc_prev) / (_HC_HIGH - _HC_LOW)  # normalised delta
    delta_u1 = (u1 - u1_prev) / _U1_HIGH
    r_smooth = -_SMOOTH_BETA * (delta_hc**2 + delta_u1**2)

    # ── 4. MAT soft log-barrier ─────────────────────────────────────────────
    # Activates when Tr enters [MAT - margin, MAT]; zero below that zone
    margin_dist = _MAT - Tr   # positive = safe, 0 = at MAT
    if margin_dist <= 0.0:
        # Actual violation
        r_barrier  = _MAT_TERMINAL
        done       = True
    elif margin_dist < _MAT_MARGIN:
        # Inside the warning zone: log-barrier grows as Tr → MAT
        # barrier = -k * log(margin_dist / MAT_MARGIN) ∈ [0, +∞)
        r_barrier = -_MAT_BARRIER_K * np.log(margin_dist / _MAT_MARGIN + 1e-8)
        done      = False
    else:
        r_barrier = 0.0
        done      = False

    # ── 5. UKF confidence reward ────────────────────────────────────────────
    # Low trace(P) → estimator is confident → small bonus
    # Normalise: at episode start trace≈1.04, decreases to ~0.01
    ukf_conf = np.exp(-ukf_P_trace / _UKF_P_SCALE)   # ∈ (0, 1]
    r_ukf    = _COOP_GAMMA * ukf_conf

    # ── 6. Chemical state awareness (monomer depletion) ────────────────────
    # Small bonus when monomer is within healthy range (not depleted)
    M_norm    = float(np.clip(M_hat / M_MAX, 0.0, 1.0))
    r_chem    = 0.05 * M_norm   # tiny — keeps chemical state in obs relevant

    # ── Total ───────────────────────────────────────────────────────────────
    if done_mat or done:
        total = r_track + r_barrier  # drop bonuses on terminal step
        done  = True
    else:
        total = r_track + r_coop + r_smooth + r_barrier + r_ukf + r_chem

    components = dict(
        r_track=r_track, r_coop=r_coop, r_smooth=r_smooth,
        r_barrier=r_barrier, r_ukf=r_ukf, r_chem=r_chem,
        total=total
    )
    return float(total), components, done

# END PATCH A ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH B — observation_normalization (full replacement)
# Drop into train_ppo.py and analyze_ppo.py, replacing build_obs().
# CRITICAL FIX: I_MAX corrected from 4.5e-3 to 4.5e-5 (actual reset value).
# ═══════════════════════════════════════════════════════════════════════════════

# Correct physical bounds — MUST match actual env reset values
_I_MAX_CORR  = 4.5e-5   # ← CORRECTED: was 4.5e-3; reset uses 4.5e-5 mol/L
_M_MAX_CORR  = 0.7034   # mol/L  (M0 at episode start)
_TR_MIN_CORR = 40.0     # °C    (below min trajectory to prevent clipping)
_TR_MAX_CORR = 80.0     # °C    (= MAT)
_TJ_MIN_CORR = 27.0     # °C    (Tc = coolant supply temperature)
_TJ_MAX_CORR = 70.0     # °C
_E_SCALE_CORR = 20.0    # °C    (tracking error normalisation)


def build_obs_corrected(I_hat: float, M_hat: float,
                        Tr: float, Tj: float, e: float) -> np.ndarray:
    """
    5D normalised observation — CORRECTED bounds.

    obs[0] = I_hat / I_MAX_CORR          ← was /4.5e-3, now /4.5e-5 → full [0,1] range
    obs[1] = M_hat / M_MAX               ∈ [0, 1]
    obs[2] = (Tr - TR_MIN) / (TR_MAX - TR_MIN)   ∈ [0, 1]
    obs[3] = (Tj - TJ_MIN) / (TJ_MAX - TJ_MIN)   ∈ [0, 1]
    obs[4] = clip(e / E_SCALE, -1, 1)             ∈ [-1, 1]

    UKF outputs (I_hat, M_hat) now span nearly the full [0,1] range,
    giving the policy gradient signal on obs[0] throughout the episode.
    """
    return np.array([
        float(np.clip(I_hat / _I_MAX_CORR,                              0., 1.)),
        float(np.clip(M_hat / _M_MAX_CORR,                              0., 1.)),
        float(np.clip((Tr - _TR_MIN_CORR) / (_TR_MAX_CORR - _TR_MIN_CORR), 0., 1.)),
        float(np.clip((Tj - _TJ_MIN_CORR) / (_TJ_MAX_CORR - _TJ_MIN_CORR), 0., 1.)),
        float(np.clip(e / _E_SCALE_CORR,                               -1., 1.)),
    ], dtype=np.float32)

# END PATCH B ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH C — actor_network (full replacement of ppo_network.py)
# Key changes:
#   - log_std is now a VECTOR of shape (n_actions,) with INDEPENDENT init
#   - Hc dim (0) init = log(0.8) = -0.22  (wider — was collapsed to -3)
#   - u1 dim (1) init = log(0.5) = -0.69
#   - log_std clamp widened to [-4, +1] — allows expansion when needed
#   - Tanh squashing correction applied to log_prob (prevents bound saturation)
# ═══════════════════════════════════════════════════════════════════════════════

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetworkCorrected(nn.Module):
    """
    Corrected Actor-Critic network.

    CHANGES vs original:
      1. log_std per-dimension with asymmetric init (Hc wider than u1)
      2. Wider clamp [-4, +1] prevents log_std from hitting floor
      3. Tanh-squashed log_prob correction: log π = Σ[log N(a_pre) - log(1-tanh²)]
         This is critical — without it the policy believes boundary actions have
         higher probability than they do, which drives saturation.
      4. Actor head outputs pre-tanh value; tanh applied explicitly so we can
         compute the log-prob correction before squashing.
    """

    def __init__(self, input_dims: int, n_actions: int,
                 n_hidden: int = 256, n_layers: int = 2,
                 chkpt_dir: str = '', name: str = 'ppo_ac'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '.pt')
        self.n_actions = n_actions

        # ── Shared backbone ───────────────────────────────────────────────
        layers = []
        in_dim = input_dims
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, n_hidden),
                       nn.LayerNorm(n_hidden),
                       nn.Tanh()]
            in_dim = n_hidden
        self.backbone = nn.Sequential(*layers)

        # ── Actor head: outputs PRE-TANH values ───────────────────────────
        # No activation here — tanh is applied explicitly in forward()
        self.actor_head = nn.Linear(n_hidden, n_actions)

        # ── Critic head ───────────────────────────────────────────────────
        self.critic_head = nn.Linear(n_hidden, 1)

        # ── Independent log_std per actuator dimension ────────────────────
        # Hc (dim 0): init at log(0.8) ≈ -0.22  — WIDE, was collapsed to -3
        # u1 (dim 1): init at log(0.5) ≈ -0.69  — moderate
        # Clamp to [-4, +1] — allows controlled expansion above 0
        _init_log_std = T.tensor([-0.22, -0.69])   # [Hc, u1]
        self.log_std = nn.Parameter(_init_log_std)

        self._init_weights()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        import numpy as np_
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np_.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: T.Tensor):
        """
        Returns
        -------
        mu_squashed : action mean in (-1,1) via tanh
        value       : scalar critic output
        std         : per-actuator std (positive)
        pre_tanh_mu : pre-tanh mean (needed for log_prob correction)
        """
        feat         = self.backbone(state)
        pre_tanh_mu  = self.actor_head(feat)
        mu_squashed  = T.tanh(pre_tanh_mu)
        value        = self.critic_head(feat)
        std          = T.exp(T.clamp(self.log_std, -4.0, 1.0))
        return mu_squashed, value, std, pre_tanh_mu

    def get_action_and_value(self, state: T.Tensor, action: T.Tensor = None):
        """
        Sample or evaluate actions with tanh-squashing log_prob correction.

        The correction term: -Σ log(1 - tanh²(x)) where x is the pre-tanh sample.
        Without this, the policy thinks boundary actions (-1, +1) have finite
        density, which drives log_std → -∞ to concentrate mass at boundaries.

        Returns
        -------
        action_squashed : (B, n_actions) in (-1, 1)
        log_prob        : (B,)  corrected log π(a|s)
        entropy         : ()    Gaussian entropy (pre-squash, per convention)
        value           : (B, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        mu_sq, value, std, pre_tanh_mu = self.forward(state)
        dist = T.distributions.Normal(pre_tanh_mu, std)   # distribution over PRE-TANH

        if action is None:
            # Sample in pre-tanh space, then squash
            pre_tanh_sample = dist.rsample()
            action_squashed = T.tanh(pre_tanh_sample)
        else:
            # Inverse-tanh to recover pre-tanh action for log_prob evaluation
            # Clamp to prevent atanh(±1) = ±∞
            action_clamped  = T.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh_sample = T.atanh(action_clamped)
            action_squashed = action_clamped

        # Log-prob with squashing correction (Eq. from SAC paper, also valid for PPO)
        log_prob_gaussian   = dist.log_prob(pre_tanh_sample)           # (B, n_actions)
        squash_correction   = T.log(1.0 - action_squashed.pow(2) + 1e-6)  # (B, n_actions)
        log_prob            = (log_prob_gaussian - squash_correction).sum(dim=-1)  # (B,)

        # Entropy: use Gaussian entropy (pre-squash) — standard PPO convention
        entropy = dist.entropy().mean()

        return action_squashed, log_prob, entropy, value

    def get_value(self, state: T.Tensor) -> T.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        feat = self.backbone(state)
        return self.critic_head(feat)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print(f'...saved checkpoint: {self.checkpoint_file}')

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
        print(f'...loaded checkpoint: {self.checkpoint_file}')

# END PATCH C ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH D — PPO entropy handling modifications
# Drop into ppo_agent.py learn() method, replacing the loss computation block.
# Key change: ent_coef is now annealed externally and passed in, not fixed.
# Also adds per-dimension entropy logging for diagnosis.
# ═══════════════════════════════════════════════════════════════════════════════

def ppo_loss_with_entropy_floor(
    advantages_b, ratio, values_new, returns_b,
    new_log_probs, old_log_probs_b,
    dist_for_entropy,   # the Normal distribution object (pre-squash)
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,    # annealed externally — see Patch H
    min_entropy_per_dim: float = 0.5,   # nats — floor per actuator dimension
):
    """
    PPO loss with:
      1. Annealed entropy coefficient (ent_coef passed from scheduler)
      2. Per-dimension entropy floor — adds bonus if any dim entropy < floor
         This prevents one actuator from collapsing while the other is fine.

    Usage in ppo_agent.learn():
        loss, pol_loss, val_loss, entropy_val = ppo_loss_with_entropy_floor(
            advantages_b, ratio, values_new, returns_b,
            new_log_probs, old_log_probs_b,
            dist_for_entropy=dist,
            clip_eps=self.clip_eps, vf_coef=self.vf_coef,
            ent_coef=current_ent_coef,
        )
    """
    # Standard PPO clipped objective
    pg_loss1   = -advantages_b * ratio
    pg_loss2   = -advantages_b * T.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pol_loss   = T.max(pg_loss1, pg_loss2).mean()

    # Value loss
    val_loss   = F.mse_loss(values_new, returns_b)

    # Per-dimension entropy (shape: n_actions)
    per_dim_entropy = dist_for_entropy.entropy()   # (B, n_actions)
    mean_per_dim    = per_dim_entropy.mean(dim=0)  # (n_actions,)
    mean_entropy    = per_dim_entropy.mean()       # scalar

    # Entropy floor: add extra bonus for any dimension below the floor
    floor_deficit    = T.clamp(min_entropy_per_dim - mean_per_dim, min=0.0)
    entropy_floor_bonus = floor_deficit.sum() * 2.0   # 2× ent_coef weight

    total_loss = (pol_loss
                  + vf_coef * val_loss
                  - ent_coef * mean_entropy
                  - entropy_floor_bonus)

    return total_loss, pol_loss, val_loss, mean_entropy, mean_per_dim

# END PATCH D ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH E — heater action scaling fix
# Drop into ppo_agent.py, replacing raw_to_physical().
# DIAGNOSIS: With HC_LOW=8 in train_ppo.py but the original code using
# a[0]=-1 → Hc=8 (minimum), the heater never goes above 8 because the policy
# found that minimal heating + high coolant tracks the trajectory.
# FIX: Heater dimension now has asymmetric initialisation (Patch C) AND
# the scaling function is preserved correctly. No range change needed —
# the fix is in the gradient signal, not the mapping.
# This patch adds diagnostic clamping and logging.
# ═══════════════════════════════════════════════════════════════════════════════

_HC_LOW_PHYS  = 8.0    # mA — matches env._Hc_min = 4.0 minimum; PPO uses 8.0
_HC_HIGH_PHYS = 20.0   # mA
_U1_LOW_PHYS  = 0.0    # L/min
_U1_HIGH_PHYS = 0.7    # L/min


def raw_to_physical_corrected(raw_action: np.ndarray,
                               warn_saturation: bool = False) -> tuple:
    """
    Map raw action in (-1, 1)^2 to physical actuator values.

    dim 0 → Hc ∈ [HC_LOW, HC_HIGH] mA
    dim 1 → u1 ∈ [U1_LOW, U1_HIGH] L/min

    Saturation check: warns if |raw| > 0.95 on either dimension,
    which indicates the policy is pressing against the action boundary.

    Returns
    -------
    Hc      : float  heater current [mA]
    u1      : float  coolant flow [L/min]
    sat_hc  : bool   True if Hc dimension is saturated
    sat_u1  : bool   True if u1 dimension is saturated
    """
    a = np.clip(raw_action, -1.0, 1.0)

    Hc = _HC_LOW_PHYS  + (a[0] + 1.0) / 2.0 * (_HC_HIGH_PHYS - _HC_LOW_PHYS)
    u1 = _U1_LOW_PHYS  + (a[1] + 1.0) / 2.0 * (_U1_HIGH_PHYS - _U1_LOW_PHYS)

    Hc = float(np.clip(Hc, _HC_LOW_PHYS,  _HC_HIGH_PHYS))
    u1 = float(np.clip(u1, _U1_LOW_PHYS,  _U1_HIGH_PHYS))

    sat_hc = bool(abs(a[0]) > 0.95)
    sat_u1 = bool(abs(a[1]) > 0.95)

    if warn_saturation:
        if sat_hc:
            print(f"  [SAT] Hc dim saturated: raw={a[0]:.3f} → Hc={Hc:.2f}mA")
        if sat_u1:
            print(f"  [SAT] u1 dim saturated: raw={a[1]:.3f} → u1={u1:.3f}L/min")

    return Hc, u1, sat_hc, sat_u1

# END PATCH E ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH F — coolant oscillation smoothing penalty
# This is already incorporated into Patch A (r_smooth).
# This patch provides the standalone function for use in logging/analysis.
# ═══════════════════════════════════════════════════════════════════════════════

def coolant_oscillation_penalty(u1_history: list,
                                 window: int = 10,
                                 threshold: float = 0.15) -> float:
    """
    Compute oscillation severity for the coolant channel over a recent window.

    Uses total variation (TV) normalised by window length.
    High TV → rapid switching → penalty.

    Parameters
    ----------
    u1_history : recent coolant flow values [L/min]
    window     : number of steps to evaluate
    threshold  : TV above this is considered oscillating

    Returns
    -------
    penalty : float ≤ 0  (negative, to subtract from reward)
    """
    if len(u1_history) < 2:
        return 0.0
    u1_arr = np.array(u1_history[-window:])
    tv     = float(np.sum(np.abs(np.diff(u1_arr)))) / (len(u1_arr) - 1)
    tv_norm = tv / _U1_HIGH_PHYS   # normalise to [0,1] range
    excess  = max(tv_norm - threshold / _U1_HIGH_PHYS, 0.0)
    return -0.5 * excess   # small penalty, does not dominate tracking

# END PATCH F ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH G — MAT soft safety barrier (standalone, also in Patch A)
# Use this if you want the barrier as a separate callable for analysis.
# ═══════════════════════════════════════════════════════════════════════════════

def mat_soft_barrier(Tr: float,
                     MAT: float = 80.0,
                     margin: float = 10.0,
                     k: float = 2.0,
                     terminal_penalty: float = -500.0) -> tuple:
    """
    Soft log-barrier for MAT safety constraint.

    Three zones:
      Tr < MAT - margin         : safe zone, barrier = 0
      MAT - margin ≤ Tr < MAT   : warning zone, barrier = -k*log(d/margin)
      Tr ≥ MAT                  : violation, terminal penalty, done=True

    The log-barrier provides a gradient that grows continuously as the reactor
    approaches MAT, unlike the hard step penalty which gives gradient only at
    the boundary and causes the policy to avoid the entire high-temperature
    regime (preventing heater use at all).

    Parameters
    ----------
    Tr              : reactor temperature [°C]
    MAT             : maximum allowable temperature [°C]
    margin          : barrier activation distance below MAT [°C]
    k               : barrier steepness
    terminal_penalty: reward on actual violation (should be large negative)

    Returns
    -------
    barrier_reward : float
    done           : bool
    zone           : str  ('safe', 'warning', 'violation')
    """
    d = MAT - Tr  # positive = safe distance from MAT

    if d <= 0.0:
        return terminal_penalty, True, 'violation'
    elif d < margin:
        barrier = -k * np.log(d / margin + 1e-8)
        return float(barrier), False, 'warning'
    else:
        return 0.0, False, 'safe'

# END PATCH G ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH H — training loop entropy annealing schedule
# Drop into train_ppo.py. Call get_ent_coef(ep_idx) each episode.
# Pass the result to agent.learn() or override agent.ent_coef before learn().
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyAnnealer:
    """
    Three-phase entropy coefficient schedule:

    Phase 1 [0, warmup_end):     ent_coef = ent_start  (high exploration)
    Phase 2 [warmup_end, decay_end): linear decay from ent_start → ent_min
    Phase 3 [decay_end, N_EPISODES): ent_coef = ent_min  (exploitation)

    Also enforces a per-dimension entropy floor via Patch D.
    If observed entropy drops below floor_nats on any dimension,
    the annealer PAUSES decay (holds current coefficient) until recovery.

    Usage
    -----
    annealer = EntropyAnnealer(N_EPISODES=500)
    # In training loop:
    ent_coef = annealer.step(ep_idx, per_dim_entropy_nats)
    agent.ent_coef = ent_coef
    """

    def __init__(
        self,
        N_EPISODES:    int   = 500,
        ent_start:     float = 0.05,   # high initial entropy encouragement
        ent_min:       float = 0.002,  # minimum — never fully zero
        warmup_frac:   float = 0.15,   # fraction of training to hold ent_start
        decay_frac:    float = 0.60,   # fraction of training to decay over
        floor_nats:    float = 0.3,    # per-dim entropy floor to pause decay
    ):
        self.N          = N_EPISODES
        self.ent_start  = ent_start
        self.ent_min    = ent_min
        self.warmup_end = int(N_EPISODES * warmup_frac)
        self.decay_end  = int(N_EPISODES * (warmup_frac + decay_frac))
        self.floor_nats = floor_nats
        self._current   = ent_start
        self._paused    = False

    def step(self, ep_idx: int, per_dim_entropy: np.ndarray = None) -> float:
        """
        Compute entropy coefficient for episode ep_idx.

        Parameters
        ----------
        ep_idx           : current episode index (0-based)
        per_dim_entropy  : array of shape (n_actions,) with current entropy
                           per actuator dimension [nats]. If any < floor_nats,
                           decay is paused.

        Returns
        -------
        ent_coef : float
        """
        # Check entropy floor — pause decay if any dim collapsed
        if per_dim_entropy is not None:
            self._paused = bool(np.any(per_dim_entropy < self.floor_nats))

        if self._paused:
            # Hold current coefficient; do not decay further
            return float(self._current)

        if ep_idx < self.warmup_end:
            coef = self.ent_start
        elif ep_idx < self.decay_end:
            progress = (ep_idx - self.warmup_end) / max(self.decay_end - self.warmup_end, 1)
            coef     = self.ent_start + progress * (self.ent_min - self.ent_start)
        else:
            coef = self.ent_min

        self._current = float(coef)
        return self._current

    @property
    def is_paused(self) -> bool:
        return self._paused


# END PATCH H ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH I — curriculum training phase logic
# Drop into train_ppo.py. Wrap the rollout collection section.
# Phase 0: restrict trajectory to first CURRICULUM_STEPS steps (easy portion)
# Phase 1: full trajectory after R_track exceeds PHASE_THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    Two-phase curriculum for the batch reactor:

    Phase 0 (warm-up): use only the first `steps_phase0` steps of the
      trajectory. This is the ramp-up portion (45→55°C) which is easier
      to track and avoids runaway in early episodes. The MAT penalty in
      Phase 0 is also reduced to prevent the buffer from being flooded
      with catastrophic episodes.

    Phase 1 (full): use the complete trajectory once tracking resilience
      exceeds `r_track_threshold` for `patience` consecutive evaluations.

    Usage in training loop
    ----------------------
    curriculum = CurriculumScheduler(N_TRAJ=len(T_ref_all))
    # Inside episode start:
    effective_len = curriculum.effective_length
    T_ref_episode = T_ref_all[:effective_len]
    mat_penalty   = curriculum.mat_penalty
    # After each episode:
    curriculum.update(ep_idx, R_track)
    """

    def __init__(
        self,
        N_TRAJ:               int   = 7200,
        steps_phase0:         int   = 1000,   # first ~500 min of 3600 min trajectory
        r_track_threshold:    float = 0.92,   # advance when R_track > this
        patience:             int   = 5,       # consecutive episodes above threshold
        mat_penalty_phase0:   float = -100.0, # reduced during warm-up
        mat_penalty_phase1:   float = -500.0, # full penalty after curriculum
    ):
        self.N                   = N_TRAJ
        self.steps0              = steps_phase0
        self.threshold           = r_track_threshold
        self.patience            = patience
        self.mat_p0              = mat_penalty_phase0
        self.mat_p1              = mat_penalty_phase1
        self._phase              = 0
        self._consecutive_good   = 0

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def effective_length(self) -> int:
        return self.N if self._phase >= 1 else self.steps0

    @property
    def mat_penalty(self) -> float:
        return self.mat_p1 if self._phase >= 1 else self.mat_p0

    def update(self, ep_idx: int, R_track: float):
        """Call after each episode with its tracking resilience."""
        if self._phase == 0:
            if R_track >= self.threshold:
                self._consecutive_good += 1
                if self._consecutive_good >= self.patience:
                    self._phase = 1
                    print(f"\n[Curriculum] Phase 0→1 at episode {ep_idx+1}. "
                          f"Full trajectory active. mat_penalty={self.mat_p1}")
            else:
                self._consecutive_good = 0

    def __repr__(self):
        return (f"CurriculumScheduler(phase={self._phase}, "
                f"eff_len={self.effective_length}, "
                f"consecutive_good={self._consecutive_good}/{self.patience})")

# END PATCH I ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION GUIDE — train_ppo.py modifications
# ═══════════════════════════════════════════════════════════════════════════════
"""
STEP-BY-STEP INTEGRATION (train_ppo.py)
========================================

1. Replace build_obs with build_obs_corrected (Patch B):
   - Change I_MAX = 4.5e-5  (was 4.5e-3)

2. Replace reward inline block with compute_reward() (Patch A):
   OLD:
       r = -abs(e); done = False
       if Tr_n >= MAT: r += MAT_PENALTY; done = True; hit_MAT = True

   NEW:
       reward, reward_info, done = compute_reward(
           e=e, Tr=Tr_n, Hc=Hc, u1=u1,
           Hc_prev=Hc_prev, u1_prev=u1_prev,
           ukf_P_trace=float(np.trace(ukf.P)),
           M_hat=ukf.M_hat,
       )
       # Track prev actions:
       Hc_prev = Hc; u1_prev = u1
       if done: hit_MAT = True

3. Add entropy annealer (Patch H):
   annealer = EntropyAnnealer(N_EPISODES=N_EPISODES)
   # After each PPO update, get current per_dim_entropy from metrics
   # Pass to annealer.step(); set agent.ent_coef = annealer.step(ep_idx, per_dim_e)

4. Add curriculum (Patch I, optional but recommended):
   curriculum = CurriculumScheduler(N_TRAJ=N_TRAJ)
   # Use curriculum.effective_length for STEPS_PER_EP
   # Update with curriculum.update(ep_idx, R_track) after each episode

5. Replace ppo_network.py with ActorCriticNetworkCorrected (Patch C):
   - Update ActorCriticNetwork import/usage in ppo_agent.py
   - The new network's get_action_and_value returns same interface

6. In ppo_agent.py learn(), add per-dim entropy tracking:
   per_dim_entropy_np = mean_per_dim.detach().cpu().numpy()
   metrics['per_dim_entropy'] = per_dim_entropy_np

7. HC_LOW in train_ppo.py should remain 8.0 mA.
   The heater fix is in the reward gradient (cooperation term) and
   network init (wider log_std[0]), NOT in the physical range.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTED OUTCOMES AFTER PATCHES
# ═══════════════════════════════════════════════════════════════════════════════
"""
After applying all patches, the training log should show:

Episodes 1-75   (curriculum phase 0, warm-up):
  - Peak temp < MAT (no RUNAWAY) due to reduced trajectory length
  - R_track climbs from ~0.2 to ~0.85
  - Entropy stays > 0.8 nats (annealer holds ent_coef=0.05)

Episodes 75-150 (curriculum phase 0→1 transition):
  - R_track > 0.92 for 5 consecutive episodes → phase advances
  - Full 7200-step trajectory activates
  - Heater Hc starts showing values in [10, 16] mA (not stuck at 8)
  - Coolant u1 oscillation frequency decreases (smoothness penalty active)

Episodes 150-350 (full trajectory, entropy annealing):
  - Entropy decays from 0.05 toward 0.002 over 300 episodes
  - R_track stabilises at 0.97-0.99
  - Heater and coolant both show structured profiles:
      * Hc high (14-18 mA) during ramp-up phase
      * u1 moderate (0.2-0.4 L/min) during ramp-up
      * Both decrease as setpoint stabilises

Episodes 350-500 (exploitation):
  - Scores: -200 to -400 range (vs -300 to -800 in original)
  - No entropy collapse (floor prevents log_std[0] hitting -3 clamp)
  - UKF P_trace in reward produces small positive signal as estimator converges

Actuator plots (episode 300) should show:
  - Heater: structured profile, NOT flat at 8 mA
  - Coolant: smoother, less high-frequency oscillation
  - Both actuators active in [10-80%] of their ranges
"""