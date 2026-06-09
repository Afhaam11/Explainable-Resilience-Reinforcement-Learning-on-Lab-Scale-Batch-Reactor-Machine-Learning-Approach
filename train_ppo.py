
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_agent import PPOAgent
from batch_reactor_env import BatchReactorEnv
from ukf_estimator import UKFEstimator
from resilience_acrylamide import tracking_resilience
os.makedirs("figures", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAJECTORY_CSV = 'Trajectory2.csv'
DATA_PATH      = 'Opeloop_HFc_TrTj.csv'
ALGO_NAME      = 'ppo_reactor'
N_EPISODES     = 500
SNAPSHOT_EP    = 299
DT             = 0.5
MAT            = 80.0
T_SCALE        = 5.0
HC_LOW         = 8.0;   HC_HIGH = 20.0
U1_LOW         = 0.0;   U1_HIGH = 0.7
N_STEPS        = 2048;  N_EPOCHS = 10;    BATCH_SZ  = 64
GAMMA          = 0.99;  GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
LR             = 3e-4;  ENT_COEF   = 0.05  # initial — will be annealed
VF_COEF        = 0.5;   MAX_GRAD   = 0.5
N_HIDDEN       = 256;   N_LAYERS   = 2

# ── Trajectory ────────────────────────────────────────────────────────────────
try:
    T_ref_all = pd.read_csv(TRAJECTORY_CSV)['x_Traject'].values
except FileNotFoundError:
    t_syn = np.arange(7201) * DT
    T_ref_all = 45.0 + 20.0 * (1 - np.exp(-t_syn / 500)) * np.exp(
        -np.maximum(t_syn - 2500, 0) / 2000)
N_TRAJ = len(T_ref_all)
T_ref_min = float(T_ref_all.min())
T_ref_max = float(T_ref_all.max())
print(f"Trajectory: {N_TRAJ} steps ({N_TRAJ*DT:.0f} min) "
      f"T_ref=[{T_ref_min:.2f},{T_ref_max:.2f}]°C")

# ══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH B — observation normalisation (I_MAX corrected)
# ══════════════════════════════════════════════════════════════════════════════
# CORRECTED: I_MAX = 4.5e-5 (actual reset concentration)
# Original had I_MAX = 4.5e-3 → obs[0] was always ≈ 0.01, giving zero gradient
I_MAX  = 4.5e-5    # ← CORRECTED (was 4.5e-3)
M_MAX  = 0.7034
TR_MIN = T_ref_min - 5.0
TR_MAX = MAT
TJ_MIN = 27.0
TJ_MAX = 70.0
E_SCALE = 20.0
OBS_DIM = 5


def build_obs(I_hat: float, M_hat: float,
              Tr: float, Tj: float, e: float) -> np.ndarray:
    """5D normalised observation with corrected I_hat range."""
    return np.array([
        float(np.clip(I_hat / I_MAX,                           0., 1.)),
        float(np.clip(M_hat / M_MAX,                           0., 1.)),
        float(np.clip((Tr - TR_MIN) / (TR_MAX - TR_MIN),       0., 1.)),
        float(np.clip((Tj - TJ_MIN) / (TJ_MAX - TJ_MIN),       0., 1.)),
        float(np.clip(e / E_SCALE,                            -1., 1.)),
    ], dtype=np.float32)
# END PATCH B ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH A — reward function
# ══════════════════════════════════════════════════════════════════════════════
_MAT_MARGIN    = 10.0    # barrier activates within 10°C of MAT
_MAT_BARRIER_K = 2.0
_MAT_TERMINAL  = -500.0  # reduced from -5000; log-barrier handles gradient
_COOP_ALPHA    = 0.30    # cooperation reward weight
_SMOOTH_BETA   = 0.15    # smoothness penalty weight
_COOP_GAMMA    = 0.20    # UKF confidence reward weight
_UKF_P_SCALE   = 2.0


def compute_reward(e, Tr, Hc, u1, Hc_prev, u1_prev,
                   ukf_P_trace, M_hat, done_mat=False):
    """
    5-component corrected reward. See ppo_patches_corrected.py Patch A.
    """
    # 1. Tracking (primary)
    r_track = -abs(e)

    # 2. Actuator cooperation
    hc_norm = (Hc - HC_LOW) / (HC_HIGH - HC_LOW)
    u1_norm = u1 / U1_HIGH
    coop_hc = 4.0 * hc_norm * (1.0 - hc_norm)
    coop_u1 = 4.0 * u1_norm * (1.0 - u1_norm)
    r_coop  = _COOP_ALPHA * coop_hc * coop_u1

    # 3. Smoothness penalty
    dhc = (Hc - Hc_prev) / (HC_HIGH - HC_LOW)
    du1 = (u1 - u1_prev) / U1_HIGH
    r_smooth = -_SMOOTH_BETA * (dhc**2 + du1**2)

    # 4. MAT soft log-barrier
    d = MAT - Tr
    done = False
    if d <= 0.0:
        r_barrier = _MAT_TERMINAL
        done      = True
    elif d < _MAT_MARGIN:
        r_barrier = -_MAT_BARRIER_K * np.log(d / _MAT_MARGIN + 1e-8)
    else:
        r_barrier = 0.0

    # 5. UKF confidence (trace P)
    ukf_conf = float(np.exp(-ukf_P_trace / _UKF_P_SCALE))
    r_ukf    = _COOP_GAMMA * ukf_conf

    # 6. Chemical state
    r_chem = 0.05 * float(np.clip(M_hat / M_MAX, 0., 1.))

    if done_mat or done:
        total = r_track + r_barrier
        done  = True
    else:
        total = r_track + r_coop + r_smooth + r_barrier + r_ukf + r_chem

    return float(total), done
# END PATCH A ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH H — entropy annealing schedule
# ══════════════════════════════════════════════════════════════════════════════
class EntropyAnnealer:
    def __init__(self, N_EPISODES=500, ent_start=0.05, ent_min=0.002,
                 warmup_frac=0.15, decay_frac=0.60, floor_nats=0.30):
        self.N          = N_EPISODES
        self.ent_start  = ent_start
        self.ent_min    = ent_min
        self.warmup_end = int(N_EPISODES * warmup_frac)
        self.decay_end  = int(N_EPISODES * (warmup_frac + decay_frac))
        self.floor_nats = floor_nats
        self._current   = ent_start
        self._paused    = False

    def step(self, ep_idx, per_dim_entropy=None):
        if per_dim_entropy is not None:
            self._paused = bool(np.any(per_dim_entropy < self.floor_nats))
        if self._paused:
            return float(self._current)
        if ep_idx < self.warmup_end:
            coef = self.ent_start
        elif ep_idx < self.decay_end:
            progress = (ep_idx - self.warmup_end) / max(
                self.decay_end - self.warmup_end, 1)
            coef = self.ent_start + progress * (self.ent_min - self.ent_start)
        else:
            coef = self.ent_min
        self._current = float(coef)
        return self._current
# END PATCH H ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BEGIN PATCH I — curriculum training scheduler
# ══════════════════════════════════════════════════════════════════════════════
class CurriculumScheduler:
    def __init__(self, N_TRAJ=7200, steps_phase0=1200,
                 r_track_threshold=0.92, patience=5,
                 mat_penalty_phase0=-100.0, mat_penalty_phase1=-500.0):
        self.N         = N_TRAJ
        self.steps0    = steps_phase0
        self.threshold = r_track_threshold
        self.patience  = patience
        self.mat_p0    = mat_penalty_phase0
        self.mat_p1    = mat_penalty_phase1
        self._phase    = 0
        self._consec   = 0

    @property
    def phase(self): return self._phase

    @property
    def effective_length(self):
        return self.N if self._phase >= 1 else self.steps0

    @property
    def mat_penalty(self):
        return self.mat_p1 if self._phase >= 1 else self.mat_p0

    def update(self, ep_idx, R_track):
        if self._phase == 0:
            if R_track >= self.threshold:
                self._consec += 1
                if self._consec >= self.patience:
                    self._phase = 1
                    print(f"\n[Curriculum] Phase 0→1 at ep {ep_idx+1}. "
                          f"Full {self.N}-step trajectory active.")
            else:
                self._consec = 0
# END PATCH I ══════════════════════════════════════════════════════════════════


# ── Agent + Env ───────────────────────────────────────────────────────────────
agent = PPOAgent(obs_dim=OBS_DIM, act_dim=2, n_steps=N_STEPS, n_epochs=N_EPOCHS,
                 batch_size=BATCH_SZ, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                 clip_eps=CLIP_EPS, lr=LR, ent_coef=ENT_COEF, vf_coef=VF_COEF,
                 max_grad_norm=MAX_GRAD, n_hidden=N_HIDDEN, n_layers=N_LAYERS,
                 chkpt_dir='', name=ALGO_NAME)
env = BatchReactorEnv(data_path=DATA_PATH, dt=DT, initial_history_size=50)

annealer   = EntropyAnnealer(N_EPISODES=N_EPISODES)
curriculum = CurriculumScheduler(N_TRAJ=N_TRAJ)

print("=" * 68)
print(f"PPO+UKF CORRECTED | obs_dim={OBS_DIM} | act_dim=2")
print(f"obs = [I_hat_norm(I_MAX=4.5e-5), M_hat_norm, Tr_norm, Tj_norm, e_norm]")
print(f"act = [Hc∈[{HC_LOW},{HC_HIGH}]mA, u1∈[{U1_LOW},{U1_HIGH}]L/min]")
print(f"Reward: tracking + cooperation + smoothness + MAT-barrier + UKF-conf")
print(f"MAT soft barrier: activates {_MAT_MARGIN}°C below MAT={MAT}°C")
print(f"Entropy annealing: {annealer.ent_start}→{annealer.ent_min} "
      f"over eps {annealer.warmup_end}–{annealer.decay_end}")
print(f"Curriculum: phase-0 uses {curriculum.steps0} steps; "
      f"phase-1 uses full {N_TRAJ} steps")
print("=" * 68)

# ── Training ──────────────────────────────────────────────────────────────────
scores = []; R_track_hist = []; best_score = -np.inf
snap_Tr = snap_Hc = snap_u1 = snap_ref = snap_steps = None
per_dim_entropy_last = np.array([1.0, 1.0])   # initial — before first update

# Saturation counters for diagnostics
sat_hc_count = 0; sat_u1_count = 0; total_steps_count = 0

# Episode initialisation
s0 = env.reset()
Tr = float(s0[2]) + np.random.uniform(-0.3, 0.3)
Tj = float(s0[3])
ukf = UKFEstimator(reactor_ode=env._BR_plant, dt=DT,
                   x0=np.array([float(s0[0]), float(s0[1]), Tr, Tj]))

ep_idx = 0; step_in_ep = 0; ep_reward = 0.0
ep_Tr_list = [Tr]; ep_Hc_list = []; ep_u1_list = []
Hc_prev = (HC_LOW + HC_HIGH) / 2.0   # init prev actions at midpoint
u1_prev = U1_HIGH / 2.0
hit_MAT = False
last_obs = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj, Tr - T_ref_all[0])
last_done = False
total_steps = N_EPISODES * N_TRAJ

while ep_idx < N_EPISODES:
    # Effective trajectory length (curriculum phase 0 = shorter)
    STEPS_PER_EP = curriculum.effective_length - 1

    for _ in range(N_STEPS):
        if ep_idx >= N_EPISODES:
            break

        tidx    = min(step_in_ep, N_TRAJ - 1)
        Tr_ref  = T_ref_all[tidx]
        e       = Tr - Tr_ref
        obs     = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj, e)

        # ── PATCH E: saturation-aware action selection ────────────────────
        raw_action, log_prob, value, Hc, u1 = agent.choose_action(obs)
        # Saturation diagnostic (silent in normal training)
        sat_hc = bool(abs(raw_action[0]) > 0.95)
        sat_u1 = bool(abs(raw_action[1]) > 0.95)
        sat_hc_count  += int(sat_hc)
        sat_u1_count  += int(sat_u1)
        total_steps_count += 1

        # ── Plant step ────────────────────────────────────────────────────
        Tr_n, Tj_n = env.step(u1, Hc)
        ukf.update(np.array([Tr_n, Tj_n]), u1=u1, Hc=Hc)

        # ── PATCH A: compute corrected reward ─────────────────────────────
        ukf_P_trace = float(np.trace(ukf.P))
        r, done     = compute_reward(
            e=e, Tr=Tr_n, Hc=Hc, u1=u1,
            Hc_prev=Hc_prev, u1_prev=u1_prev,
            ukf_P_trace=ukf_P_trace,
            M_hat=ukf.M_hat,
            done_mat=False,
        )
        if done:
            hit_MAT = True

        Hc_prev = Hc; u1_prev = u1

        ep_reward  += r
        ep_Tr_list.append(Tr_n)
        ep_Hc_list.append(Hc)
        ep_u1_list.append(u1)

        agent.store(obs, raw_action, log_prob, r, value, done)
        step_in_ep += 1

        # Next obs
        tidx_n  = min(step_in_ep, N_TRAJ - 1)
        e_n     = Tr_n - T_ref_all[tidx_n]
        next_obs = build_obs(ukf.I_hat, ukf.M_hat, Tr_n, Tj_n, e_n)
        last_obs  = next_obs
        last_done = done

        ep_ended = done or (step_in_ep >= STEPS_PER_EP)

        if ep_ended:
            Tr_arr  = np.array(ep_Tr_list)
            ref_arr = T_ref_all[:len(Tr_arr)]
            t_arr_ep = np.arange(len(Tr_arr)) * DT

            Q_p = np.exp(-np.abs(Tr_arr - ref_arr) / T_SCALE)
            R_track = (float(np.trapezoid(Q_p, t_arr_ep) / max(t_arr_ep[-1] - t_arr_ep[0], 1e-9))
                       if len(t_arr_ep) >= 2 else float(np.mean(Q_p)))
            R_track = float(np.clip(R_track, 0, 1))

            scores.append(ep_reward)
            R_track_hist.append(R_track)

            # ── PATCH I: curriculum update ─────────────────────────────────
            curriculum.update(ep_idx, R_track)

            # ── PATCH H: entropy annealing ─────────────────────────────────
            current_ent_coef = annealer.step(ep_idx, per_dim_entropy_last)
            agent.ent_coef   = current_ent_coef

            if ep_reward > best_score:
                agent.save()
                best_score = ep_reward

            if ep_idx == SNAPSHOT_EP:
                snap_Tr    = list(ep_Tr_list)
                snap_Hc    = list(ep_Hc_list)
                snap_u1    = list(ep_u1_list)
                snap_ref   = list(T_ref_all[:len(ep_Tr_list)])
                snap_steps = list(range(len(ep_Tr_list)))

            if ep_idx % 50 == 0:
                m50   = float(np.mean(scores[-50:]) if len(scores) >= 50
                              else np.mean(scores))
                sat_r = sat_hc_count / max(total_steps_count, 1)
                print(f"Ep {ep_idx+1:4d}/{N_EPISODES} | "
                      f"Score={ep_reward:8.1f} | Mean50={m50:8.1f} | "
                      f"R_track={R_track:.4f} | peak={max(ep_Tr_list):.1f}°C | "
                      f"{'RUNAWAY' if hit_MAT else 'OK':7s} | "
                      f"ent_coef={current_ent_coef:.4f} | "
                      f"Hc_sat={sat_r:.2%} | "
                      f"phase={curriculum.phase}")

            ep_idx    += 1
            s0         = env.reset()
            Tr         = float(s0[2]) + np.random.uniform(-0.3, 0.3)
            Tj         = float(s0[3])
            ukf.reset(x0=np.array([float(s0[0]), float(s0[1]), Tr, Tj]))
            step_in_ep = 0; ep_reward = 0.0
            ep_Tr_list = [Tr]; ep_Hc_list = []; ep_u1_list = []
            Hc_prev    = (HC_LOW + HC_HIGH) / 2.0
            u1_prev    = U1_HIGH / 2.0
            hit_MAT    = False; last_done = False
            last_obs   = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj,
                                   Tr - T_ref_all[0])

        Tr = Tr_n; Tj = Tj_n
        if ep_idx >= N_EPISODES:
            break

    if agent.buffer.is_full():
        metrics = agent.learn(last_obs, last_done)

        # ── PATCH D: per-dim entropy for floor check ─────────────────────
        # Approximate per-dim entropy from log_std directly (exact for Gaussian)
        import torch as _T
        with _T.no_grad():
            log_std_vals = agent.network.log_std.detach().cpu().numpy()
        per_dim_entropy_last = 0.5 * (1 + np.log(2 * np.pi)) + log_std_vals
        # per_dim_entropy_last shape: (n_actions,) in nats

        if agent.train_step % 20 == 0:
            ent_per_dim_str = ' '.join(
                f"dim{i}={v:.3f}" for i, v in enumerate(per_dim_entropy_last))
            print(f"  [PPO {agent.train_step:4d}] "
                  f"pol={metrics['policy_loss']:7.4f}  "
                  f"val={metrics['value_loss']:7.4f}  "
                  f"ent={metrics['entropy']:.4f}  "
                  f"kl={metrics['approx_kl']:.4f}  "
                  f"per_dim_ent: {ent_per_dim_str}  "
                  f"ent_coef={agent.ent_coef:.4f}"
                  f"{' [PAUSED]' if annealer._paused else ''}")

    if ep_idx >= N_EPISODES:
        break

print("=" * 68)
print(f"Training complete. Best score={best_score:.1f}")
print(f"Hc saturation rate: {sat_hc_count}/{total_steps_count} "
      f"= {sat_hc_count/max(total_steps_count,1):.2%}")
print(f"u1 saturation rate: {sat_u1_count}/{total_steps_count} "
      f"= {sat_u1_count/max(total_steps_count,1):.2%}")
print("=" * 68)

# ── Plots ─────────────────────────────────────────────────────────────────────
if snap_Tr is None:
    snap_Tr    = list(ep_Tr_list)
    snap_Hc    = list(ep_Hc_list)
    snap_u1    = list(ep_u1_list)
    snap_ref   = list(T_ref_all[:len(ep_Tr_list)])
    snap_steps = list(range(len(ep_Tr_list)))

snap_Tr    = np.array(snap_Tr)
snap_Hc    = np.array(snap_Hc)
snap_u1    = np.array(snap_u1)
snap_ref   = np.array(snap_ref)
snap_steps = np.array(snap_steps)
ctrl_steps = snap_steps[1:]

fig, axes = plt.subplots(4, 1, figsize=(12, 14))
fig.suptitle(
    f'Training Episode {SNAPSHOT_EP+1}/{N_EPISODES} — PPO+UKF CORRECTED\n'
    f'I_MAX=4.5e-5 | cooperation+smoothness+barrier reward | per-dim log_std',
    fontsize=11, fontweight='bold')

axes[0].plot(snap_steps, snap_Tr,  color='blue',  lw=1.5, label='Reactor Temp')
axes[0].plot(snap_steps, snap_ref, color='green', lw=1.5, linestyle='--', label='Setpoint')
axes[0].axhline(MAT, color='red',  lw=1.0, linestyle=':', alpha=0.7, label=f'MAT={MAT}°C')
axes[0].axhspan(MAT - _MAT_MARGIN, MAT, alpha=0.08, color='orange',
                label=f'Barrier zone ({_MAT_MARGIN}°C)')
axes[0].set_ylabel('Temp (°C)', fontsize=11)
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, max(len(snap_steps) - 1, 1)])

axes[1].plot(ctrl_steps, snap_Hc, color='red', lw=0.8, alpha=0.85)
axes[1].set_ylabel('Heater (mA)', fontsize=11)
axes[1].set_ylim([HC_LOW - 1, HC_HIGH + 1])
axes[1].set_yticks([HC_LOW, (HC_LOW + HC_HIGH) / 2, HC_HIGH])
axes[1].axhline((HC_LOW + HC_HIGH) / 2, color='gray', lw=0.5,
                linestyle='--', alpha=0.5, label='midpoint')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, max(len(snap_steps) - 1, 1)])

axes[2].plot(ctrl_steps, snap_u1, color='cyan', lw=0.8, alpha=0.85)
axes[2].set_ylabel('Coolant (L/min)', fontsize=11)
axes[2].set_ylim([-0.02, U1_HIGH + 0.1])
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, max(len(snap_steps) - 1, 1)])

axes[3].plot(range(len(scores)), scores, color='magenta', lw=1.0, alpha=0.6)
if len(scores) >= 10:
    axes[3].plot(range(len(scores)),
                 pd.Series(scores).rolling(min(50, len(scores)), min_periods=1).mean(),
                 color='darkmagenta', lw=2.0, label='50-ep mean')
axes[3].axvline(x=SNAPSHOT_EP, color='gray', lw=1.0,
                linestyle='--', alpha=0.6, label=f'Ep {SNAPSHOT_EP+1}')
axes[3].set_xlabel('Episode Number', fontsize=11)
axes[3].set_ylabel('Total Reward', fontsize=11)
axes[3].set_title('Learning Curve: Total Reward per Episode', fontsize=11)
axes[3].legend(loc='lower right', fontsize=9)
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim([0, N_EPISODES])

plt.tight_layout()
plt.savefig('figures/ppo_training_episode.png', dpi=300, bbox_inches='tight')
plt.close()
print("Training-episode plot → figures/ppo_training_episode.png")

# Resilience + reward curves
fig2, (b1, b2) = plt.subplots(2, 1, figsize=(10, 7))
b1.plot(R_track_hist, lw=0.5, color='steelblue', alpha=0.4)
if len(R_track_hist) >= 10:
    b1.plot(pd.Series(R_track_hist).rolling(50, min_periods=1).mean(),
            lw=2.0, color='steelblue', label='50-ep mean R_track')
b1.axhline(1.0, color='green', lw=0.8, linestyle='--', label='R=1 (perfect)')
b1.set_ylim([0, 1.05]); b1.legend(); b1.grid(True, alpha=0.3)
b1.set_xlabel('Episode'); b1.set_ylabel('Tracking Resilience R')
b2.plot(scores, lw=0.5, color='darkorange', alpha=0.4)
if len(scores) >= 10:
    b2.plot(pd.Series(scores).rolling(50, min_periods=1).mean(),
            lw=2.0, color='darkorange', label='50-ep mean')
b2.legend(); b2.grid(True, alpha=0.3)
b2.set_xlabel('Episode'); b2.set_ylabel('Total Reward')
plt.tight_layout()
plt.savefig('figures/ppo_training_curves.png', dpi=300)
plt.close()
print("Training curves → figures/ppo_training_curves.png")
print("Done.")