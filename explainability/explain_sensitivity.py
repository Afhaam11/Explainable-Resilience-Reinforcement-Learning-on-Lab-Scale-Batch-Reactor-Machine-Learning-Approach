
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_agent         import PPOAgent
from batch_reactor_env import BatchReactorEnv
from ukf_estimator     import UKFEstimator

os.makedirs("figures", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = 'Opeloop_HFc_TrTj.csv'
TRAJ_CSV  = 'Trajectory2.csv'
DT        = 0.5
I_MAX     = 4.5e-5;  M_MAX  = 0.7034
TR_MIN    = 40.0;    TR_MAX = 80.0
TJ_MIN    = 27.0;    TJ_MAX = 70.0
E_SCALE   = 20.0
HC_LOW    = 8.0;     HC_HIGH = 20.0
U1_LOW    = 0.0;     U1_HIGH = 0.7
MAT       = 80.0
N_STEPS_ROLLOUT = 3000
GRID_N    = 60    # for 2D heatmap grid

# obs feature indices
IDX_I = 0; IDX_M = 1; IDX_TR = 2; IDX_TJ = 3; IDX_E = 4

# ── Observation builder ───────────────────────────────────────────────────────
def build_obs(I_hat, M_hat, Tr, Tj, e):
    return np.array([
        float(np.clip(I_hat / I_MAX,                          0., 1.)),
        float(np.clip(M_hat / M_MAX,                          0., 1.)),
        float(np.clip((Tr - TR_MIN) / (TR_MAX - TR_MIN),      0., 1.)),
        float(np.clip((Tj - TJ_MIN) / (TJ_MAX - TJ_MIN),      0., 1.)),
        float(np.clip(e / E_SCALE,                           -1., 1.)),
    ], dtype=np.float32)

# ── Load agent ────────────────────────────────────────────────────────────────
print("Loading PPO agent ...")
agent = PPOAgent(obs_dim=5, act_dim=2, chkpt_dir='', name='ppo_reactor')
agent.load()
agent.network.eval()

# ── Gradient computation function ────────────────────────────────────────────
def compute_gradients_batch(obs_array: np.ndarray) -> dict:
    """
    For each observation in obs_array, compute:
        dHc/d(obs_i) and du1/d(obs_i)  for all i using autograd.

    Returns dict of gradient arrays, shape (N,) for each (output, input) pair.
    """
    X = T.tensor(obs_array, dtype=T.float32, requires_grad=False).to(agent.device)
    X.requires_grad_(True)

    feats   = agent.network.backbone(X)
    pre_t   = agent.network.actor_head(feats)       # (N, 2) pre-tanh
    mu_sq   = T.tanh(pre_t)                          # (N, 2) in (-1,1)

    # Physical outputs — differentiable mapping
    Hc_phys = HC_LOW + (mu_sq[:, 0] + 1.0) / 2.0 * (HC_HIGH - HC_LOW)  # (N,)
    U1_phys = U1_LOW + (mu_sq[:, 1] + 1.0) / 2.0 * (U1_HIGH - U1_LOW)  # (N,)

    def get_grad(output_vec, input_tensor):
        """Compute d(sum(output)) / d(input) via autograd."""
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        output_vec.sum().backward(retain_graph=True)
        return input_tensor.grad.detach().cpu().numpy().copy()

    grad_hc = get_grad(Hc_phys, X)   # (N, 5)
    X.grad.zero_()
    grad_u1 = get_grad(U1_phys, X)   # (N, 5)

    return {
        'dHc_dTr':  grad_hc[:, IDX_TR],
        'dHc_dM':   grad_hc[:, IDX_M],
        'dHc_dE':   grad_hc[:, IDX_E],
        'dHc_dTj':  grad_hc[:, IDX_TJ],
        'du1_dTr':  grad_u1[:, IDX_TR],
        'du1_dM':   grad_u1[:, IDX_M],
        'du1_dE':   grad_u1[:, IDX_E],
        'du1_dTj':  grad_u1[:, IDX_TJ],
    }

# ── Run rollout + collect gradients along trajectory ─────────────────────────
print("Running rollout and collecting gradients ...")
try:
    T_ref_all = pd.read_csv(TRAJ_CSV)['x_Traject'].values
except FileNotFoundError:
    t_syn = np.arange(7201) * DT
    T_ref_all = 45.0 + 20.0*(1 - np.exp(-t_syn/500))*np.exp(
        -np.maximum(t_syn-2500,0)/2000)

env = BatchReactorEnv(data_path=DATA_PATH, dt=DT, initial_history_size=50)
s0  = env.reset()
Tr, Tj = float(s0[2]), float(s0[3])
ukf = UKFEstimator(reactor_ode=env._BR_plant, dt=DT,
                   x0=np.array([float(s0[0]), float(s0[1]), Tr, Tj]))

obs_list, Tr_traj, M_traj, Hc_traj, u1_traj, t_list = [], [], [], [], [], []
N = min(N_STEPS_ROLLOUT, len(T_ref_all) - 1)

for step in range(N):
    e   = Tr - T_ref_all[step]
    obs = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj, e)
    obs_list.append(obs.copy())
    Tr_traj.append(Tr); M_traj.append(ukf.M_hat); t_list.append(step * DT)
    Hc, u1 = agent.deterministic_action(obs)
    Hc_traj.append(Hc); u1_traj.append(u1)
    Tr_n, Tj_n = env.step(u1, Hc)
    ukf.update(np.array([Tr_n, Tj_n]), u1=u1, Hc=Hc)
    Tr, Tj = Tr_n, Tj_n

obs_arr = np.array(obs_list, dtype=np.float32)
t_arr   = np.array(t_list)
Tr_arr  = np.array(Tr_traj); M_arr  = np.array(M_traj)
Hc_arr  = np.array(Hc_traj); u1_arr = np.array(u1_traj)
print(f"  {len(obs_arr)} steps collected.")

grads = compute_gradients_batch(obs_arr)
print("  Gradients computed.")

# ── Rolling mean smoother ─────────────────────────────────────────────────────
def smooth(x, w=30):
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values

# ── Time-series sensitivity plots ─────────────────────────────────────────────
BG = '#F9F9F9'
PAIRS = [
    ('dHc_dTr',  'Heater sensitivity  ∂Hc/∂Tr_norm',
     '#C1121F', 'figures/heater_temperature_sensitivity.png',
     r'$\partial H_c / \partial T_r^{norm}$  [mA / norm]', Tr_arr,
     r'$T_r$ [°C]'),
    ('du1_dTr',  'Coolant sensitivity  ∂u1/∂Tr_norm',
     '#2B7CB0', 'figures/coolant_temperature_sensitivity.png',
     r'$\partial u_1 / \partial T_r^{norm}$  [L/min / norm]', Tr_arr,
     r'$T_r$ [°C]'),
    ('dHc_dM',   'Heater sensitivity  ∂Hc/∂M_hat_norm',
     '#C1121F', 'figures/heater_monomer_sensitivity.png',
     r'$\partial H_c / \partial \hat{M}_{norm}$  [mA / norm]', M_arr,
     r'$\hat{M}$ [mol/L]'),
    ('du1_dM',   'Coolant sensitivity  ∂u1/∂M_hat_norm',
     '#2B7CB0', 'figures/coolant_monomer_sensitivity.png',
     r'$\partial u_1 / \partial \hat{M}_{norm}$  [L/min / norm]', M_arr,
     r'$\hat{M}$ [mol/L]'),
]

for key, title, color, fname, ylabel, state_var, state_label in PAIRS:
    g = grads[key]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1.5]})
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG); ax2.set_facecolor(BG)

    # Raw gradient
    ax1.plot(t_arr, g, color=color, alpha=0.25, lw=0.6, label='Raw gradient')
    ax1.plot(t_arr, smooth(g, 30), color=color, lw=2.0, label='30-step rolling mean')
    ax1.axhline(0, color='#888', lw=1.0, linestyle='--')
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax1.legend(fontsize=9)
    ax1.grid(color='#E0E0E0', lw=0.6)

    # State variable on lower panel
    ax2.plot(t_arr, state_var, color='#555', lw=1.2)
    ax2.set_ylabel(state_label, fontsize=10)
    ax2.set_xlabel('Time [min]', fontsize=11)
    ax2.grid(color='#E0E0E0', lw=0.6)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

