
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_agent         import PPOAgent
from batch_reactor_env import BatchReactorEnv
from ukf_estimator     import UKFEstimator

os.makedirs("figures", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = 'Opeloop_HFc_TrTj.csv'
TRAJ_CSV   = 'Trajectory2.csv'
DT         = 0.5
I_MAX      = 4.5e-5;  M_MAX  = 0.7034
TR_MIN     = 40.0;    TR_MAX = 80.0
TJ_MIN     = 27.0;    TJ_MAX = 70.0
E_SCALE    = 20.0
HC_LOW     = 8.0;     HC_HIGH = 20.0
U1_LOW     = 0.0;     U1_HIGH = 0.7
MAT        = 80.0
GRID_N     = 80       # grid resolution per axis

# Nominal fixed values for the non-scanned dimensions
I_HAT_NOMINAL  = 1.5e-5   # mid-episode UKF estimate
TJ_NOMINAL     = 42.5     # nominal jacket temp
E_NOMINAL      = 0.0      # zero tracking error (on-setpoint)

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

# ── Collect trajectory for overlay ───────────────────────────────────────────
print("Running rollout for trajectory overlay ...")
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

traj_Tr, traj_M, traj_Hc, traj_u1 = [], [], [], []
N = min(3000, len(T_ref_all) - 1)
for step in range(N):
    e   = Tr - T_ref_all[step]
    obs = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj, e)
    Hc, u1 = agent.deterministic_action(obs)
    traj_Tr.append(Tr); traj_M.append(ukf.M_hat)
    traj_Hc.append(Hc); traj_u1.append(u1)
    Tr_n, Tj_n = env.step(u1, Hc)
    ukf.update(np.array([Tr_n, Tj_n]), u1=u1, Hc=Hc)
    Tr, Tj = Tr_n, Tj_n

traj_Tr = np.array(traj_Tr); traj_M  = np.array(traj_M)
traj_Hc = np.array(traj_Hc); traj_u1 = np.array(traj_u1)

# ── Build 2D policy surface ───────────────────────────────────────────────────
print(f"Building {GRID_N}×{GRID_N} policy surface grid ...")

# Physical axes
Tr_phys = np.linspace(TR_MIN, TR_MAX - 1.0, GRID_N)   # avoid MAT
M_phys  = np.linspace(0.05,   M_MAX,         GRID_N)

Tr_grid, M_grid = np.meshgrid(Tr_phys, M_phys)   # shape (GRID_N, GRID_N)

# Build observation matrix — fix I_hat, Tj, e at nominals
obs_matrix = np.zeros((GRID_N * GRID_N, 5), dtype=np.float32)
obs_flat   = obs_matrix
idx = 0
for mi in range(GRID_N):
    for ti in range(GRID_N):
        obs_flat[idx] = build_obs(
            I_HAT_NOMINAL, M_grid[mi, ti],
            Tr_grid[mi, ti], TJ_NOMINAL, E_NOMINAL
        )
        idx += 1

# Batch inference
with T.no_grad():
    X_t   = T.tensor(obs_flat, dtype=T.float32).to(agent.device)
    feats = agent.network.backbone(X_t)
    pre_t = agent.network.actor_head(feats)
    mu_sq = T.tanh(pre_t).cpu().numpy()

Hc_raw = mu_sq[:, 0]
U1_raw = mu_sq[:, 1]
Hc_surf = (HC_LOW + (Hc_raw + 1.0) / 2.0 * (HC_HIGH - HC_LOW)).reshape(GRID_N, GRID_N)
U1_surf = (U1_LOW + (U1_raw + 1.0) / 2.0 * (U1_HIGH - U1_LOW)).reshape(GRID_N, GRID_N)
print("  Grid computed.")

# ── Plotting helper ───────────────────────────────────────────────────────────
BG = '#111116'

def surface_plot(surf, cmap, vmin, vmax, traj_x, traj_y, traj_c,
                 traj_cmap, traj_vmin, traj_vmax,
                 xlabel, ylabel, cbar_label,
                 title, fname,
                 contour_levels=8,
                 mat_line=True):

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Filled contour surface
    cf = ax.contourf(Tr_phys, M_phys, surf,
                     levels=64, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9)

    # Contour lines
    cs = ax.contour(Tr_phys, M_phys, surf,
                    levels=contour_levels, colors='white', alpha=0.35, linewidths=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f', colors='white')

    # MAT line
    if mat_line:
        ax.axvline(MAT, color='#FF4444', lw=2.0, linestyle='--',
                   label=f'MAT = {MAT}°C', zorder=5)

    # Trajectory overlay
    sc = ax.scatter(traj_x, traj_y, c=traj_c, s=4, cmap=traj_cmap,
                    vmin=traj_vmin, vmax=traj_vmax,
                    alpha=0.6, zorder=6, linewidths=0, label='Trajectory')

    # Colorbars
    cbar1 = fig.colorbar(cf, ax=ax, pad=0.01, shrink=0.85)
    cbar1.set_label(cbar_label, color='white', fontsize=11)
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

    cbar2 = fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.55,
                         location='right')
    cbar2.set_label('Time →', color='white', fontsize=9)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')

    ax.set_xlabel(xlabel, fontsize=12, color='white')
    ax.set_ylabel(ylabel, fontsize=12, color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')
    ax.legend(loc='upper left', fontsize=9,
              facecolor='#333', edgecolor='#666', labelcolor='white')

    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {fname}")

# Time axis for trajectory colormap
t_traj = np.arange(len(traj_Tr))

surface_plot(
    Hc_surf, 'YlOrRd', HC_LOW, HC_HIGH,
    traj_Tr, traj_M, t_traj,
    'cool_r', 0, len(traj_Tr),
    r'Reactor Temperature $T_r$ [°C]',
    r'Monomer Concentration $\hat{M}$ [mol/L]',
    r'$H_c$ [mA]',
    r'Policy Surface: Heater Current $H_c(T_r, \hat{M})$',
    'figures/heater_surface.png',
)

surface_plot(
    U1_surf, 'Blues', U1_LOW, U1_HIGH,
    traj_Tr, traj_M, t_traj,
    'autumn_r', 0, len(traj_Tr),
    r'Reactor Temperature $T_r$ [°C]',
    r'Monomer Concentration $\hat{M}$ [mol/L]',
    r'$u_1$ [L/min]',
    r'Policy Surface: Coolant Flow $u_1(T_r, \hat{M})$',
    'figures/coolant_surface.png',
)

# ── Side-by-side overlay ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.patch.set_facecolor(BG)

for ax, surf, cmap, vmin, vmax, traj_c, tcmap, tlbl in [
    (axes[0], Hc_surf, 'YlOrRd', HC_LOW, HC_HIGH,
     traj_Hc, 'YlOrRd', r'$H_c$ [mA]'),
    (axes[1], U1_surf, 'Blues',   U1_LOW, U1_HIGH,
     traj_u1, 'Blues',   r'$u_1$ [L/min]'),
]:
    ax.set_facecolor(BG)
    cf = ax.contourf(Tr_phys, M_phys, surf, levels=64,
                     cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.88)
    ax.contour(Tr_phys, M_phys, surf, levels=8,
               colors='white', alpha=0.3, linewidths=0.6)
    ax.axvline(MAT, color='#FF4444', lw=1.8, linestyle='--', label=f'MAT={MAT}°C')
    sc = ax.scatter(traj_Tr, traj_M, c=traj_c, s=5,
                    cmap=tcmap, vmin=vmin, vmax=vmax,
                    alpha=0.7, zorder=5, linewidths=0)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(tlbl, color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax.set_xlabel(r'$T_r$ [°C]', fontsize=11, color='white')
    ax.set_ylabel(r'$\hat{M}$ [mol/L]', fontsize=11, color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#555')
    ax.legend(loc='upper left', fontsize=8,
              facecolor='#333', edgecolor='#666', labelcolor='white')

axes[0].set_title(r'Heater $H_c(T_r, \hat{M})$',
                  fontsize=13, fontweight='bold', color='white', pad=8)
axes[1].set_title(r'Coolant $u_1(T_r, \hat{M})$',
                  fontsize=13, fontweight='bold', color='white', pad=8)
fig.suptitle('PPO Policy Surfaces with Trajectory Overlay',
             fontsize=15, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig('figures/policy_surface_overlay.png', dpi=180,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved: figures/policy_surface_overlay.png")
print("\nAll policy surface plots saved to figures/")
