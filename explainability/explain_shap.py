
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch as T

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import shap
except ImportError:
    raise ImportError("Install shap: pip install shap")

from ppo_agent       import PPOAgent
from batch_reactor_env import BatchReactorEnv
from ukf_estimator   import UKFEstimator

os.makedirs("figures", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = 'ppo_reactor.pt'
DATA_PATH    = 'Opeloop_HFc_TrTj.csv'
TRAJ_CSV     = 'Trajectory2.csv'
DT           = 0.5
I_MAX        = 4.5e-5
M_MAX        = 0.7034
TR_MIN       = 40.0;  TR_MAX = 80.0
TJ_MIN       = 27.0;  TJ_MAX = 70.0
E_SCALE      = 20.0
HC_LOW       = 8.0;   HC_HIGH = 20.0
U1_LOW       = 0.0;   U1_HIGH = 0.7
N_BACKGROUND = 100    # SHAP background samples
N_EXPLAIN    = 500    # samples to explain
FEATURE_NAMES = [
    r'$\hat{I}_{norm}$  (initiator)',
    r'$\hat{M}_{norm}$  (monomer)',
    r'$T_r^{norm}$  (reactor temp)',
    r'$T_j^{norm}$  (jacket temp)',
    r'$e_{norm}$  (tracking error)',
]
FEATURE_NAMES_SHORT = ['I_hat_norm', 'M_hat_norm', 'Tr_norm', 'Tj_norm', 'e_norm']

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
print("  Agent loaded.")

# ── Rollout to collect state dataset ─────────────────────────────────────────
print("Collecting state-action dataset via deterministic rollout ...")
try:
    T_ref_all = pd.read_csv(TRAJ_CSV)['x_Traject'].values
except FileNotFoundError:
    t_syn = np.arange(7201) * DT
    T_ref_all = 45.0 + 20.0 * (1 - np.exp(-t_syn / 500)) * np.exp(
        -np.maximum(t_syn - 2500, 0) / 2000)

env = BatchReactorEnv(data_path=DATA_PATH, dt=DT, initial_history_size=50)
s0  = env.reset()
Tr, Tj = float(s0[2]), float(s0[3])
ukf = UKFEstimator(reactor_ode=env._BR_plant, dt=DT,
                   x0=np.array([float(s0[0]), float(s0[1]), Tr, Tj]))

states_list = []
hc_list     = []
u1_list     = []
N_STEPS_ROLLOUT = min(len(T_ref_all) - 1, 3000)

for step in range(N_STEPS_ROLLOUT):
    Tr_ref = T_ref_all[step]
    e      = Tr - Tr_ref
    obs    = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj, e)
    Hc, u1 = agent.deterministic_action(obs)
    states_list.append(obs.copy())
    hc_list.append(Hc)
    u1_list.append(u1)
    Tr_n, Tj_n = env.step(u1, Hc)
    ukf.update(np.array([Tr_n, Tj_n]), u1=u1, Hc=Hc)
    Tr, Tj = Tr_n, Tj_n

states_arr = np.array(states_list, dtype=np.float32)
hc_arr     = np.array(hc_list,     dtype=np.float32)
u1_arr     = np.array(u1_list,     dtype=np.float32)
print(f"  Collected {len(states_arr)} state-action pairs.")

# ── SHAP wrapper functions ────────────────────────────────────────────────────
def predict_hc(X):
    """Predict Hc for a 2D array of observations."""
    X_t   = T.tensor(X, dtype=T.float32).to(agent.device)
    feats = agent.network.backbone(X_t)
    pre_t = agent.network.actor_head(feats)
    mu_sq = T.tanh(pre_t)
    raw   = mu_sq.detach().cpu().numpy()
    Hc_out = HC_LOW + (raw[:, 0] + 1.0) / 2.0 * (HC_HIGH - HC_LOW)
    return np.clip(Hc_out, HC_LOW, HC_HIGH)

def predict_u1(X):
    """Predict u1 for a 2D array of observations."""
    X_t   = T.tensor(X, dtype=T.float32).to(agent.device)
    feats = agent.network.backbone(X_t)
    pre_t = agent.network.actor_head(feats)
    mu_sq = T.tanh(pre_t)
    raw   = mu_sq.detach().cpu().numpy()
    u1_out = U1_LOW + (raw[:, 1] + 1.0) / 2.0 * (U1_HIGH - U1_LOW)
    return np.clip(u1_out, U1_LOW, U1_HIGH)

# ── Build background and explain sets ─────────────────────────────────────────
np.random.seed(42)
idx_bg  = np.random.choice(len(states_arr), size=N_BACKGROUND, replace=False)
idx_exp = np.random.choice(len(states_arr), size=N_EXPLAIN,    replace=False)
background = states_arr[idx_bg]
X_explain  = states_arr[idx_exp]

# ── SHAP — Heater ─────────────────────────────────────────────────────────────
print(f"Computing SHAP values for Hc (background={N_BACKGROUND}, explain={N_EXPLAIN}) ...")
explainer_hc = shap.KernelExplainer(predict_hc, background)
shap_hc      = explainer_hc.shap_values(X_explain, nsamples=200, silent=True)
print("  Hc SHAP done.")

# ── SHAP — Coolant ────────────────────────────────────────────────────────────
print(f"Computing SHAP values for u1 ...")
explainer_u1 = shap.KernelExplainer(predict_u1, background)
shap_u1      = explainer_u1.shap_values(X_explain, nsamples=200, silent=True)
print("  u1 SHAP done.")

# ── Plot helpers ──────────────────────────────────────────────────────────────
COLORS = {
    'hc':   '#E63946',
    'u1':   '#457B9D',
    'pos':  '#2A9D8F',
    'neg':  '#E76F51',
    'grid': '#E5E5E5',
    'bg':   '#FAFAFA',
}

def plot_shap_summary(shap_vals, X, feature_names, title, fname, color):
    """Beeswarm-style SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    n_feat  = shap_vals.shape[1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order   = np.argsort(mean_abs)   # ascending → bottom to top

    for rank, fi in enumerate(order):
        sv  = shap_vals[:, fi]
        fv  = X[:, fi]
        # jitter y
        jitter = np.random.uniform(-0.25, 0.25, size=len(sv))
        # colour by feature value
        norm_fv = (fv - np.min(fv)) / (np.ptp(fv) + 1e-9)
        c_vals  = plt.cm.RdYlBu_r(norm_fv)
        ax.scatter(sv, rank + jitter, c=c_vals, s=8, alpha=0.55,
                   linewidths=0, zorder=3)
        ax.axhline(rank, color=COLORS['grid'], lw=0.5, zorder=1)

    ax.axvline(0, color='#888', lw=1.2, zorder=2)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=11)
    ax.set_xlabel('SHAP value  (impact on output)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='x', color=COLORS['grid'], lw=0.7, zorder=0)

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
    cbar.set_label('Feature value\n(low → high)', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_shap_bar(shap_vals, feature_names, title, fname, color):
    """Mean |SHAP| bar chart."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    bars = ax.barh(range(len(order)),
                   mean_abs[order],
                   color=color, alpha=0.82, edgecolor='white', linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=11)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='x', color=COLORS['grid'], lw=0.7)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, mean_abs[order]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=9, color='#444')

    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ── Generate plots ────────────────────────────────────────────────────────────
print("Generating SHAP plots ...")

plot_shap_summary(
    shap_hc, X_explain, FEATURE_NAMES,
    'SHAP Summary — Heater Current $H_c$ [mA]',
    'figures/shap_heater_summary.png', COLORS['hc']
)
plot_shap_summary(
    shap_u1, X_explain, FEATURE_NAMES,
    'SHAP Summary — Coolant Flow $u_1$ [L/min]',
    'figures/shap_coolant_summary.png', COLORS['u1']
)
plot_shap_bar(
    shap_hc, FEATURE_NAMES_SHORT,
    'Feature Importance (Mean |SHAP|) — Heater $H_c$',
    'figures/shap_heater_bar.png', COLORS['hc']
)
plot_shap_bar(
    shap_u1, FEATURE_NAMES_SHORT,
    'Feature Importance (Mean |SHAP|) — Coolant $u_1$',
    'figures/shap_coolant_bar.png', COLORS['u1']
)

# ── Combined bar comparison ───────────────────────────────────────────────────
mean_hc = np.abs(shap_hc).mean(axis=0)
mean_u1 = np.abs(shap_u1).mean(axis=0)
x       = np.arange(len(FEATURE_NAMES_SHORT))
w       = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.bar(x - w/2, mean_hc, w, label='Heater $H_c$',  color=COLORS['hc'], alpha=0.85, edgecolor='white')
ax.bar(x + w/2, mean_u1, w, label='Coolant $u_1$', color=COLORS['u1'], alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_NAMES_SHORT, fontsize=10)
ax.set_ylabel('Mean |SHAP value|', fontsize=11)
ax.set_title('Feature Importance Comparison: Heater vs Coolant', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', color=COLORS['grid'], lw=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('figures/shap_combined_bar.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved: figures/shap_combined_bar.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SHAP Feature Importance Summary")
print("="*55)
print(f"  {'Feature':<20}  {'|SHAP| Hc':>10}  {'|SHAP| u1':>10}")
print("-"*55)
order_hc = np.argsort(mean_hc)[::-1]
for i in order_hc:
    print(f"  {FEATURE_NAMES_SHORT[i]:<20}  {mean_hc[i]:10.5f}  {mean_u1[i]:10.5f}")
print("="*55)
print("\nAll SHAP plots saved to figures/")
