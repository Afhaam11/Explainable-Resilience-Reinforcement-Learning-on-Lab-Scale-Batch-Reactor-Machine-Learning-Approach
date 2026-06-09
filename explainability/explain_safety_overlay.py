
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from matplotlib.collections import LineCollection

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_agent import PPOAgent
from batch_reactor_env import BatchReactorEnv
from ukf_estimator import UKFEstimator

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

DATA_PATH = 'Opeloop_HFc_TrTj.csv'
TRAJ_CSV  = 'Trajectory2.csv'

DT = 0.5

I_MAX = 4.5e-5
M_MAX = 0.7034

TR_MIN = 40.0
TR_MAX = 80.0

TJ_MIN = 27.0
TJ_MAX = 70.0

E_SCALE = 20.0

HC_LOW  = 8.0
HC_HIGH = 20.0

U1_LOW  = 0.0
U1_HIGH = 0.7

MAT = 80.0

# MDC worst-case cooling configuration
U1_MDC = U1_HIGH
HC_MDC = HC_LOW

# MDC grid
M_MDC_VALUES = np.linspace(0.05, M_MAX, 40)

I_NOMINAL  = 1.5e-5
TJ_NOMINAL = 42.0

HORIZON_MIN = 60.0


# ─────────────────────────────────────────────────────────
# Observation builder
# ─────────────────────────────────────────────────────────

def build_obs(I_hat, M_hat, Tr, Tj, e):

    return np.array([
        np.clip(I_hat / I_MAX, 0., 1.),
        np.clip(M_hat / M_MAX, 0., 1.),
        np.clip((Tr - TR_MIN)/(TR_MAX-TR_MIN), 0., 1.),
        np.clip((Tj - TJ_MIN)/(TJ_MAX-TJ_MIN), 0., 1.),
        np.clip(e / E_SCALE, -1., 1.)
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Reactor model for MDC computation
# ─────────────────────────────────────────────────────────

env_mdc = BatchReactorEnv(data_path=DATA_PATH,
                          dt=DT,
                          initial_history_size=50)


def max_temp_with_full_cooling(Tr0, M0):

    x0 = np.array([I_NOMINAL, M0, Tr0, TJ_NOMINAL])

    sol = solve_ivp(
        env_mdc._BR_plant,
        [0, HORIZON_MIN],
        x0,
        args=(U1_MDC, HC_MDC),
        rtol=1e-5,
        atol=1e-7,
        max_step=0.5
    )

    return np.max(sol.y[2])


def find_mdc_temperature(M0):

    Tr_lo = TR_MIN
    Tr_hi = MAT - 0.5

    if max_temp_with_full_cooling(Tr_lo, M0) >= MAT:
        return None

    if max_temp_with_full_cooling(Tr_hi, M0) < MAT:
        return Tr_hi

    try:
        return brentq(
            lambda Tr: max_temp_with_full_cooling(Tr, M0) - MAT,
            Tr_lo,
            Tr_hi,
            xtol=0.5,
            maxiter=20
        )
    except:
        return None


# ─────────────────────────────────────────────────────────
# Compute MDC curve
# ─────────────────────────────────────────────────────────

print("Computing MDC curve...")

mdc_T = []
mdc_M = []

for M0 in M_MDC_VALUES:

    T_val = find_mdc_temperature(M0)

    if T_val is not None:

        mdc_T.append(T_val)
        mdc_M.append(M0)

mdc_T = np.array(mdc_T)
mdc_M = np.array(mdc_M)

print("MDC curve computed.")


# ─────────────────────────────────────────────────────────
# PPO rollout trajectory
# ─────────────────────────────────────────────────────────

print("Running PPO rollout...")

try:
    T_ref_all = pd.read_csv(TRAJ_CSV)['x_Traject'].values
except:
    t_syn = np.arange(7201) * DT
    T_ref_all = 45 + 20*(1-np.exp(-t_syn/500))*np.exp(
        -np.maximum(t_syn-2500,0)/2000
    )

agent = PPOAgent(obs_dim=5,
                 act_dim=2,
                 chkpt_dir='',
                 name='ppo_reactor')

agent.load()
agent.network.eval()

env = BatchReactorEnv(DATA_PATH, DT, 50)

s0 = env.reset()

Tr = float(s0[2])
Tj = float(s0[3])

ukf = UKFEstimator(env._BR_plant,
                   DT,
                   x0=np.array([s0[0], s0[1], Tr, Tj]))

traj_Tr = []
traj_M  = []
traj_t  = []

for step in range(min(3000, len(T_ref_all)-1)):

    traj_Tr.append(Tr)
    traj_M.append(ukf.M_hat)
    traj_t.append(step*DT)

    e = Tr - T_ref_all[step]

    obs = build_obs(ukf.I_hat,
                    ukf.M_hat,
                    Tr,
                    Tj,
                    e)

    Hc, u1 = agent.deterministic_action(obs)

    Tr, Tj = env.step(u1, Hc)

    ukf.update(np.array([Tr, Tj]), u1, Hc)


traj_Tr = np.array(traj_Tr)
traj_M  = np.array(traj_M)
traj_t  = np.array(traj_t)


# ─────────────────────────────────────────────────────────
# PROFESSIONAL PUBLICATION-QUALITY PLOT
# ─────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(9,7))

# Safe / unsafe zones

ax.fill_betweenx(
    mdc_M,
    mdc_T,
    MAT,
    color="#f4a6a6",
    alpha=0.35,
    label="Unsafe region"
)

ax.fill_betweenx(
    mdc_M,
    TR_MIN,
    mdc_T,
    color="#b8e0d2",
    alpha=0.35,
    label="Safe region"
)

# MDC curve

ax.plot(
    mdc_T,
    mdc_M,
    color="#d95f02",
    linewidth=2.2,
    label="MDC boundary (60 min horizon)"
)

# MAT line

ax.axvline(
    MAT,
    linestyle="--",
    linewidth=1.8,
    color="#b22222",
    label="MAT = 80°C"
)


# Gradient trajectory line

points = np.array([traj_Tr, traj_M]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(
    segments,
    cmap="plasma",
    linewidth=2.2
)

lc.set_array(traj_t[:-1])
ax.add_collection(lc)
ax.autoscale()

# Start / End markers

ax.scatter(traj_Tr[0],
           traj_M[0],
           s=120,
           color="green",
           label="Start")

ax.scatter(traj_Tr[-1],
           traj_M[-1],
           s=140,
           marker="*",
           color="gold",
           label="End")


# Colorbar

cbar = fig.colorbar(lc, ax=ax)
cbar.set_label("Time [min]")


# Labels

ax.set_xlabel("Reactor Temperature $T_r$ [°C]",
              fontsize=12)

ax.set_ylabel("Monomer Concentration $\\hat{M}$ [mol/L]",
              fontsize=12)

ax.set_title(
    "Trajectory with Modified Dynamic Condition Safety Boundary",
    fontsize=14,
    fontweight="bold"
)


# Safe fraction annotation

from scipy.interpolate import interp1d

interp_func = interp1d(mdc_M,
                       mdc_T,
                       fill_value="extrapolate")

safe_fraction = np.mean(
    traj_Tr <= interp_func(traj_M)
)

ax.text(
    0.02,
    0.04,
    f"Safe fraction: {safe_fraction:.1%}",
    transform=ax.transAxes,
    fontsize=11
)


ax.legend(fontsize=10)

plt.tight_layout()

plt.savefig(
    "figures/trajectory_MDC_overlay.png",
    dpi=300
)

plt.close()

print("Saved: figures/trajectory_MDC_overlay.png")
print("Done.")