import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

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
T_SCALE   = 5.0      # for Qp(t) = exp(-|e|/T_SCALE)
MAT_MARGIN = 10.0    # barrier activation zone [°C]

# Style
BG        = '#FAFAF8'
C_TR      = '#1A5276'
C_REF     = '#117A65'
C_HC      = '#C0392B'
C_U1      = '#2874A6'
C_QP      = '#884EA0'
C_MAT     = '#E74C3C'
C_BARRIER = '#F5CBA7'
C_GRID    = '#E8E8E8'

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

# ── Trajectory reference ──────────────────────────────────────────────────────
try:
    T_ref_all = pd.read_csv(TRAJ_CSV)['x_Traject'].values
except FileNotFoundError:
    t_syn = np.arange(7201) * DT
    T_ref_all = 45.0 + 20.0*(1 - np.exp(-t_syn/500))*np.exp(
        -np.maximum(t_syn-2500,0)/2000)

# ── Full deterministic rollout ────────────────────────────────────────────────
print("Running full deterministic rollout ...")
env = BatchReactorEnv(data_path=DATA_PATH, dt=DT, initial_history_size=50)
s0  = env.reset()
Tr, Tj = float(s0[2]), float(s0[3])
ukf = UKFEstimator(reactor_ode=env._BR_plant, dt=DT,
                   x0=np.array([float(s0[0]), float(s0[1]), Tr, Tj]))

(Tr_list, Tj_list, Tref_list, Hc_list, u1_list,
 e_list, M_list, I_list, t_list, Ptr_list) = [[] for _ in range(10)]

N = len(T_ref_all) - 1
for step in range(N):
    Tr_ref = T_ref_all[step]
    e      = Tr - Tr_ref
    obs    = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj, e)
    Hc, u1 = agent.deterministic_action(obs)

    Tr_list.append(Tr);   Tj_list.append(Tj)
    Tref_list.append(Tr_ref);  Hc_list.append(Hc); u1_list.append(u1)
    e_list.append(e);     M_list.append(ukf.M_hat); I_list.append(ukf.I_hat)
    t_list.append(step * DT); Ptr_list.append(float(np.trace(ukf.P)))

    Tr_n, Tj_n = env.step(u1, Hc)
    ukf.update(np.array([Tr_n, Tj_n]), u1=u1, Hc=Hc)
    Tr, Tj = Tr_n, Tj_n

t   = np.array(t_list)
Tr  = np.array(Tr_list);   Tj  = np.array(Tj_list)
Ref = np.array(Tref_list); Hc  = np.array(Hc_list); u1 = np.array(u1_list)
E   = np.array(e_list);    M   = np.array(M_list);   I  = np.array(I_list)

# ── Resilience metric ──────────────────────────────────────────────────────────
Qp       = np.exp(-np.abs(E) / T_SCALE)
cum_Qp   = np.cumsum(Qp) / (np.arange(len(Qp)) + 1)
dt_arr   = np.diff(t, prepend=0)
R_track  = float(np.trapezoid(Qp, t) / max(t[-1] - t[0], 1e-9))
R_running = np.array([float(np.trapezoid(Qp[:k+1], t[:k+1]) / max(t[k]-t[0],1e-9))
                      for k in range(0, len(t), max(1, len(t)//200))])
t_running = t[np.arange(0, len(t), max(1, len(t)//200))]

print(f"  Steps: {len(t)} | R_track: {R_track:.4f} | "
      f"Peak Tr: {Tr.max():.2f}°C | "
      f"MAT breaches: {int(np.sum(Tr >= MAT))}")

# ── Helper for axis decoration ────────────────────────────────────────────────
def decorate(ax, ylabel, title=None, legend=True):
    ax.set_facecolor(BG)
    ax.grid(color=C_GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
    if legend:
        ax.legend(fontsize=9, loc='best')
    ax.set_xlim([t[0], t[-1]])

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Trajectory tracking
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.patch.set_facecolor(BG)
fig.suptitle(f'PPO Controller — Full Trajectory Tracking\n'
             f'R_track = {R_track:.4f}  |  Peak Tr = {Tr.max():.2f}°C  |  '
             f'Steps = {len(t)}',
             fontsize=12, fontweight='bold')

ax0 = axes[0]
ax0.axhspan(MAT - MAT_MARGIN, MAT, alpha=0.12, color=C_BARRIER, zorder=1,
            label=f'Barrier zone ({MAT_MARGIN}°C)')
ax0.axhline(MAT, color=C_MAT, lw=1.5, linestyle='--', zorder=5,
            label=f'MAT = {MAT}°C')
ax0.plot(t, Ref, color=C_REF, lw=2.0, linestyle='--', alpha=0.85, label='$T_{ref}$', zorder=3)
ax0.plot(t, Tr,  color=C_TR,  lw=1.5, alpha=0.90, label='$T_r$', zorder=4)
ax0.fill_between(t, Ref, Tr, alpha=0.10, color=C_TR, label='Tracking error')
ax0.set_ylabel('Temperature [°C]', fontsize=11)
decorate(ax0, 'Temperature [°C]', 'Reactor Temperature vs Setpoint')

ax1 = axes[1]
ax1.plot(t, np.abs(E), color='#E67E22', lw=1.2, alpha=0.85, label='|e(t)| = |Tr − Tref|')
ax1.axhline(np.mean(np.abs(E)), color='#E67E22', lw=1.0, linestyle=':',
            alpha=0.7, label=f'Mean |e| = {np.mean(np.abs(E)):.3f}°C')
ax1.set_ylabel('|Error| [°C]', fontsize=11)
decorate(ax1, '|Error| [°C]', 'Tracking Error')

ax2 = axes[2]
ax2.plot(t, Tj, color='#1ABC9C', lw=1.2, label='$T_j$ (jacket)')
ax2.set_ylabel('Jacket Temp [°C]', fontsize=11)
ax2.set_xlabel('Time [min]', fontsize=11)
decorate(ax2, 'Jacket Temp [°C]', 'Jacket Temperature')

plt.tight_layout()
plt.savefig('figures/trajectory_tracking.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved: figures/trajectory_tracking.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Control signals
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.patch.set_facecolor(BG)
fig.suptitle('PPO Control Signals', fontsize=12, fontweight='bold')

axes[0].plot(t, Hc, color=C_HC, lw=1.0, alpha=0.85, label='$H_c$ heater')
axes[0].axhline(np.mean(Hc), color=C_HC, lw=1.0, linestyle=':',
                label=f'Mean = {np.mean(Hc):.2f} mA')
axes[0].axhline((HC_LOW+HC_HIGH)/2, color='#888', lw=0.8, linestyle='--',
                alpha=0.5, label=f'Midpoint = {(HC_LOW+HC_HIGH)/2:.0f} mA')
axes[0].set_ylim([HC_LOW - 0.5, HC_HIGH + 0.5])
decorate(axes[0], '$H_c$ [mA]', 'Heater Current')

axes[1].plot(t, u1, color=C_U1, lw=1.0, alpha=0.85, label='$u_1$ coolant')
axes[1].axhline(np.mean(u1), color=C_U1, lw=1.0, linestyle=':',
                label=f'Mean = {np.mean(u1):.3f} L/min')
axes[1].set_ylim([U1_LOW - 0.02, U1_HIGH + 0.05])
decorate(axes[1], '$u_1$ [L/min]', 'Coolant Flow Rate')

# Normalised usage comparison
hc_norm = (Hc - HC_LOW) / (HC_HIGH - HC_LOW)
u1_norm = u1 / U1_HIGH
axes[2].plot(t, hc_norm, color=C_HC, lw=0.9, alpha=0.7, label='$H_c^{norm}$')
axes[2].plot(t, u1_norm, color=C_U1, lw=0.9, alpha=0.7, label='$u_1^{norm}$')
axes[2].axhline(0.5, color='#AAA', lw=0.8, linestyle='--', alpha=0.5)
axes[2].set_xlabel('Time [min]', fontsize=11)
decorate(axes[2], 'Normalised [0,1]', 'Normalised Actuator Usage')

plt.tight_layout()
plt.savefig('figures/control_signals.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved: figures/control_signals.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Resilience curve
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor(BG)
gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ax_qp    = fig.add_subplot(gs[0, :])
ax_cum   = fig.add_subplot(gs[1, 0])
ax_hist  = fig.add_subplot(gs[1, 1])

fig.suptitle(f'Tracking Resilience  (T_scale={T_SCALE}°C)\n'
             f'$R_{{track}} = {R_track:.4f}$',
             fontsize=12, fontweight='bold')

# Qp(t)
ax_qp.plot(t, Qp, color=C_QP, lw=1.0, alpha=0.55, label='$Q_p(t)$')
ax_qp.plot(t, pd.Series(Qp).rolling(50, min_periods=1).mean(),
           color=C_QP, lw=2.0, label='$Q_p$ (50-step mean)')
ax_qp.axhline(0.95, color='#2ECC71', lw=1.0, linestyle='--',
              label='$Q_p = 0.95$')
ax_qp.fill_between(t, 0, Qp, alpha=0.12, color=C_QP)
ax_qp.set_ylim([0, 1.05]); ax_qp.set_xlim([t[0], t[-1]])
ax_qp.set_xlabel('Time [min]', fontsize=10)
ax_qp.set_ylabel('$Q_p(t)$', fontsize=11)
ax_qp.set_title('Performance Function $Q_p(t) = \\exp(-|e|/T_{scale})$',
                fontsize=11, fontweight='bold')
ax_qp.legend(fontsize=9)
ax_qp.grid(color=C_GRID, lw=0.6)
ax_qp.set_facecolor(BG)

# Cumulative R_track
ax_cum.plot(t_running, R_running, color=C_QP, lw=2.0)
ax_cum.axhline(R_track, color='#2ECC71', lw=1.0, linestyle='--',
               label=f'Final R = {R_track:.4f}')
ax_cum.axhline(0.95, color='#E74C3C', lw=1.0, linestyle=':',
               label='R = 0.95 target')
ax_cum.set_ylim([0, 1.05])
ax_cum.set_xlabel('Time [min]', fontsize=10)
ax_cum.set_ylabel('Cumulative $R_{track}$', fontsize=11)
ax_cum.set_title('Running Resilience', fontsize=11, fontweight='bold')
ax_cum.legend(fontsize=8)
ax_cum.grid(color=C_GRID, lw=0.6)
ax_cum.set_facecolor(BG)

# Qp histogram
ax_hist.hist(Qp, bins=40, color=C_QP, alpha=0.75, edgecolor='white', lw=0.5)
ax_hist.axvline(np.mean(Qp), color='#E74C3C', lw=1.5, linestyle='--',
                label=f'Mean = {np.mean(Qp):.3f}')
ax_hist.axvline(np.median(Qp), color='#2ECC71', lw=1.5, linestyle=':',
                label=f'Median = {np.median(Qp):.3f}')
ax_hist.set_xlabel('$Q_p$ value', fontsize=10)
ax_hist.set_ylabel('Count', fontsize=10)
ax_hist.set_title('Distribution of $Q_p(t)$', fontsize=11, fontweight='bold')
ax_hist.legend(fontsize=8)
ax_hist.grid(color=C_GRID, lw=0.6)
ax_hist.set_facecolor(BG)

plt.savefig('figures/resilience_curve.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved: figures/resilience_curve.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Phase portrait Tr vs M_hat coloured by time
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

# Gradient-coloured line
points = np.array([Tr, M]).T.reshape(-1, 1, 2)
segs   = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segs, cmap='plasma',
                    norm=plt.Normalize(t.min(), t.max()), lw=1.8, alpha=0.85)
lc.set_array(t[:-1])
ax.add_collection(lc)
ax.autoscale()

cbar = fig.colorbar(lc, ax=ax, pad=0.02)
cbar.set_label('Time [min]', color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.axvline(MAT, color='#FF4444', lw=1.5, linestyle='--', label=f'MAT={MAT}°C')
ax.scatter([Tr[0]], [M[0]], s=100, color='#44FF88', zorder=5, label='Start', marker='o')
ax.scatter([Tr[-1]], [M[-1]], s=100, color='#FFFF44', zorder=5, label='End',   marker='*')
ax.set_xlabel(r'$T_r$ [°C]',              fontsize=12, color='white')
ax.set_ylabel(r'$\hat{M}$ [mol/L]',       fontsize=12, color='white')
ax.set_title(r'Phase Portrait: $T_r$ vs $\hat{M}$ (time coloured)',
             fontsize=13, fontweight='bold', color='white')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor('#444')
ax.legend(fontsize=9, facecolor='#1E2128', edgecolor='#555', labelcolor='white')
plt.tight_layout()
plt.savefig('figures/phase_portrait.png', dpi=150,
            bbox_inches='tight', facecolor='#0D1117')
plt.close()
print("  Saved: figures/phase_portrait.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Actuator usage histogram
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Actuator Usage Distribution', fontsize=13, fontweight='bold')

for ax, data, label, color, lo, hi, unit in [
    (axes[0], Hc, '$H_c$ Heater Current', C_HC, HC_LOW, HC_HIGH, 'mA'),
    (axes[1], u1, '$u_1$ Coolant Flow',   C_U1, U1_LOW, U1_HIGH, 'L/min'),
]:
    ax.set_facecolor(BG)
    ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor='white', lw=0.5)
    ax.axvline(np.mean(data),  color='#333', lw=1.5, linestyle='--',
               label=f'Mean  = {np.mean(data):.3f}')
    ax.axvline(np.median(data), color='#777', lw=1.5, linestyle=':',
               label=f'Median = {np.median(data):.3f}')
    ax.axvline(lo, color='#E74C3C', lw=1.0, linestyle='-.', alpha=0.6,
               label=f'Min = {lo}')
    ax.axvline(hi, color='#E74C3C', lw=1.0, linestyle='-.', alpha=0.6,
               label=f'Max = {hi}')
    ax.set_xlabel(f'{label} [{unit}]', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figures/actuator_histogram.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved: figures/actuator_histogram.png")

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ROLLOUT SUMMARY STATISTICS")
print("="*60)
print(f"  Total steps          : {len(t)}")
print(f"  Duration             : {t[-1]:.1f} min")
print(f"  R_track (final)      : {R_track:.4f}")
print(f"  Mean |error|         : {np.mean(np.abs(E)):.4f} °C")
print(f"  Max |error|          : {np.max(np.abs(E)):.4f} °C")
print(f"  Peak Tr              : {Tr.max():.3f} °C")
print(f"  MAT breaches         : {int(np.sum(Tr >= MAT))}")
print(f"  Steps in barrier zone: {int(np.sum(Tr >= (MAT - MAT_MARGIN)))} "
      f"({100*np.mean(Tr >= (MAT-MAT_MARGIN)):.1f}%)")
print(f"  Hc mean / std        : {np.mean(Hc):.3f} / {np.std(Hc):.3f} mA")
print(f"  u1 mean / std        : {np.mean(u1):.4f} / {np.std(u1):.4f} L/min")
print(f"  Hc at HC_LOW {HC_LOW:.0f} mA : "
      f"{100*np.mean(Hc <= HC_LOW+0.05):.1f}% of steps")
print(f"  u1 at U1_MAX {U1_HIGH:.2f}   : "
      f"{100*np.mean(u1 >= U1_HIGH-0.01):.1f}% of steps")
print("="*60)
print("\nAll rollout plots saved to figures/")
