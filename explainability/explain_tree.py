
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import (
    DecisionTreeRegressor,
    export_text,
    plot_tree,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
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
MAX_DEPTH  = 5        # keep tree interpretable
N_STEPS_ROLLOUT = 4000

FEATURE_NAMES = ['I_hat_norm', 'M_hat_norm', 'Tr_norm', 'Tj_norm', 'e_norm']
FEAT_LABELS   = [
    r'$\hat{I}_{norm}$', r'$\hat{M}_{norm}$',
    r'$T_r^{norm}$',     r'$T_j^{norm}$',
    r'$e_{norm}$',
]

BG = '#F7F7F7'
C_HC = '#C1121F'
C_U1 = '#2B7CB0'

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

# ── Rollout ───────────────────────────────────────────────────────────────────
print("Collecting rollout dataset ...")
try:
    T_ref_all = pd.read_csv(TRAJ_CSV)['x_Traject'].values
except FileNotFoundError:
    t_syn = np.arange(7201) * DT
    T_ref_all = 45.0 + 20.0 * (1 - np.exp(-t_syn / 500)) * \
                np.exp(-np.maximum(t_syn - 2500, 0) / 2000)

env = BatchReactorEnv(data_path=DATA_PATH, dt=DT, initial_history_size=50)
s0  = env.reset()
Tr, Tj = float(s0[2]), float(s0[3])
ukf = UKFEstimator(reactor_ode=env._BR_plant, dt=DT,
                   x0=np.array([float(s0[0]), float(s0[1]), Tr, Tj]))

X_data, y_hc, y_u1 = [], [], []
N = min(N_STEPS_ROLLOUT, len(T_ref_all) - 1)

for step in range(N):
    e   = Tr - T_ref_all[step]
    obs = build_obs(ukf.I_hat, ukf.M_hat, Tr, Tj, e)
    Hc, u1 = agent.deterministic_action(obs)
    X_data.append(obs.copy())
    y_hc.append(Hc)
    y_u1.append(u1)
    Tr_n, Tj_n = env.step(u1, Hc)
    ukf.update(np.array([Tr_n, Tj_n]), u1=u1, Hc=Hc)
    Tr, Tj = Tr_n, Tj_n

X  = np.array(X_data, dtype=np.float32)
yH = np.array(y_hc,   dtype=np.float32)
yU = np.array(y_u1,   dtype=np.float32)
print(f"  {len(X)} samples collected.")

# ── Train surrogate trees ──────────────────────────────────────────────────────
X_tr, X_te, yH_tr, yH_te, yU_tr, yU_te = \
    train_test_split(X, yH, yU, test_size=0.2, random_state=42)

print(f"Training surrogate trees (max_depth={MAX_DEPTH}) ...")
tree_hc = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42)
tree_u1 = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42)
tree_hc.fit(X_tr, yH_tr)
tree_u1.fit(X_tr, yU_tr)

def report(name, tree, X_te, y_te):
    pred = tree.predict(X_te)
    r2   = r2_score(y_te, pred)
    mae  = mean_absolute_error(y_te, pred)
    n_leaves = tree.get_n_leaves()
    print(f"  {name:10s}  R²={r2:.4f}  MAE={mae:.4f}  leaves={n_leaves}")
    return pred

pred_hc = report('Heater Hc', tree_hc, X_te, yH_te)
pred_u1 = report('Coolant u1', tree_u1, X_te, yU_te)

# ── Tree visualization ────────────────────────────────────────────────────────
def save_tree_plot(tree, feature_names, title, fname, color, unit):
    fig, ax = plt.subplots(figsize=(22, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        precision=3,
        ax=ax,
        fontsize=7,
        impurity=False,
    )
    # Tint filled boxes — matplotlib's plot_tree uses its own colors;
    # we add a title strip
    ax.set_title(f'{title}  (max_depth={MAX_DEPTH})', fontsize=14,
                 fontweight='bold', color='#222', pad=12)
    plt.tight_layout(pad=0.5)
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Saved: {fname}")

save_tree_plot(tree_hc, FEATURE_NAMES,
               'Surrogate Decision Tree — Heater Current $H_c$ [mA]',
               'figures/policy_tree_heater.png', C_HC, 'mA')
save_tree_plot(tree_u1, FEATURE_NAMES,
               'Surrogate Decision Tree — Coolant Flow $u_1$ [L/min]',
               'figures/policy_tree_coolant.png', C_U1, 'L/min')

# ── Feature importance comparison ─────────────────────────────────────────────
imp_hc = tree_hc.feature_importances_
imp_u1 = tree_u1.feature_importances_
x      = np.arange(len(FEATURE_NAMES))
w      = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.bar(x - w/2, imp_hc, w, label='Heater $H_c$',  color=C_HC, alpha=0.88, edgecolor='white', lw=0.8)
ax.bar(x + w/2, imp_u1, w, label='Coolant $u_1$', color=C_U1, alpha=0.88, edgecolor='white', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_NAMES, fontsize=11)
ax.set_ylabel('Gini Importance', fontsize=11)
ax.set_title('Surrogate Tree: Feature Importance Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', color='#DDD', lw=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('figures/tree_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/tree_feature_importance.png")

# ── Parity plots ──────────────────────────────────────────────────────────────
def parity_plot(y_true, y_pred, title, fname, color, unit):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, color=color, linewidths=0)
    lo = min(y_true.min(), y_pred.min()) - 0.05
    hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect fit')
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_xlabel(f'PPO policy output [{unit}]', fontsize=11)
    ax.set_ylabel(f'Surrogate tree prediction [{unit}]', fontsize=11)
    ax.set_title(f'{title}\nR²={r2:.4f}  MAE={mae:.4f} {unit}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(color='#DDD', lw=0.6)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

parity_plot(yH_te, pred_hc, 'Heater Surrogate Parity',
            'figures/tree_parity_heater.png', C_HC, 'mA')
parity_plot(yU_te, pred_u1, 'Coolant Surrogate Parity',
            'figures/tree_parity_coolant.png', C_U1, 'L/min')

# ── Rule extraction ───────────────────────────────────────────────────────────
def extract_rules_to_file(tree, feature_names, fname_txt, output_name,
                           X_data, y_data, output_unit):
    """
    Walk each leaf path to produce human-readable IF–THEN rules
    annotated with the physical output value and sample count.
    """
    tree_ = tree.tree_
    n_nodes     = tree_.node_count
    children_l  = tree_.children_left
    children_r  = tree_.children_right
    feature     = tree_.feature
    threshold   = tree_.threshold
    value       = tree_.value
    n_samples   = tree_.n_node_samples

    # ── Map from normalised feature values back to physical values ──
    PHYS_MIN = np.array([0.0,    0.0,   TR_MIN, TJ_MIN, -E_SCALE])
    PHYS_MAX = np.array([I_MAX,  M_MAX, TR_MAX, TJ_MAX,  E_SCALE])
    PHYS_UNITS = ['mol/L', 'mol/L', '°C', '°C', '°C']

    lines = [
        f"Surrogate Decision Tree Rules — {output_name}",
        f"max_depth={MAX_DEPTH}  |  R²={r2_score(y_data, tree.predict(X_data)):.4f}",
        "="*70,
        "",
        "Format:  IF <condition> THEN <output> ≈ <value> [samples=N]",
        "Features (normalised 0–1 unless e_norm in -1–1):",
    ]
    for i, fn in enumerate(feature_names):
        lo = PHYS_MIN[i]; hi = PHYS_MAX[i]
        lines.append(f"  {fn:15s} → physical [{lo:.3g}, {hi:.3g}] {PHYS_UNITS[i]}")
    lines += ["", "-"*70, ""]

    def traverse(node_id, conditions):
        is_leaf = (children_l[node_id] == children_r[node_id])
        if is_leaf:
            pred_val = float(value[node_id].ravel()[0])
            ns       = int(n_samples[node_id])
            rule = "IF " + ("\n   AND ".join(conditions) if conditions else "TRUE")
            rule += f"\n   THEN {output_name} ≈ {pred_val:.4f} {output_unit}  [samples={ns}]"
            lines.append(rule)
            lines.append("")
            return
        fi   = feature[node_id]
        thr  = threshold[node_id]
        fn   = feature_names[fi]
        # physical threshold
        phys_thr = PHYS_MIN[fi] + thr * (PHYS_MAX[fi] - PHYS_MIN[fi])
        pu = PHYS_UNITS[fi]
        cond_l = f"{fn} ≤ {thr:.4f}  (≈ {phys_thr:.4f} {pu})"
        cond_r = f"{fn} >  {thr:.4f}  (≈ {phys_thr:.4f} {pu})"
        traverse(children_l[node_id], conditions + [cond_l])
        traverse(children_r[node_id], conditions + [cond_r])

    traverse(0, [])

    with open(fname_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Saved: {fname_txt}")

extract_rules_to_file(tree_hc, FEATURE_NAMES,
                      'tree_rules_heater.txt',
                      'Hc', X, yH, 'mA')
extract_rules_to_file(tree_u1, FEATURE_NAMES,
                      'tree_rules_coolant.txt',
                      'u1', X, yU, 'L/min')

# ── Print top-level split summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("  SURROGATE TREE TOP-LEVEL SPLIT")
print("="*55)
for name, tree in [('Heater Hc', tree_hc), ('Coolant u1', tree_u1)]:
    fi  = tree.tree_.feature[0]
    thr = tree.tree_.threshold[0]
    print(f"  {name}: root split on '{FEATURE_NAMES[fi]}' <= {thr:.4f}")
print("="*55)
print("\nAll tree outputs saved.")
