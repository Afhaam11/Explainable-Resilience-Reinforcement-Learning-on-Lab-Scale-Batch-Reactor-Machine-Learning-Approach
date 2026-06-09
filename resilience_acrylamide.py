
import numpy as np
from scipy import integrate as sci_int

# ── Constants ────────────────────────────────────────────────────────────────
MAT  = 80.0    # Maximum Allowable Temperature [deg C]
Tj0  = 42.47   # Initial jacket temperature    [deg C]
UA   = 33.3083 # Heat transfer coefficient     [W/K]

# T_SCALE for tracking resilience — MUST match training script
T_SCALE = 5.0  # deg C

def _Qh(T, Tj=Tj0):
    return UA * (T - Tj)

def _Qmax(Tj=Tj0):
    return UA * (MAT - Tj)

def resilience_acrylamide(args):
    """
    Compute heat-duty resilience R in [0, 1] per paper Appendix A.

    Parameters
    ----------
    args : tuple  (t_l, T_l, tinj_l, inj_duration, T0)
        t_l          : 1-D array of time values [min]
        T_l          : 1-D array of reactor temperatures [deg C]
        tinj_l       : [] if no brake applied, [t_brake] otherwise
        inj_duration : duration of brake action [min]
        T0           : initial reactor temperature [deg C]

    Returns
    -------
    Q_res : np.ndarray  — performance trajectory
    R     : float       — resilience scalar in [0, 1]
    """
    t_l, T_l, tinj_l, inj_duration, T0 = args

    if len(t_l) < 3 or len(T_l) < 3:
        return np.array([1.0]), 1.0
    if max(T_l) <= T_l[0] + 1e-3:
        return np.ones(max(len(t_l) - 2, 1)), 1.0

    # Phase boundaries
    ta_0 = 0
    if tinj_l and len(tinj_l) > 0 and max(T_l) < MAT:
        inj_end = tinj_l[0] + inj_duration
        cands   = np.where(t_l >= inj_end)[0]
        ta_end  = int(cands[0]) if len(cands) > 0 else len(t_l) - 1
    else:
        ta_end = int(np.where(T_l == max(T_l))[0][0])

    ta_end = max(ta_end, ta_0 + 4)
    ta_end = min(ta_end, len(t_l) - 1)
    ta = t_l[ta_0:ta_end - 1]
    Ta = T_l[ta_0:ta_end - 1]

    if len(ta) < 3:
        return np.array([1.0]), 1.0

    # Absorption performance (Eqs. A1-A4)
    m_abs = np.diff(Ta) / np.diff(ta)
    m_abs = np.where(m_abs <= 0, np.abs(m_abs) + 1e-8, m_abs)
    T_slice = T_l[ta_0:ta_end - 2]
    min_len = min(len(T_slice), len(m_abs))

    if min_len < 2:
        return np.array([1.0]), 1.0

    T_slice = T_slice[:min_len]
    m_abs   = m_abs[:min_len]

    PTTF = (MAT - T_slice) / m_abs
    PTTF = np.where((PTTF <= 0) | np.isinf(PTTF) | np.isnan(PTTF), 1e-6, PTTF)
    lambda_0 = 1.0 / PTTF

    Qmax = _Qmax()
    Qh   = _Qh(T_slice)
    ratio = np.clip(Qh / Qmax, 0.0, 10.0)
    lambda_abs = lambda_0 * np.exp(ratio)

    ta_int = ta[:min_len]
    if len(lambda_abs) < 2 or len(ta_int) < 2:
        return np.array([1.0]), 1.0

    i_abs = sci_int.cumulative_trapezoid(lambda_abs, ta_int)
    Q_abs = np.exp(-i_abs)

    # Recovery performance (Eqs. A5-A7)
    tr_end = len(t_l) - 1
    if ta_end >= tr_end:
        Q_rec = np.array([])
    elif max(T_l) >= MAT:
        Q_rec = np.zeros(tr_end - ta_end)
    else:
        rec_T = T_l[ta_end:tr_end]
        rec_t = t_l[ta_end:tr_end]
        if len(rec_T) < 3:
            Q_rec = np.array([])
        else:
            m_rec = np.diff(rec_T) / np.diff(rec_t)
            m_rec = np.where(m_rec == 0, 1e-8, m_rec)
            T_rec = rec_T[:-1]
            ml2   = min(len(T_rec), len(m_rec))
            T_rec = T_rec[:ml2]; m_rec = m_rec[:ml2]
            PTTR  = (T0 - T_rec) / m_rec
            PTTR  = np.where((PTTR <= 0) | np.isinf(PTTR) | np.isnan(PTTR), 1e-6, PTTR)
            mu_rec = 1.0 / PTTR
            t_r    = rec_t[:ml2]
            if len(mu_rec) < 2 or len(t_r) < 2:
                Q_rec = np.array([])
            else:
                i_rec = sci_int.cumulative_trapezoid(mu_rec, t_r)
                Q_rec = Q_abs[-1] + (1.0 - Q_abs[-1]) * (1.0 - np.exp(-i_rec))

    Q_res = (np.concatenate((Q_abs, Q_rec)) if len(Q_rec) > 0 else Q_abs)
    Q_res = np.clip(Q_res, 0.0, 1.0)

    if len(Q_res) < 2:
        return Q_res, float(np.mean(Q_res))

    if min(Q_res) < 0.95:
        t_res_idx = max(int(np.where(Q_res < 0.95)[0][-1]) - 1, 1)
    else:
        t_res_idx = max(int(np.where(Q_res <= 1.0)[0][-1]), 1)

    t_res_idx = min(t_res_idx, len(t_l) - 1)
    t_window  = t_l[:t_res_idx]

    if len(t_window) < 2 or len(Q_res[:t_res_idx]) < 2:
        return Q_res, float(np.clip(np.mean(Q_res), 0.0, 1.0))

    i_Q = np.trapezoid(Q_res[:t_res_idx], t_window)
    dt  = t_window[-1] - t_window[0]
    R   = float(np.clip(i_Q / dt, 0.0, 1.0)) if dt > 1e-9 else float(np.mean(Q_res[:t_res_idx]))

    return Q_res, R


def tracking_resilience(Tr_arr, T_ref_arr, t_arr, T_scale=None):
    """
    Tracking resilience using paper Eq.1 with tracking performance function.

    Paper Eq.1: R = integral(Q_p(t) dt) / integral(dt)
    Q_p(t) = exp(-|T_r(t) - T_ref(t)| / T_scale)

    CORRECTED: T_scale defaults to module-level T_SCALE=5.0°C.
    This produces R_track >= 0.95 for well-tracking policies.

    Parameters
    ----------
    Tr_arr    : reactor temperature trajectory [deg C]
    T_ref_arr : reference trajectory [deg C]
    t_arr     : time array [min]
    T_scale   : normalization temperature [deg C] (default: T_SCALE=5.0°C)

    Returns
    -------
    R_track : float  tracking resilience in [0, 1]
    Q_p     : np.ndarray  per-step performance
    """
    if T_scale is None:
        T_scale = T_SCALE

    n    = min(len(Tr_arr), len(T_ref_arr), len(t_arr))
    Q_p  = np.exp(-np.abs(Tr_arr[:n] - T_ref_arr[:n]) / T_scale)

    if n < 2:
        return float(np.mean(Q_p)), Q_p

    integral = np.trapezoid(Q_p, t_arr[:n])
    dt       = t_arr[n-1] - t_arr[0]
    R_track  = float(np.clip(integral / dt, 0.0, 1.0)) if dt > 1e-9 \
               else float(np.mean(Q_p))

    return R_track, Q_p
