
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class BatchReactorEnv:
    """Acrylamide batch reactor ODE plant model (Shettigar et al. 2021)."""

    def __init__(
        self,
        data_path:             str = 'Opeloop_HFc_TrTj.csv',
        dt:                    float = 0.5,
        initial_history_size:  int   = 50,
    ):
        # Physical constants (Table 2)
        self.Ad       = 4.4e16
        self.Ed       = 140.06e3
        self.Ap       = 1.7e11 / 60
        self.Ep       = 16.9e3 / 0.239
        self.deltaHp  = -82.2e3
        self.UA       = 33.3083
        self.V        = 0.5
        self.Tc       = 27.0
        self.Cpc      = 4.184
        self.R        = 8.3145
        self.alpha    = 1.212827
        self.beta1    = 0.000267
        self.epsilon  = 0.5
        self.theta    = 1.25
        self.m1       = 450.0;  self.cp1 = 4.184
        self.cp2      = 187.0
        self.cp3      = 110.58
        self.cp4      = 84.95
        self.m5       = 220.0;  self.cp5 = 0.49
        self.m6       = 7900.0; self.cp6 = 0.49
        self.mjCpj    = (18 * 4.184) + (240 * 0.49)
        self.Qs       = 12.41e-2
        self.M0       = 0.7034
        self.I0       = 4.5e-3
        self._Hc_min  = 4.0
        self._Hc_max  = 20.0
        self._Pw_max  = 1500.0

        self.dt                  = dt
        self.initial_history_size = initial_history_size

        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Warning: '{data_path}' not found. Using synthetic ICs.")
            self.df = pd.DataFrame({
                'Jacket Temperature':  np.linspace(36.8, 40.0, 100),
                'Reactor Temperature': np.linspace(45.0, 47.0, 100),
            })

        self.state_history = None
        self.reset()

    def _normalize_heater_current(self, current_mA: float) -> float:
        current_mA = np.clip(current_mA, self._Hc_min, self._Hc_max)
        return ((current_mA - self._Hc_min) /
                (self._Hc_max - self._Hc_min)) * self._Pw_max

    def _BR_plant(
        self,
        t:  float,
        X0: np.ndarray,
        F:  float,
        Hc: float,
    ) -> list:
        """ODE RHS: state = [I, M, Tr, Tj], inputs = [F [L/min], Hc [mA]]."""
        I, M, Tr, Tj = X0
        F_conv  = F * 16.667
        Qc_val  = self._normalize_heater_current(Hc)
        k_d     = self.Ad * np.exp(-self.Ed / (self.R * (Tr + 273.15)))
        k_p     = self.Ap * np.exp(-self.Ep / (self.R * (Tr + 273.15)))
        Ri      = k_d * I
        Rp      = k_p * (I ** self.epsilon) * (M ** self.theta)
        mrCpr   = (self.m1 * self.cp1
                   + I * self.cp2 * self.V
                   + M * self.cp3 * self.V
                   + (self.M0 - M) * self.cp4 * self.V
                   + self.m5 * self.cp5
                   + self.m6 * self.cp6)
        Qpr     = self.alpha * (Tr - self.Tc) ** self.beta1
        dI_dt   = -Ri
        dM_dt   = -Rp
        dTr_dt  = (Rp * self.V * (-self.deltaHp)
                   - self.UA * (Tr - Tj)
                   + Qc_val + self.Qs - Qpr) / mrCpr
        dTj_dt  = (self.UA * (Tr - Tj)
                   - F_conv * self.Cpc * (Tj - self.Tc)) / self.mjCpj
        return [dI_dt, dM_dt, dTr_dt, dTj_dt]

    def reset(self) -> np.ndarray:
        """Reset to warm-up initial conditions. Returns initial state [I,M,Tr,Tj]."""
        t = self.initial_history_size
        self.state_history          = np.zeros((t, 4))
        self.state_history[:, 0]    = 4.5e-5
        self.state_history[:, 1]    = self.M0
        self.state_history[:, 2]    = self.df['Reactor Temperature'].values[:t]
        self.state_history[:, 3]    = self.df['Jacket Temperature'].values[:t]
        return self.state_history[-1]

    def step(self, u1: float, u2: float):
        """
        Advance by one dt.

        Parameters
        ----------
        u1 : float  coolant flow [L/min]
        u2 : float  heater current [mA]

        Returns
        -------
        (Tr, Tj) : tuple[float, float]  new reactor and jacket temperatures
        """
        current_state = self.state_history[-1]
        solution      = solve_ivp(
            self._BR_plant, [0, self.dt], current_state,
            method='RK45', args=(u1, u2), rtol=1e-6, atol=1e-8,
        )
        next_state     = solution.y[:, -1]
        next_state[0]  = max(next_state[0], 0.0)
        next_state[1]  = max(next_state[1], 0.0)
        self.state_history = np.vstack((self.state_history, next_state))
        return float(next_state[2]), float(next_state[3])

    @property
    def full_state(self) -> np.ndarray:
        return self.state_history[-1].copy()

    @property
    def measurements(self) -> np.ndarray:
        return self.state_history[-1, 2:4].copy()