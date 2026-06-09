
import numpy as np


class UKFEstimator:
    """
    UKF state estimator for [I, M, Tr, Tj] with measurements [Tr, Tj].

    Parameters
    ----------
    reactor_ode : callable  f(t, x, u1, Hc) → list[float] of 4 derivatives
    dt          : float     sample time [min]
    x0          : array(4)  initial state [I, M, Tr, Tj]
    P0          : array(4,4) or None
    Q           : array(4,4) or None  process noise
    R           : array(2,2) or None  measurement noise
    alpha       : float     UKF spread (paper: 0.5)
    beta        : float     distribution prior (paper: 0)
    kappa       : float     secondary scaling (paper: 0.5)
    """

    Cm = np.array([[0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

    def __init__(
        self,
        reactor_ode,
        dt,
        x0    = None,
        P0    = None,
        Q     = None,
        R     = None,
        alpha = 0.5,
        beta  = 0.0,
        kappa = 0.5,
    ):
        self.f   = reactor_ode
        self.dt  = float(dt)
        self.n   = 4
        self.p   = 2

        lam      = alpha ** 2 * (self.n + kappa) - self.n
        self.lam = lam
        c        = self.n + lam

        self.Wm      = np.full(2 * self.n + 1, 1.0 / (2 * c))
        self.Wc      = np.full(2 * self.n + 1, 1.0 / (2 * c))
        self.Wm[0]   = lam / c
        self.Wc[0]   = lam / c + (1 - alpha ** 2 + beta)

        if P0 is None:
            P0 = np.diag([5.2e-6, 5.2e-4, 0.52, 0.52])
        if Q is None:
            Q  = np.diag([5.2e-6, 5.2e-4, 1e-4, 1e-4])
        if R is None:
            R  = np.diag([0.01, 0.01])

        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        if x0 is None:
            x0 = np.array([4.5e-5, 0.7034, 55.0, 36.8])
        self.x_hat = np.array(x0, dtype=float)
        self.P     = np.array(P0, dtype=float)

    def update(
        self,
        y_measured: np.ndarray,
        u1:  float,
        Hc:  float,
    ) -> np.ndarray:
        """One UKF predict-correct cycle (Eqs 12-24)."""
        chi      = self._sigma_points(self.x_hat, self.P)
        chi_pred = np.zeros_like(chi)
        for i in range(2 * self.n + 1):
            chi_pred[:, i] = self._euler_step(chi[:, i], u1, Hc)

        x_pred = chi_pred @ self.Wm
        dx     = chi_pred - x_pred[:, None]
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            P_pred += self.Wc[i] * np.outer(dx[:, i], dx[:, i])

        y_pts  = self.Cm @ chi_pred
        y_pred = y_pts @ self.Wm
        dy     = y_pts - y_pred[:, None]
        P_ee   = self.R.copy()
        for i in range(2 * self.n + 1):
            P_ee += self.Wc[i] * np.outer(dy[:, i], dy[:, i])

        P_xe = np.zeros((self.n, self.p))
        for i in range(2 * self.n + 1):
            P_xe += self.Wc[i] * np.outer(dx[:, i], dy[:, i])

        try:
            K = P_xe @ np.linalg.inv(P_ee)
        except np.linalg.LinAlgError:
            K = P_xe @ np.linalg.pinv(P_ee)

        gamma      = np.array(y_measured, dtype=float) - y_pred
        self.x_hat = x_pred + K @ gamma
        self.P     = P_pred - K @ P_ee @ K.T
        self.P     = 0.5 * (self.P + self.P.T) + 1e-10 * np.eye(self.n)

        self.x_hat[0] = max(self.x_hat[0], 0.0)
        self.x_hat[1] = max(self.x_hat[1], 0.0)

        return self.x_hat.copy()

    def _sigma_points(self, mean, P):
        c   = self.n + self.lam
        chi = np.empty((self.n, 2 * self.n + 1))
        chi[:, 0] = mean
        try:
            S = np.linalg.cholesky(c * P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(c * P + 1e-8 * np.eye(self.n))
        for i in range(self.n):
            chi[:, i + 1]          = mean + S[:, i]
            chi[:, self.n + i + 1] = mean - S[:, i]
        return chi

    def _euler_step(self, x, u1, Hc):
        x_phys    = x.copy()
        x_phys[0] = max(x_phys[0], 0.0)
        x_phys[1] = max(x_phys[1], 0.0)
        dxdt      = np.array(self.f(0, x_phys, u1, Hc), dtype=float)
        x_next    = x_phys + dxdt * self.dt
        x_next[0] = max(x_next[0], 0.0)
        x_next[1] = max(x_next[1], 0.0)
        return x_next

    def reset(self, x0=None, P0=None):
        if x0 is not None:
            self.x_hat = np.array(x0, dtype=float)
        if P0 is not None:
            self.P = np.array(P0, dtype=float)
        self.x_hat[0] = max(self.x_hat[0], 0.0)
        self.x_hat[1] = max(self.x_hat[1], 0.0)

    @property
    def I_hat(self) -> float:
        return float(self.x_hat[0])

    @property
    def M_hat(self) -> float:
        return float(self.x_hat[1])

    @property
    def Tr_hat(self) -> float:
        return float(self.x_hat[2])

    @property
    def Tj_hat(self) -> float:
        return float(self.x_hat[3])