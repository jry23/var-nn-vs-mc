# ar_garch.py
# AR(1)-GARCH(1,1)-t wrapper using the 'arch' library.
# Minimal, clean API: fit → forecast → simulate_paths

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np

try:
    import pandas as pd  # optional, only for type flexibility
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from arch.univariate import ARX, GARCH, StudentsT


ArrayLike = Union[np.ndarray, "pd.Series"]


@dataclass
class ARGARCHConfig:
    """Hyperparams for AR(1)-GARCH(1,1)-t."""
    ar_lags: int = 1  
    garch_p: int = 1        
    garch_q: int = 1        
    dist: str = "student_t"     
    enforce_stationarity: bool = True
    enforce_pos_var: bool = True
    # Fitting
    maxiter: int = 1000
    disp: str = "off"           # "off" to silence solver
    # Forecast/sim
    horizon: int = 20
    random_state: Optional[int] = None  # for simulate_paths reproducibility


class ARGARCHModel:
    """
    AR(1)-GARCH(1,1)-t model wrapper (arch library).

    API:
        m = ARGARCHModel().fit(returns)
        mu, var = m.forecast(horizon=20)
        paths = m.simulate_paths(horizon=20, n_paths=10000)
    """

    def __init__(self, cfg: Optional[ARGARCHConfig] = None):
        self.cfg = cfg or ARGARCHConfig()
        self._res = None  # arch.univariate.base.ARCHModelResult
        self._y = None    # training returns as np.ndarray

    # --------------------------
    # Utilities
    # --------------------------
    @staticmethod
    def _to_1d_array(x: ArrayLike) -> np.ndarray:
        if "pd" in globals() and pd is not None and isinstance(x, pd.Series):
            arr = x.to_numpy()
        else:
            arr = np.asarray(x)
        if arr.ndim != 1:
            arr = np.ravel(arr)
        return arr.astype(float)

    def is_fitted(self) -> bool:
        return self._res is not None

    # --------------------------
    # Fit
    # --------------------------
    def fit(self, returns: ArrayLike):
        """
        Fit AR(1)-GARCH(1,1) with Student-t errors via MLE.

        Parameters
        ----------
        returns : 1D array-like
            Return series (e.g., log or simple returns). No NaNs.

        Returns
        -------
        self
        """
        r = self._to_1d_array(returns)
        if np.isnan(r).any():
            raise ValueError("Input returns contain NaNs; please clean the series.")

        # Mean: ARX with lags=1
        model = ARX(
            r,
            lags=self.cfg.ar_lags,
            hold_back=None,
            rescale=True,
            # enforce_stationarity handled by volatility fit below
        )
        # Volatility: GARCH(1,1)
        model.volatility = GARCH(
            p=self.cfg.garch_p,
            q=self.cfg.garch_q,
            power=2.0,
        )
        # Errors: Student-t
        model.distribution = StudentsT()

        # Fit
        self._res = model.fit(
            update_freq=0,
            disp=self.cfg.disp,
            options={"maxiter": self.cfg.maxiter},
        )
        self._y = r
        return self

    # --------------------------
    # Forecast
    # --------------------------
    def forecast(self, horizon: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-step-ahead conditional mean/variance forecasts.

        Parameters
        ----------
        horizon : int
            Steps ahead. Defaults to cfg.horizon.

        Returns
        -------
        mu : np.ndarray, shape (horizon,)
        var: np.ndarray, shape (horizon,)
        """
        if not self.is_fitted():
            raise RuntimeError("Model not fitted. Call .fit() first.")
        h = int(horizon or self.cfg.horizon)
        fc = self._res.forecast(horizon=h)
        # arch returns pandas structures; get the last row (the future horizon)
        mu = np.asarray(fc.mean.values[-1], dtype=float)      # shape (h,)
        var = np.asarray(fc.variance.values[-1], dtype=float) # shape (h,)
        return mu, var

    # --------------------------
    # Simulation
    # --------------------------
    def simulate_paths(self, horizon=None, n_paths=10000, random_state=None, burn=0):
        if not self.is_fitted():
            raise RuntimeError("Model not fitted. Call .fit() first.")
        h = int(horizon or self.cfg.horizon)

        if random_state is not None:
            np.random.seed(random_state)

        paths = np.empty((h, n_paths))
        for i in range(n_paths):
            sim = self._res.model.simulate(
                params=self._res.params,
                nobs=h,
                burn=burn,
                initial_value=None
            )
            paths[:, i] = np.asarray(sim["data"]).ravel()
        return paths

    # --------------------------
    # Introspection helpers
    # --------------------------
    def params(self):
        """Return fitted parameter vector as a pandas Series (arch result)."""
        if not self.is_fitted():
            raise RuntimeError("Model not fitted.")
        return self._res.params

    def summary(self) -> str:
        """Return text summary from arch result."""
        if not self.is_fitted():
            raise RuntimeError("Model not fitted.")
        return str(self._res.summary())

    def last_conditional_stats(self) -> Tuple[float, float]:
        """
        Return last fitted conditional mean and variance (t = T).
        """
        if not self.is_fitted():
            raise RuntimeError("Model not fitted.")
        mu_t = float(self._res.params.get("Const", 0.0))
        # Better: pull from res.conditional_volatility and conditional mean series
        vol = getattr(self._res, "conditional_volatility", None)
        if vol is None:
            return mu_t, np.nan
        return mu_t, float(vol[-1] ** 2)


# --------------------------
# Tiny usage example (optional)
# --------------------------
if __name__ == "__main__":  # quick smoke test
    np.random.seed(7)
    # toy series: t(8) noise scaled
    toy = np.random.standard_t(df=8, size=800) * 0.01

    model = ARGARCHModel().fit(toy)
    mu, var = model.forecast(horizon=5)
    print("mu[:5] =", mu)
    print("var[:5] =", var)

    paths = model.simulate_paths(horizon=10, n_paths=1000, random_state=123)
    print("paths shape =", paths.shape)

