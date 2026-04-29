"""
RegressionModeler — fits OLS (statsmodels) and LinearRegression (sklearn)
on Rolling_Vol ~ Year_Float.
"""

import numpy as np
import pandas as pd
import streamlit as st

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from utils.logger import logger


class RegressionModeler:
    """Dual regression engine: statsmodels OLS + sklearn LinearRegression."""

    def __init__(self, df_clean: pd.DataFrame):
        self.df = df_clean.copy()
        self.model_ols  = None   # statsmodels OLS result
        self.model_sk   = None   # sklearn fitted model
        self.X          = None   # feature array (Year_Float)
        self.y          = None   # target array (Rolling_Vol)
        self.y_pred_sk  = None   # sklearn predictions
        self.dw_stat    = None   # Durbin-Watson statistic
        self._fitted    = False

    # ------------------------------------------------------------------
    def fit(self, year_min: int | None = None, year_max: int | None = None):
        """Fit both models, optionally restricting to a year range."""
        df = self.df.copy()
        if year_min is not None:
            df = df[df["Year"] >= year_min]
        if year_max is not None:
            df = df[df["Year"] <= year_max]

        if len(df) < 20:
            logger.error("Not enough data to fit regression (need ≥ 20 rows).")
            return self

        self.X = df[["Year_Float"]].values
        self.y = df["Rolling_Vol"].values

        # ── OLS (statsmodels) ─────────────────────────────────────────
        X_const = sm.add_constant(self.X)
        self.model_ols = sm.OLS(self.y, X_const).fit()
        residuals = np.asarray(self.model_ols.resid)
        self.dw_stat = durbin_watson(residuals)

        # ── sklearn LinearRegression ───────────────────────────────────
        self.model_sk  = LinearRegression()
        self.model_sk.fit(self.X, self.y)
        self.y_pred_sk = self.model_sk.predict(self.X)

        self._fitted = True
        logger.success("Regression models fitted successfully.")
        return self

    # ------------------------------------------------------------------
    def get_residuals(self) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.model_ols.resid)

    def predict(self) -> np.ndarray:
        self._check_fitted()
        return self.y_pred_sk

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def ols_summary(self) -> dict:
        """Return key OLS metrics as a plain dict."""
        self._check_fitted()
        r = self.model_ols
        return {
            "intercept":       r.params[0],
            "intercept_pval":  r.pvalues[0],
            "year_coef":       r.params[1],
            "year_pval":       r.pvalues[1],
            "r2":              r.rsquared,
            "adj_r2":          r.rsquared_adj,
            "f_stat":          r.fvalue,
            "f_pval":          r.f_pvalue,
            "durbin_watson":   self.dw_stat,
            "n_obs":           int(r.nobs),
        }

    def sklearn_summary(self) -> dict:
        """Return key sklearn metrics as a plain dict."""
        self._check_fitted()
        mse  = mean_squared_error(self.y, self.y_pred_sk)
        rmse = np.sqrt(mse)
        r2   = r2_score(self.y, self.y_pred_sk)
        return {
            "r2":   r2,
            "mse":  mse,
            "rmse": rmse,
        }

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing model results.")
