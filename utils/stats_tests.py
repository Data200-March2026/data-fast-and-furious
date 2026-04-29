"""
StatsTests — runs Shapiro-Wilk, ANOVA, Levene, and Kruskal-Wallis
on Rolling_Vol grouped by Decade.
"""

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import logger


class StatsTests:
    """Wraps the four hypothesis tests used in the dashboard."""

    def __init__(self, df_clean: pd.DataFrame):
        self.df = df_clean.copy()
        self._groups: list = []   # list of arrays, one per decade
        self._decade_labels: list = []
        self._build_groups()

    # ------------------------------------------------------------------
    def _build_groups(self):
        grp = self.df.groupby("Decade")["Rolling_Vol"]
        for decade, series in grp:
            self._decade_labels.append(decade)
            self._groups.append(series.dropna().values)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def shapiro_wilk(self) -> dict:
        """
        Shapiro-Wilk normality test on Rolling_Vol.
        Sub-samples to 5 000 observations if needed.
        """
        data = self.df["Rolling_Vol"].dropna().values
        if len(data) > 5000:
            rng  = np.random.default_rng(42)
            data = rng.choice(data, size=5000, replace=False)

        stat, pval = stats.shapiro(data)
        result = {
            "test":    "Shapiro-Wilk",
            "stat":    float(stat),
            "pval":    float(pval),
            "normal":  bool(pval > 0.05),
            "n":       len(data),
        }
        logger.info(f"Shapiro-Wilk: W={stat:.4f}, p={pval:.4f}")
        return result

    def one_way_anova(self) -> dict:
        """One-way ANOVA across decades. Includes eta-squared effect size."""
        f_stat, pval = stats.f_oneway(*self._groups)

        # eta-squared: SS_between / SS_total
        grand_mean = self.df["Rolling_Vol"].mean()
        ss_total   = sum((v - grand_mean) ** 2
                         for g in self._groups for v in g)
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2
                         for g in self._groups)
        eta_sq     = ss_between / ss_total if ss_total else np.nan

        result = {
            "test":      "One-Way ANOVA",
            "f_stat":    float(f_stat),
            "pval":      float(pval),
            "eta_sq":    float(eta_sq),
            "reject_h0": bool(pval < 0.05),
        }
        logger.info(f"ANOVA: F={f_stat:.4f}, p={pval:.4f}, η²={eta_sq:.4f}")
        return result

    def levene_test(self) -> dict:
        """Levene's test for equality of variances across decades."""
        stat, pval = stats.levene(*self._groups)
        result = {
            "test":          "Levene's Test",
            "stat":          float(stat),
            "pval":          float(pval),
            "equal_var":     bool(pval > 0.05),
        }
        logger.info(f"Levene: stat={stat:.4f}, p={pval:.4f}")
        return result

    def kruskal_wallis(self) -> dict:
        """Kruskal-Wallis non-parametric test across decades."""
        h_stat, pval = stats.kruskal(*self._groups)
        result = {
            "test":      "Kruskal-Wallis",
            "h_stat":    float(h_stat),
            "pval":      float(pval),
            "reject_h0": bool(pval < 0.05),
        }
        logger.info(f"Kruskal-Wallis: H={h_stat:.4f}, p={pval:.4f}")
        return result

    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        """Run all four tests and return a combined results dict."""
        return {
            "shapiro":   self.shapiro_wilk(),
            "anova":     self.one_way_anova(),
            "levene":    self.levene_test(),
            "kruskal":   self.kruskal_wallis(),
        }
