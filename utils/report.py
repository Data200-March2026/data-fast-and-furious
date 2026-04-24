"""
ReportGenerator — exports the final text report.
"""

from datetime import datetime

class ReportGenerator:
    """Generates a plain text summary report of all analysis results."""

    def __init__(self, df_len: int, mean_vol: float, std_resid: float, tests_dict: dict):
        self.df_len = df_len
        self.mean_vol = mean_vol
        self.std_resid = std_resid
        self.tests = tests_dict

    def generate(self) -> str:
        """Return the multi-line text report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            "==================================================",
            "      OIL PRICE VOLATILITY ANALYSIS REPORT      ",
            "==================================================",
            f"Generated: {timestamp}",
            "",
            "1. GLOBAL METRICS",
            "-----------------",
            f"  - Total Observations: {self.df_len:,}",
            f"  - Mean Rolling Vol:   {self.mean_vol:.4f}",
            f"  - Residual Std Dev:   {self.std_resid:.4f}",
            "",
            "2. HYPOTHESIS TESTS",
            "-------------------"
        ]

        # Shapiro
        s = self.tests.get("shapiro", {})
        if s:
            lines.append(f"  [Shapiro-Wilk] Normal? {s.get('normal')}")
            lines.append(f"    W={s.get('stat', 0):.4f}, p={s.get('pval', 0):.4f}")

        # ANOVA
        a = self.tests.get("anova", {})
        if a:
            lines.append(f"  [One-Way ANOVA] Reject H0? {a.get('reject_h0')}")
            lines.append(f"    F={a.get('f_stat', 0):.4f}, p={a.get('pval', 0):.4f}")
            lines.append(f"    Effect Size (η²): {a.get('eta_sq', 0):.4f}")

        # Levene
        l = self.tests.get("levene", {})
        if l:
            lines.append(f"  [Levene's Test] Equal Variance? {l.get('equal_var')}")
            lines.append(f"    Stat={l.get('stat', 0):.4f}, p={l.get('pval', 0):.4f}")

        # Kruskal
        k = self.tests.get("kruskal", {})
        if k:
            lines.append(f"  [Kruskal-Wallis] Reject H0? {k.get('reject_h0')}")
            lines.append(f"    H={k.get('h_stat', 0):.4f}, p={k.get('pval', 0):.4f}")

        lines.extend([
            "",
            "==================================================",
            "                   END OF REPORT                  ",
            "==================================================",
        ])
        
        return "\n".join(lines)
