"""
DataProcessor — loads, engineers features, and cleans the oil-price dataset.
Wrapped in a Streamlit-cached factory function.
"""

import pandas as pd
import numpy as np
import streamlit as st

from utils.config import Config
from utils.logger import logger


class DataProcessor:
    """Handles CSV loading and feature engineering for the oil-price dataset."""

    def __init__(self, filepath: str = None):
        self.filepath = filepath or Config.DATA_PATH
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_clean: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------

    def load(self, uploaded_file=None) -> "DataProcessor":
        """Read CSV from disk or an uploaded Streamlit file object."""
        try:
            if uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file, parse_dates=["Date"])
            else:
                self.df = pd.read_csv(self.filepath, parse_dates=["Date"])
            self.df.sort_values("Date", inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f"Loaded {len(self.df):,} rows from dataset.")
        except Exception as exc:
            logger.error(f"Failed to load data: {exc}")
            raise
        return self

    def engineer_features(self) -> "DataProcessor":
        """Create all derived columns needed by the analysis pages."""
        df = self.df.copy()

        # ── Temporal features ────────────────────────────────────────
        df["Year"]       = df["Date"].dt.year
        df["Month"]      = df["Date"].dt.month
        df["Year_Float"] = df["Year"] + (df["Month"] - 1) / 12
        df["Decade"]     = (df["Year"] // 10 * 10).astype(str) + "s"

        # ── Rolling statistics ────────────────────────────────────────
        df["Rolling_Vol"] = (
            df["Crude_Oil_Price"]
            .rolling(window=12, min_periods=12)
            .std()
        )
        df["MA_3M"]  = df["Crude_Oil_Price"].rolling(window=3,  min_periods=1).mean()
        df["MA_12M"] = df["Crude_Oil_Price"].rolling(window=12, min_periods=1).mean()

        # ── Month-over-month % change ─────────────────────────────────
        df["Pct_Change"] = df["Crude_Oil_Price"].pct_change() * 100

        self.df = df
        logger.info("Feature engineering complete.")
        return self

    def clean(self) -> "DataProcessor":
        """Drop rows where Rolling_Vol is NaN; store as df_clean."""
        self.df_clean = self.df.dropna(subset=["Rolling_Vol"]).reset_index(drop=True)
        logger.success(
            f"Clean dataset: {len(self.df_clean):,} rows "
            f"(dropped {len(self.df) - len(self.df_clean)} warm-up rows)."
        )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_decades(self) -> list:
        """Return sorted list of decade labels present in df_clean."""
        return sorted(self.df_clean["Decade"].unique().tolist())

    def descriptive_stats_by_decade(self) -> pd.DataFrame:
        """
        Compute N, Mean, Median, Std, Min, Max, CV% of Rolling_Vol
        grouped by Decade.
        """
        grp = self.df_clean.groupby("Decade")["Rolling_Vol"]
        stats = grp.agg(
            N="count",
            Mean="mean",
            Median="median",
            Std="std",
            Min="min",
            Max="max",
        ).reset_index()
        stats["CV%"] = (stats["Std"] / stats["Mean"] * 100).round(2)
        for col in ["Mean", "Median", "Std", "Min", "Max"]:
            stats[col] = stats[col].round(4)
        return stats


# ── Cached factory function ────────────────────────────────────────────────

def build_processor(uploaded_bytes: bytes | None = None) -> tuple:
    """
    Build and run the DataProcessor pipeline.
    Returns (df, df_clean) as plain DataFrames so st.cache_data can serialise them.
    """
    proc = DataProcessor()
    if uploaded_bytes is not None:
        import io
        proc.load(io.BytesIO(uploaded_bytes))
    else:
        proc.load()
    proc.engineer_features()
    proc.clean()
    return proc.df, proc.df_clean
