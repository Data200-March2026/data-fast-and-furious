"""
Config — global constants for the Oil Price Volatility Dashboard.
"""

import os


class Config:
    # ── Colour palette ──────────────────────────────────────────────────────
    COLORS = {
        "dark":   "#1F4E79",
        "mid":    "#2E75B6",
        "light":  "#5BA3D9",
        "pale":   "#D6E8F7",
        "orange": "#C55A11",
        "green":  "#375623",
        "red":    "#C00000",
    }

    # ── Plot settings ───────────────────────────────────────────────────────
    PLOT_DPI   = 150
    FIG_SINGLE = (12, 5)
    FIG_MULTI  = (14, 10)

    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "fuel_prices_1970_2026.csv")

    # ── Key historical events for annotation ────────────────────────────────
    EVENTS = {
        1973: "1973 Oil Crisis",
        1979: "1979 Iranian Revolution",
        1986: "1986 Price Collapse",
        2008: "2008 Financial Crisis",
        2014: "2014 OPEC Glut",
        2020: "2020 COVID Crash",
    }
