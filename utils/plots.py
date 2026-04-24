"""
Plots — all charting functions for the dashboard.
Mixes Plotly (for interactivity) and Matplotlib/Seaborn (for statistical plots).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

from utils.config import Config


# ----------------------------------------------------------------------
# Page 1: Home & Data
# ----------------------------------------------------------------------

def plot_sparkline(df: pd.DataFrame) -> go.Figure:
    """Mini Plotly sparkline of Crude_Oil_Price."""
    fig = px.line(df, x="Date", y="Crude_Oil_Price")
    fig.update_traces(line_color=Config.COLORS["mid"], line_width=1.5)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=100,
        xaxis_visible=False,
        yaxis_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ----------------------------------------------------------------------
# Page 2: Descriptive Stats
# ----------------------------------------------------------------------

def plot_mean_vol_bar(stats_df: pd.DataFrame) -> go.Figure:
    """Plotly bar chart of Mean Rolling Volatility per decade."""
    fig = px.bar(
        stats_df, x="Decade", y="Mean",
        text_auto='.2f',
        title="Mean Rolling Volatility by Decade"
    )
    fig.update_traces(marker_color=Config.COLORS["mid"])
    fig.update_layout(height=400, template="plotly_white")
    return fig

def plot_monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap of average price by Year x Month."""
    pivot = df.pivot_table(
        index="Month", columns="Year",
        values="Crude_Oil_Price", aggfunc="mean"
    )
    fig = px.imshow(
        pivot,
        labels=dict(x="Year", y="Month", color="Price"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Average Crude Oil Price Heatmap"
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


# ----------------------------------------------------------------------
# Page 4: Regression Model
# ----------------------------------------------------------------------

def plot_regression_fit(df: pd.DataFrame, model_ols, r2: float, 
                        show_ma3: bool = False, show_ma12: bool = False) -> go.Figure:
    """Plotly scatter with regression line and optional MAs."""
    fig = go.Figure()

    # Scatter
    fig.add_trace(go.Scatter(
        x=df["Year_Float"], y=df["Rolling_Vol"],
        mode="markers",
        marker=dict(color=Config.COLORS["mid"], size=5, opacity=0.6),
        name="Observed Volatility"
    ))

    # OLS Line
    x_range = np.linspace(df["Year_Float"].min(), df["Year_Float"].max(), 100)
    X_const = sm.add_constant(x_range)
    y_pred = model_ols.predict(X_const)
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_pred,
        mode="lines",
        line=dict(color=Config.COLORS["orange"], width=3),
        name=f"OLS Fit (R²={r2:.3f})"
    ))

    # Optional MAs
    if show_ma3:
        fig.add_trace(go.Scatter(
            x=df["Year_Float"], y=df["MA_3M"],
            mode="lines",
            line=dict(color=Config.COLORS["light"], width=1.5),
            name="3M Moving Avg"
        ))
    if show_ma12:
        fig.add_trace(go.Scatter(
            x=df["Year_Float"], y=df["MA_12M"],
            mode="lines",
            line=dict(color=Config.COLORS["dark"], width=1.5),
            name="12M Moving Avg"
        ))

    fig.update_layout(
        title="Volatility vs Time (Regression Fit)",
        xaxis_title="Year",
        yaxis_title="Rolling Volatility",
        template="plotly_white",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_residual_diagnostics(residuals: np.ndarray, y_fitted: np.ndarray) -> plt.Figure:
    """3-panel Matplotlib/Seaborn residual diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=Config.FIG_MULTI, dpi=Config.PLOT_DPI)
    
    # 1. Residuals vs Fitted
    sns.scatterplot(x=y_fitted, y=residuals, ax=axes[0], alpha=0.6, color=Config.COLORS["mid"])
    axes[0].axhline(0, color=Config.COLORS["orange"], ls="--")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")

    # 2. Histogram + KDE
    sns.histplot(residuals, kde=True, ax=axes[1], color=Config.COLORS["mid"])
    axes[1].set_title("Residual Histogram")
    axes[1].set_xlabel("Residuals")

    # 3. Q-Q Plot
    sm.qqplot(residuals, line='45', fit=True, ax=axes[2], color=Config.COLORS["mid"])
    axes[2].set_title("Normal Q-Q Plot")

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Page 5: Time Series Analysis
# ----------------------------------------------------------------------

def plot_acf_pacf(df: pd.DataFrame, lags: int = 36) -> plt.Figure:
    """Matplotlib ACF/PACF plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=Config.PLOT_DPI)
    
    sm.graphics.tsa.plot_acf(df["Crude_Oil_Price"].dropna(), lags=lags, ax=axes[0], color=Config.COLORS["mid"])
    sm.graphics.tsa.plot_pacf(df["Crude_Oil_Price"].dropna(), lags=lags, ax=axes[1], color=Config.COLORS["orange"])
    
    axes[0].set_title(f"Autocorrelation (Lags={lags})")
    axes[1].set_title(f"Partial Autocorrelation (Lags={lags})")
    
    plt.tight_layout()
    return fig


def plot_decomposition(decomp, model_type: str) -> go.Figure:
    """4-panel Plotly subplot for seasonal decompose."""
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.05
    )

    x_vals = decomp.observed.index

    # Observed
    fig.add_trace(go.Scatter(x=x_vals, y=decomp.observed, mode="lines", 
                             line_color=Config.COLORS["dark"]), row=1, col=1)
    # Trend
    fig.add_trace(go.Scatter(x=x_vals, y=decomp.trend, mode="lines", 
                             line_color=Config.COLORS["mid"]), row=2, col=1)
    # Seasonal
    fig.add_trace(go.Scatter(x=x_vals, y=decomp.seasonal, mode="lines", 
                             line_color=Config.COLORS["orange"]), row=3, col=1)
    # Residual
    fig.add_trace(go.Scatter(x=x_vals, y=decomp.resid, mode="markers", 
                             marker=dict(color=Config.COLORS["red"], size=3)), row=4, col=1)

    fig.update_layout(
        height=700, 
        showlegend=False, 
        template="plotly_white",
        title=f"Seasonal Decomposition ({model_type})"
    )
    return fig


def plot_price_timeseries(df: pd.DataFrame) -> go.Figure:
    """Plotly line chart with fill area and event annotations."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Crude_Oil_Price"],
        mode="lines",
        line=dict(color=Config.COLORS["mid"], width=2),
        fill='tozeroy',
        fillcolor=Config.COLORS["pale"],
        name="Price"
    ))

    # Add Events
    for year, event in Config.EVENTS.items():
        # Find closest date
        event_date = pd.to_datetime(f"{year}-01-01")
        fig.add_vline(
            x=event_date.timestamp() * 1000, 
            line_width=1, line_dash="dash", line_color=Config.COLORS["red"],
            annotation_text=event, annotation_position="top right"
        )

    # Range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        title="Crude Oil Price History (1970 - 2026)",
        yaxis_title="USD / Barrel",
        template="plotly_white",
        height=600
    )
    return fig


def plot_volatility_by_decade(df: pd.DataFrame, plot_type: str = "box") -> go.Figure:
    """Plotly box or violin plot per decade."""
    if plot_type == "box":
        fig = px.box(df, x="Decade", y="Rolling_Vol", color="Decade", 
                     title="Volatility Distribution by Decade")
    else:
        fig = px.violin(df, x="Decade", y="Rolling_Vol", color="Decade", box=True, 
                        title="Volatility Distribution by Decade")
        
    fig.update_layout(template="plotly_white", height=500, showlegend=False)
    return fig
