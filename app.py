import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from utils.config import Config
from utils.logger import logger
from utils.data_processor import build_processor, DataProcessor
from utils.regression import RegressionModeler
from utils.stats_tests import StatsTests
from utils.report import ReportGenerator
import utils.plots as pt

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Oil Price Volatility Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_clean" not in st.session_state:
    st.session_state["df_clean"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "stats_dict" not in st.session_state:
    st.session_state["stats_dict"] = None
if "test_results" not in st.session_state:
    st.session_state["test_results"] = None
if "adf_result" not in st.session_state:
    st.session_state["adf_result"] = None
if "decomp" not in st.session_state:
    st.session_state["decomp"] = None


# -----------------------------------------------------------------------------
# Global Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛢️ Oil Dashboard")
    
    page = st.radio("Navigate", [
        "🏠 Home & Data",
        "📊 Descriptive Stats",
        "🧪 Hypothesis Testing",
        "📈 Regression Model",
        "🔍 Time Series Analysis",
        "💾 Export Results"
    ])
    
    st.markdown("---")
    
    # Date Range Filter
    if st.session_state["df"] is not None:
        min_d = st.session_state["df"]["Date"].min()
        max_d = st.session_state["df"]["Date"].max()
        date_range = st.date_input("Date range filter (Global)", [min_d, max_d], min_value=min_d, max_value=max_d)
    else:
        date_range = None
        
    st.markdown("---")
    
    # Run Pipeline Button
    if st.button("▶ Run Full Pipeline", use_container_width=True):
        if st.session_state["df_clean"] is not None:
            with st.spinner("Running statistical pipeline..."):
                # Stats tests
                tester = StatsTests(st.session_state["df_clean"])
                st.session_state["test_results"] = tester.run_all()
                
                # Regression
                modeler = RegressionModeler(st.session_state["df_clean"])
                modeler.fit()
                st.session_state["model"] = modeler
                
                # TS Decompose
                ts = st.session_state["df_clean"].set_index("Date")["Crude_Oil_Price"].dropna()
                st.session_state["decomp"] = seasonal_decompose(ts, model='multiplicative', period=12)
                st.session_state["adf_result"] = adfuller(ts)
                
            st.success("Pipeline executed successfully!")
        else:
            st.error("Load data on Home page first.")

    st.markdown("### Analysis Status")
    st.write("✅ Data loaded" if st.session_state["df"] is not None else "⬜ Data not loaded")
    st.write("✅ Model fitted" if st.session_state["model"] is not None else "⬜ Model not fitted")
    st.write("✅ Tests run" if st.session_state["test_results"] is not None else "⬜ Tests not run")


# -----------------------------------------------------------------------------
# Page 1: Home & Data
# -----------------------------------------------------------------------------
if page == "🏠 Home & Data":
    st.title("Oil Price Volatility Dashboard 1970–2026")
    
    uploaded_file = st.file_uploader("Upload custom CSV (Optional)", type=["csv"])
    
    # Process data when page loads or file is uploaded
    with st.spinner("Loading and processing data..."):
        file_bytes = uploaded_file.getvalue() if uploaded_file else None
        try:
            df, df_clean = build_processor(file_bytes)
            st.session_state["df"] = df
            st.session_state["df_clean"] = df_clean
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    # Filter dataframe by global date range
    df = st.session_state["df"]
    if date_range and len(date_range) == 2:
        mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
        df_filtered = df[mask]
    else:
        df_filtered = df

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df_filtered):,}")
    with col2:
        st.metric("Date Range", f"{df_filtered['Year'].min()} - {df_filtered['Year'].max()}")
    with col3:
        latest = df_filtered.iloc[-1]
        st.metric("Latest Price", f"${latest['Crude_Oil_Price']:.2f}", f"{latest['Pct_Change']:.2f}%")
    with col4:
        st.metric("Mean 12M Volatility", f"{df_filtered['Rolling_Vol'].mean():.2f}")

    # Sparkline
    st.plotly_chart(pt.plot_sparkline(df_filtered), use_container_width=True)

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df_filtered.head(10), use_container_width=True)
    
    with st.expander("Full dataset summary"):
        st.dataframe(df_filtered.describe().style.format("{:.2f}"))


# -----------------------------------------------------------------------------
# Page 2: Descriptive Stats
# -----------------------------------------------------------------------------
elif page == "📊 Descriptive Stats":
    st.title("Descriptive Statistics")
    
    if st.session_state["df_clean"] is None:
        st.warning("Please load data on the Home page first.")
        st.stop()
        
    proc = DataProcessor()
    proc.df_clean = st.session_state["df_clean"]
    decades = proc.get_decades()
    stats_df = proc.descriptive_stats_by_decade()
    
    selected_decades = st.multiselect("Filter Decades", decades, default=decades)
    stats_filtered = stats_df[stats_df["Decade"].isin(selected_decades)]
    
    st.dataframe(stats_filtered, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pt.plot_mean_vol_bar(stats_filtered), use_container_width=True)
    with col2:
        st.plotly_chart(pt.plot_monthly_heatmap(st.session_state["df_clean"]), use_container_width=True)


# -----------------------------------------------------------------------------
# Page 3: Hypothesis Testing
# -----------------------------------------------------------------------------
elif page == "🧪 Hypothesis Testing":
    st.title("Hypothesis Testing")
    
    if st.session_state["df_clean"] is None:
        st.warning("Please load data on the Home page first.")
        st.stop()
        
    st.info("""
    **H₀:** No significant difference in rolling volatility across decades.  
    **H₁:** Significant difference exists (α = 0.05).
    """)
    
    # Auto-run if not run
    if st.session_state["test_results"] is None:
        tester = StatsTests(st.session_state["df_clean"])
        st.session_state["test_results"] = tester.run_all()
        
    res = st.session_state["test_results"]
    
    c1, c2 = st.columns(2)
    
    # Shapiro
    with c1:
        st.subheader("1. Shapiro-Wilk (Normality)")
        s = res["shapiro"]
        st.metric("W-Statistic", f"{s['stat']:.4f}")
        st.metric("p-value", f"{s['pval']:.4e}")
        if s["normal"]:
            st.success("Normal distribution (p > 0.05)")
        else:
            st.error("Non-normal distribution (p <= 0.05)")
            
    # ANOVA
    with c2:
        st.subheader("2. One-Way ANOVA")
        a = res["anova"]
        st.metric("F-Statistic", f"{a['f_stat']:.4f}")
        st.metric("p-value", f"{a['pval']:.4e}")
        st.metric("Effect Size (η²)", f"{a['eta_sq']:.4f}")
        if a["reject_h0"]:
            st.error("REJECT H₀ (Significant differences exist)")
        else:
            st.success("FAIL TO REJECT H₀ (No significant difference)")
            
    st.markdown("---")
    c3, c4 = st.columns(2)
    
    # Levene
    with c3:
        st.subheader("3. Levene's Test (Equal Variance)")
        l = res["levene"]
        st.metric("Statistic", f"{l['stat']:.4f}")
        st.metric("p-value", f"{l['pval']:.4e}")
        if l["equal_var"]:
            st.success("Equal variances (Homoscedasticity)")
        else:
            st.error("Unequal variances (Heteroscedasticity)")
            
    # Kruskal
    with c4:
        st.subheader("4. Kruskal-Wallis (Non-Parametric)")
        k = res["kruskal"]
        st.metric("H-Statistic", f"{k['h_stat']:.4f}")
        st.metric("p-value", f"{k['pval']:.4e}")
        if k["reject_h0"]:
            st.error("REJECT H₀ (Significant differences exist)")
        else:
            st.success("FAIL TO REJECT H₀ (No significant difference)")
            
    with st.expander("Plain-English Explanations"):
        st.markdown("""
        * **Shapiro-Wilk:** Checks if our volatility data follows a normal (bell-curve) distribution.
        * **One-Way ANOVA:** Checks if the average volatility is statistically the same across all decades. Assumes normality and equal variance.
        * **Levene's Test:** Checks if the *spread* (variance) of volatility is the same across all decades.
        * **Kruskal-Wallis:** A robust alternative to ANOVA that doesn't require the data to be normally distributed.
        """)


# -----------------------------------------------------------------------------
# Page 4: Regression Model
# -----------------------------------------------------------------------------
elif page == "📈 Regression Model":
    st.title("Regression Model (Vol ~ Time)")
    
    if st.session_state["df_clean"] is None:
        st.warning("Please load data on the Home page first.")
        st.stop()
        
    df_clean = st.session_state["df_clean"]
    min_yr, max_yr = int(df_clean["Year"].min()), int(df_clean["Year"].max())
    
    year_range = st.slider("Filter Year Range for Model", min_yr, max_yr, (min_yr, max_yr))
    
    modeler = RegressionModeler(df_clean)
    modeler.fit(year_min=year_range[0], year_max=year_range[1])
    st.session_state["model"] = modeler
    
    ols_res = modeler.ols_summary()
    sk_res = modeler.sklearn_summary()
    
    # Interpretation
    sig = "statistically significant" if ols_res["year_pval"] < 0.05 else "NOT significant"
    inc_dec = "increases" if ols_res["year_coef"] > 0 else "decreases"
    st.info(f"**Interpretation:** Volatility {inc_dec} by {abs(ols_res['year_coef']):.4f} units per year ({sig} at α=0.05).")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Statsmodels (OLS)")
        st.dataframe(pd.DataFrame({
            "Metric": ["Intercept (β₀)", "Year Coef (β₁)", "R²", "Adj R²", "F-Stat", "Durbin-Watson", "N Obs"],
            "Value": [
                f"{ols_res['intercept']:.4f} (p={ols_res['intercept_pval']:.3e})",
                f"{ols_res['year_coef']:.4f} (p={ols_res['year_pval']:.3e})",
                f"{ols_res['r2']:.4f}",
                f"{ols_res['adj_r2']:.4f}",
                f"{ols_res['f_stat']:.2f} (p={ols_res['f_pval']:.3e})",
                f"{ols_res['durbin_watson']:.4f}",
                ols_res['n_obs']
            ]
        }).set_index("Metric"))
        
    with c2:
        st.subheader("Scikit-Learn")
        st.dataframe(pd.DataFrame({
            "Metric": ["R² Score", "MSE", "RMSE"],
            "Value": [f"{sk_res['r2']:.4f}", f"{sk_res['mse']:.4f}", f"{sk_res['rmse']:.4f}"]
        }).set_index("Metric"))
        
    # Charts
    tab1, tab2 = st.tabs(["Regression Fit", "Residual Diagnostics"])
    
    with tab1:
        c3, c4 = st.columns([1, 4])
        with c3:
            show_ma3 = st.checkbox("Show 3M MA")
            show_ma12 = st.checkbox("Show 12M MA")
        with c4:
            df_filtered = modeler.df[(modeler.df["Year"] >= year_range[0]) & (modeler.df["Year"] <= year_range[1])]
            fig = pt.plot_regression_fit(df_filtered, modeler.model_ols, ols_res['r2'], show_ma3, show_ma12)
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        residuals = modeler.get_residuals()
        y_fitted = modeler.predict()
        st.pyplot(pt.plot_residual_diagnostics(residuals, y_fitted))


# -----------------------------------------------------------------------------
# Page 5: Time Series Analysis
# -----------------------------------------------------------------------------
elif page == "🔍 Time Series Analysis":
    st.title("Time Series Analysis")
    
    if st.session_state["df_clean"] is None:
        st.warning("Please load data on the Home page first.")
        st.stop()
        
    df_clean = st.session_state["df_clean"]
    ts_data = df_clean.set_index("Date")["Crude_Oil_Price"].dropna()
    
    st.subheader("A. Stationarity Test (ADF)")
    if st.session_state["adf_result"] is None:
        st.session_state["adf_result"] = adfuller(ts_data)
        
    adf = st.session_state["adf_result"]
    st.write(f"**ADF Statistic:** {adf[0]:.4f}")
    st.write(f"**p-value:** {adf[1]:.4f}")
    st.write("**Critical Values:**", adf[4])
    
    if adf[1] < 0.05:
        st.success("Series is stationary")
    else:
        st.warning("Non-stationary")
        
    st.markdown("---")
    st.subheader("B. Autocorrelation")
    lags = st.slider("Number of Lags", 12, 48, 36)
    st.pyplot(pt.plot_acf_pacf(df_clean, lags=lags))
    
    st.markdown("---")
    st.subheader("C. Decomposition")
    model_type = st.radio("Model type", ["multiplicative", "additive"])
    decomp = seasonal_decompose(ts_data, model=model_type, period=12)
    st.plotly_chart(pt.plot_decomposition(decomp, model_type), use_container_width=True)
    
    st.markdown("---")
    st.subheader("D. Price Time Series")
    st.plotly_chart(pt.plot_price_timeseries(st.session_state["df"]), use_container_width=True)
    
    st.markdown("---")
    st.subheader("E. Volatility Distribution")
    plot_type = st.radio("Plot Type", ["box", "violin"])
    st.plotly_chart(pt.plot_volatility_by_decade(df_clean, plot_type), use_container_width=True)


# -----------------------------------------------------------------------------
# Page 6: Export Results
# -----------------------------------------------------------------------------
elif page == "💾 Export Results":
    st.title("Export Results")
    
    if st.session_state["df_clean"] is None:
        st.warning("Please load data on the Home page first.")
        st.stop()
        
    # Ensure all pipeline is run
    if st.session_state["test_results"] is None or st.session_state["model"] is None:
        with st.spinner("Running full pipeline to generate exports..."):
            tester = StatsTests(st.session_state["df_clean"])
            st.session_state["test_results"] = tester.run_all()
            
            modeler = RegressionModeler(st.session_state["df_clean"])
            modeler.fit()
            st.session_state["model"] = modeler

    df_clean = st.session_state["df_clean"]
    tests = st.session_state["test_results"]
    model = st.session_state["model"]
    
    # 1. JSON Export
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "rows": len(df_clean),
        "mean_vol": float(df_clean["Rolling_Vol"].mean()),
        "hypothesis_tests": tests,
        "regression": {**model.ols_summary(), **model.sklearn_summary()}
    }
    json_str = json.dumps(json_data, indent=2, encoding="utf-8") if hasattr(json, 'encoding') else json.dumps(json_data, indent=2)
    
    # 2. Text Report
    report_gen = ReportGenerator(
        df_len=len(df_clean),
        mean_vol=df_clean["Rolling_Vol"].mean(),
        std_resid=float(np.std(model.get_residuals())),
        tests_dict=tests
    )
    txt_str = report_gen.generate()
    
    # 3. CSV Export
    csv_str = df_clean.to_csv(index=False).encode('utf-8')
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download JSON", json_str, "results.json", "application/json", use_container_width=True)
    with c2:
        st.download_button("Download Text Report", txt_str, "week7_analysis_report.txt", "text/plain", use_container_width=True)
    with c3:
        st.download_button("Download Processed Data", csv_str, "processed_data.csv", "text/csv", use_container_width=True)
        
    st.markdown("---")
    st.subheader("Preview JSON")
    st.json(json_data)
    
    st.subheader("Preview Text Report")
    st.code(txt_str, language="text")
    
    st.subheader("Preview Processed Data")
    st.dataframe(df_clean.head(), use_container_width=True)
