"""
dashboard.py
------------
Solar Energy Forecasting Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(
    page_title="Solar Energy Forecasting",
    page_icon="☀️",
    layout="wide"
)

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e0e0e0;
}
.metric-value { font-size: 28px; font-weight: 600; color: #1F4E79; }
.metric-label { font-size: 12px; color: #666; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    historical = pd.read_csv("data/ml_ready_sites.csv")
    historical['month_year'] = pd.to_datetime(historical['month_year'])

    forecasts = None
    if os.path.exists("outputs/forecasts_2026.csv"):
        forecasts = pd.read_csv("outputs/forecasts_2026.csv")
        forecasts['month_year'] = pd.to_datetime(forecasts['month_year'])

    evaluation = None
    if os.path.exists("outputs/model_evaluation.csv"):
        evaluation = pd.read_csv("outputs/model_evaluation.csv")

    shap_df = None
    if os.path.exists("outputs/shap_values.csv"):
        shap_df = pd.read_csv("outputs/shap_values.csv")

    return historical, forecasts, evaluation, shap_df

historical, forecasts, evaluation, shap_df = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("☀️ Solar Energy Production Forecasting")
st.markdown("**Self-initiated ML pipeline** — XGBoost · LightGBM · Random Forest · Gradient Boosting · Linear Regression")

# 🟡 Beginner intro box
st.info("""
👋 **What is this dashboard?**

Solar panels generate electricity from sunlight — measured in **kWh (kilowatt-hours)**, the same unit on your electricity bill.

This dashboard uses **Machine Learning (AI)** to:
- Look at past solar energy data from 6 real solar sites
- Learn patterns (like: summers produce more energy than winters)
- **Predict** how much energy each site will produce in the coming months

Think of it like a weather forecast — but for solar energy! ☀️
""")

st.divider()

# ── TOP METRICS ───────────────────────────────────────────────────────────────
st.subheader("📌 Quick Summary")
st.caption("These numbers give you a quick overview of the entire project at a glance.")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-value">{historical['site_id'].nunique()}</div>
    <div class="metric-label">☀️ Solar Sites Tracked</div>
    </div>""", unsafe_allow_html=True)
    st.caption("Number of solar panel locations we have data for.")

with col2:
    months = historical[historical['actual_kwh'] > 0].shape[0]
    st.markdown(f"""<div class="metric-card">
    <div class="metric-value">{months}</div>
    <div class="metric-label">📅 Months of Data Used</div>
    </div>""", unsafe_allow_html=True)
    st.caption("How many months of real data the AI learned from.")

with col3:
    if evaluation is not None:
        best_rmse = evaluation['rmse'].min()
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{best_rmse:,.0f}</div>
        <div class="metric-label">🎯 Best Prediction Error (kWh)</div>
        </div>""", unsafe_allow_html=True)
        st.caption("On average, our best AI model's prediction is off by this many kWh. Lower = more accurate!")
    else:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">—</div>
        <div class="metric-label">Run pipeline first</div>
        </div>""", unsafe_allow_html=True)

with col4:
    if evaluation is not None:
        best_model = evaluation.loc[evaluation['rmse'].idxmin(), 'model']
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:16px">{best_model}</div>
        <div class="metric-label">🏆 Winning AI Model</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Out of 5 AI models tested, this one made the most accurate predictions.")
    else:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">—</div>
        <div class="metric-label">Best Model</div>
        </div>""", unsafe_allow_html=True)

with col5:
    if forecasts is not None:
        total_forecast = forecasts['forecast_kwh'].sum()
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total_forecast/1000:,.0f}k</div>
        <div class="metric-label">⚡ Total Forecasted kWh</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Total energy we expect all sites to produce in 2026 combined.")
    else:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">—</div>
        <div class="metric-label">Forecasted kWh</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── SITE SELECTOR & FORECAST CHART ───────────────────────────────────────────
st.subheader("📈 Forecast vs Actual — Site View")

st.markdown("""
**What am I looking at?**
- 🔵 **Blue line (Actual kWh)** — How much energy the solar site *really* produced each month (historical data)
- ⬜ **Grey dashed line (Expected kWh)** — How much energy the manufacturer said it *should* produce (the benchmark)
- 🟠 **Orange dashed line (Forecast)** — How much our AI *predicts* it will produce in 2026
- 🟠 **Orange shaded area (Confidence interval ±15%)** — The range where the actual value will most likely fall. Think of it like saying *"we're pretty sure it'll be somewhere in this zone"*
""")

sites = sorted(historical['site_id'].unique())
selected_site = st.selectbox("🔍 Select a solar site to explore:", sites)

hist_site = historical[(historical['site_id'] == selected_site) & (historical['actual_kwh'] > 0)]

fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(
    x=hist_site['month_year'], y=hist_site['actual_kwh'],
    mode='lines+markers', name='Actual kWh',
    line=dict(color='#2E75B6', width=2),
    marker=dict(size=4)
))

# Expected
fig.add_trace(go.Scatter(
    x=hist_site['month_year'], y=hist_site['expected_kwh'],
    mode='lines', name='Expected kWh (Benchmark)',
    line=dict(color='#A0A0A0', width=1, dash='dash')
))

# Forecast
if forecasts is not None:
    fut_site = forecasts[forecasts['site_id'] == selected_site]
    if len(fut_site) > 0:
        fig.add_trace(go.Scatter(
            x=fut_site['month_year'], y=fut_site['forecast_kwh'],
            mode='lines+markers', name='AI Forecast (2026)',
            line=dict(color='#ED7D31', width=2, dash='dash'),
            marker=dict(size=5, symbol='square')
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fut_site['month_year'], fut_site['month_year'][::-1]]),
            y=pd.concat([fut_site['upper_bound'], fut_site['lower_bound'][::-1]]),
            fill='toself', fillcolor='rgba(237,125,49,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Likely Range (±15%)'
        ))

fig.update_layout(
    height=420,
    xaxis_title="Month",
    yaxis_title="Energy Produced (kWh)",
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    plot_bgcolor='white',
    yaxis=dict(gridcolor='#f0f0f0'),
    xaxis=dict(gridcolor='#f0f0f0'),
    margin=dict(t=20, b=40)
)
st.plotly_chart(fig, use_container_width=True)

st.caption("💡 Tip: You can zoom in by clicking and dragging on the chart. Double-click to zoom back out.")

st.divider()

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🏆 Which AI Model Won?")
    st.markdown("""
We tested **5 different AI models** — think of them as 5 students all trying to solve the same maths problem.
We then checked whose answers were closest to the real values.

**RMSE (Root Mean Squared Error)** = the average mistake each model makes in kWh.
📌 **Lower RMSE = better model = more accurate predictions.**

The 🔵 dark blue bar is the winner!
    """)
    if evaluation is not None:
        eval_sorted = evaluation.sort_values('rmse')
        fig_eval = go.Figure(go.Bar(
            y=eval_sorted['model'],
            x=eval_sorted['rmse'],
            orientation='h',
            marker_color=['#1F4E79' if i == 0 else '#B8CCE4' for i in range(len(eval_sorted))],
            text=[f"{v:,.0f} kWh error" for v in eval_sorted['rmse']],
            textposition='outside'
        ))
        fig_eval.update_layout(
            height=280, xaxis_title="Average Prediction Error (RMSE in kWh)",
            plot_bgcolor='white', margin=dict(t=10, b=10),
            xaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_eval, use_container_width=True)
    else:
        st.info("Run forecasting_pipeline.py to see model comparison")

with col_b:
    st.subheader("🔍 What Factors Matter Most?")
    st.markdown("""
**SHAP Feature Importance** tells us: *"Which pieces of information did the AI rely on most to make its predictions?"*

Think of it like asking a doctor: *"What symptoms made you give this diagnosis?"*

For example:
- 📅 **Last year's same month** = strongest clue (solar energy is very seasonal!)
- 🌤️ **Expected output** = how much the panels were designed to produce
- 📊 **Recent averages** = recent trends in production

**Longer bar = more important factor.**
    """)
    if shap_df is not None:
        top10 = shap_df.head(10)
        fig_shap = go.Figure(go.Bar(
            y=top10['feature'][::-1],
            x=top10['importance'][::-1],
            orientation='h',
            marker_color='#2E75B6'
        ))
        fig_shap.update_layout(
            height=280, xaxis_title="Importance Score (higher = more influential)",
            plot_bgcolor='white', margin=dict(t=10, b=10),
            xaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("Run forecasting_pipeline.py to see feature importance")

st.divider()

# ── SITE RANKINGS ─────────────────────────────────────────────────────────────
st.subheader("📊 How Is Each Site Performing?")
st.markdown("""
This table ranks all solar sites by their **Performance Ratio** — a score that tells us how efficiently a site is working.

- **Performance Ratio** = Actual energy produced ÷ Expected energy × 100%
- A score of **90%+** means the site is working great 🟢
- A score of **70–90%** means it needs monitoring 🟡
- A score **below 70%** means something might be wrong 🔴

Think of it like a student's exam score — 90%+ is excellent, below 70% needs attention!
""")

site_summary = historical[historical['actual_kwh'] > 0].groupby('site_id').agg(
    avg_performance=('performance_ratio', 'mean'),
    total_actual=('actual_kwh', 'sum'),
    months=('actual_kwh', 'count')
).reset_index().sort_values('avg_performance', ascending=False)

site_summary['avg_performance_pct'] = (site_summary['avg_performance'] * 100).round(1)
site_summary['status'] = site_summary['avg_performance'].apply(
    lambda x: '🟢 Healthy' if x >= 0.9 else ('🟡 Monitor' if x >= 0.7 else '🔴 Alert')
)

st.dataframe(
    site_summary[['site_id', 'avg_performance_pct', 'total_actual', 'months', 'status']].rename(columns={
        'site_id': 'Site',
        'avg_performance_pct': 'Performance Score (%)',
        'total_actual': 'Total Energy Produced (kWh)',
        'months': 'Months of Data',
        'status': 'Health Status'
    }),
    use_container_width=True, hide_index=True
)

st.caption("💡 kWh = kilowatt-hour. 1 kWh can power a ceiling fan for about 10 hours or charge your phone around 50 times!")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("🤔 Glossary — What do these terms mean?"):
    st.markdown("""
| Term | Simple Explanation |
|------|-------------------|
| **kWh** | Unit of energy — like litres for water, but for electricity |
| **ML / Machine Learning** | Teaching a computer to find patterns in data and make predictions |
| **RMSE** | Average mistake the AI makes — lower is better |
| **SHAP** | A tool that explains *why* the AI made a specific prediction |
| **Forecast** | A prediction of what will happen in the future |
| **Performance Ratio** | How well a solar site is doing vs how well it should be doing |
| **Confidence Interval** | The range within which the real answer will most likely fall |
| **XGBoost / LightGBM** | Names of specific AI algorithms — like different brands of calculators |
    """)

st.markdown("""
<div style='text-align:center; color:#999; font-size:12px'>
Self-initiated ML project | Real solar operational data (anonymized) |
Stack: Python · XGBoost · LightGBM · Scikit-learn · SHAP · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)

st.divider()

# ──owner ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#999; font-size:12px; margin-top:8px'>
Built by <strong>Geeta Bhimsen Maharana</strong> · 
<a href='https://www.linkedin.com/in/geetabhimsenmaharana/' target='_blank'>Connect on LinkedIn</a>
</div>
""", unsafe_allow_html=True)
