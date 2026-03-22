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
st.divider()

# ── TOP METRICS ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{historical['site_id'].nunique()}</div>
        <div class="metric-label">Sites Monitored</div>
    </div>""", unsafe_allow_html=True)

with col2:
    months = historical[historical['actual_kwh'] > 0].shape[0]
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{months}</div>
        <div class="metric-label">Training Months</div>
    </div>""", unsafe_allow_html=True)

with col3:
    if evaluation is not None:
        best_rmse = evaluation['rmse'].min()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best_rmse:,.0f}</div>
            <div class="metric-label">Best RMSE (kWh)</div>
        </div>""", unsafe_allow_html=True)
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
            <div class="metric-label">Best Model</div>
        </div>""", unsafe_allow_html=True)
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
            <div class="metric-label">Forecasted kWh (Total)</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">—</div>
            <div class="metric-label">Forecasted kWh</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── SITE SELECTOR & FORECAST CHART ───────────────────────────────────────────
st.subheader("📈 Forecast vs Actual — Site View")

sites = sorted(historical['site_id'].unique())
selected_site = st.selectbox("Select a site", sites)

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
    mode='lines', name='Expected kWh',
    line=dict(color='#A0A0A0', width=1, dash='dash')
))

# Forecast
if forecasts is not None:
    fut_site = forecasts[forecasts['site_id'] == selected_site]
    if len(fut_site) > 0:
        fig.add_trace(go.Scatter(
            x=fut_site['month_year'], y=fut_site['forecast_kwh'],
            mode='lines+markers', name='Forecast',
            line=dict(color='#ED7D31', width=2, dash='dash'),
            marker=dict(size=5, symbol='square')
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fut_site['month_year'], fut_site['month_year'][::-1]]),
            y=pd.concat([fut_site['upper_bound'], fut_site['lower_bound'][::-1]]),
            fill='toself', fillcolor='rgba(237,125,49,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence interval (±15%)'
        ))

fig.update_layout(
    height=420,
    xaxis_title="Month",
    yaxis_title="Energy (kWh)",
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    plot_bgcolor='white',
    yaxis=dict(gridcolor='#f0f0f0'),
    xaxis=dict(gridcolor='#f0f0f0'),
    margin=dict(t=20, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🏆 Model Comparison")
    if evaluation is not None:
        eval_sorted = evaluation.sort_values('rmse')
        fig_eval = go.Figure(go.Bar(
            y=eval_sorted['model'],
            x=eval_sorted['rmse'],
            orientation='h',
            marker_color=['#1F4E79' if i == 0 else '#B8CCE4' for i in range(len(eval_sorted))],
            text=[f"{v:,.0f}" for v in eval_sorted['rmse']],
            textposition='outside'
        ))
        fig_eval.update_layout(
            height=280, xaxis_title="RMSE (kWh)",
            plot_bgcolor='white', margin=dict(t=10, b=10),
            xaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_eval, use_container_width=True)
    else:
        st.info("Run forecasting_pipeline.py to see model comparison")

with col_b:
    st.subheader("🔍 SHAP Feature Importance")
    if shap_df is not None:
        top10 = shap_df.head(10)
        fig_shap = go.Figure(go.Bar(
            y=top10['feature'][::-1],
            x=top10['importance'][::-1],
            orientation='h',
            marker_color='#2E75B6'
        ))
        fig_shap.update_layout(
            height=280, xaxis_title="Mean |SHAP Value|",
            plot_bgcolor='white', margin=dict(t=10, b=10),
            xaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("Run forecasting_pipeline.py to see feature importance")

# ── SITE RANKINGS ─────────────────────────────────────────────────────────────
st.subheader("📊 Site Performance Rankings")

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
        'site_id': 'Site', 'avg_performance_pct': 'Avg Performance %',
        'total_actual': 'Total kWh', 'months': 'Months of Data', 'status': 'Status'
    }),
    use_container_width=True, hide_index=True
)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#999; font-size:12px'>
Self-initiated ML project | Real solar operational data (anonymized) | 
Stack: Python · XGBoost · LightGBM · Scikit-learn · SHAP · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
