"""
Interactive Streamlit dashboard for monitoring RL trading agents.

This dashboard provides real-time visualization of:
- Agent performance metrics
- Trading signals and positions  
- Credit spread dynamics
- Risk metrics and P&L
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.risk_metrics import RiskMetrics

# Page configuration
st.set_page_config(
    page_title="Credit Alpha RL Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=300)
def generate_sample_data(agent_type: str, days: int = 365) -> Dict[str, Any]:
    """
    Generate sample data for demonstration.
    
    Args:
        agent_type: Type of agent (dqn, ppo, constrained)
        days: Number of days to generate
        
    Returns:
        Dictionary containing sample time series data
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(42)
    
    # Generate returns with agent-specific characteristics
    daily_returns = np.random.normal(0.0005, 0.01, days)
    if agent_type == 'ppo':
        daily_returns += 0.0002  # Better performance
    elif agent_type == 'constrained':
        daily_returns = np.clip(daily_returns, -0.02, 0.02)  # Risk-constrained
    
    cumulative = (1 + pd.Series(daily_returns, index=dates)).cumprod()
    
    # Generate positions
    positions = pd.DataFrame({
        'long': np.random.uniform(0, 1, days),
        'short': np.random.uniform(-0.5, 0, days),
        'net': np.random.uniform(-0.3, 0.7, days)
    }, index=dates)
    
    # Generate credit spreads
    spreads = pd.DataFrame({
        'IG': 100 + np.cumsum(np.random.normal(0, 2, days)),
        'HY': 400 + np.cumsum(np.random.normal(0, 5, days)),
        'BBB': 150 + np.cumsum(np.random.normal(0, 3, days))
    }, index=dates)
    spreads = spreads.clip(lower=50)  # Floor at 50 bps
    
    # Generate trade data
    n_trades = 200
    trade_dates = np.random.choice(dates, n_trades)
    trades = pd.DataFrame({
        'date': trade_dates,
        'pnl': np.random.normal(100, 500, n_trades),
        'size': np.random.uniform(10000, 100000, n_trades),
        'asset': np.random.choice(['IG Corp', 'HY Corp', 'CDX', 'iTraxx'], n_trades)
    }).sort_values('date')
    
    return {
        'returns': pd.Series(daily_returns, index=dates),
        'cumulative': cumulative,
        'positions': positions,
        'spreads': spreads,
        'trades': trades
    }


def create_sidebar() -> Dict[str, Any]:
    """Create sidebar with controls."""
    st.sidebar.title("📊 Control Panel")
    
    # Agent selection
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        options=['dqn', 'ppo', 'constrained'],
        index=1,
        format_func=lambda x: {
            'dqn': 'DQN',
            'ppo': 'PPO',
            'constrained': 'Constrained PPO'
        }[x]
    )
    
    # Date range
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=datetime.now()
        )
    
    # Benchmark selection
    benchmark = st.sidebar.selectbox(
        "Benchmark",
        options=['iboxx', 'cdx', 'bnh'],
        format_func=lambda x: {
            'iboxx': 'iBoxx USD Corporate',
            'cdx': 'CDX.NA.IG',
            'bnh': 'Buy & Hold'
        }[x]
    )
    
    # Update frequency
    st.sidebar.subheader("Settings")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=30
        )
    else:
        refresh_interval = None
    
    # Manual refresh button
    refresh = st.sidebar.button("🔄 Refresh Data", use_container_width=True)
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Dashboard Info**
    
    Real-time monitoring of Deep RL agents trading credit markets.
    
    - Performance metrics
    - Risk analytics
    - Trading activity
    - Credit spread analysis
    """)
    
    return {
        'agent_type': agent_type,
        'start_date': start_date,
        'end_date': end_date,
        'benchmark': benchmark,
        'refresh_interval': refresh_interval,
        'refresh': refresh
    }


def display_metrics(data: Dict[str, Any]) -> None:
    """Display KPI metrics."""
    returns = data['returns']
    cumulative = data['cumulative']
    
    # Calculate metrics
    total_return = (cumulative.iloc[-1] - 1) * 100
    sharpe = RiskMetrics.sharpe_ratio(returns, risk_free_rate=0.02)
    max_dd = float(abs(RiskMetrics.max_drawdown(cumulative))) * 100
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sortino ratio
    sortino = RiskMetrics.sortino_ratio(returns, risk_free_rate=0.02)
    
    # Display in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            delta=f"{returns.iloc[-1]*100:.2f}% (1D)"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="Annualized"
        )
    
    with col3:
        st.metric(
            "Sortino Ratio",
            f"{sortino:.2f}",
            delta="Downside Risk"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"-{max_dd:.2f}%",
            delta="Peak-to-Trough",
            delta_color="inverse"
        )
    
    with col5:
        st.metric(
            "Volatility",
            f"{volatility:.2f}%",
            delta="Annualized"
        )
    
    with col6:
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{len(data['trades'])} trades"
        )


def plot_performance(data: Dict[str, Any], benchmark: str) -> None:
    """Plot performance charts."""
    tab1, tab2, tab3, tab4 = st.tabs([
        "Cumulative Returns",
        "Drawdown",
        "Rolling Metrics",
        "Returns Distribution"
    ])
    
    with tab1:
        # Cumulative returns
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['cumulative'].index,
            y=(data['cumulative'] - 1) * 100,
            mode='lines',
            name='Agent',
            line=dict(color='#00d4ff', width=2)
        ))
        
        # Add benchmark (simulated)
        benchmark_returns = (data['cumulative'] * 0.8).values
        fig.add_trace(go.Scatter(
            x=data['cumulative'].index,
            y=(benchmark_returns - 1) * 100,
            mode='lines',
            name=benchmark.upper(),
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_dark",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Drawdown
        cumulative = data['cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#ff6b6b', width=2),
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))
        
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Rolling Sharpe and Volatility
        window = 30
        rolling_sharpe = data['returns'].rolling(window).apply(
            lambda x: RiskMetrics.sharpe_ratio(x, 0.02) if len(x) == window else np.nan
        )
        rolling_vol = data['returns'].rolling(window).std() * np.sqrt(252) * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Rolling Sharpe Ratio (30D)", "Rolling Volatility (30D)"),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Sharpe',
                line=dict(color='#51cf66', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#ffd43b', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            hovermode='x unified',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Returns distribution
        returns_pct = data['returns'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name='Returns',
            marker_color='#00d4ff',
            opacity=0.7
        ))
        
        # Add VaR and CVaR lines
        var_95 = RiskMetrics.value_at_risk(data['returns'], 0.95) * 100
        cvar_95 = RiskMetrics.conditional_var(data['returns'], 0.95) * 100
        
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color='#ffd43b',
            annotation_text=f"VaR (95%): {var_95:.2f}%"
        )
        
        fig.add_vline(
            x=cvar_95,
            line_dash="dash",
            line_color='#ff6b6b',
            annotation_text=f"CVaR (95%): {cvar_95:.2f}%"
        )
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)


def plot_trading_activity(data: Dict[str, Any]) -> None:
    """Plot trading activity charts."""
    tab1, tab2, tab3 = st.tabs(["Positions", "Trade P&L", "Turnover"])
    
    with tab1:
        # Positions over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['positions'].index,
            y=data['positions']['long'],
            mode='lines',
            name='Long',
            fill='tozeroy',
            line=dict(color='#51cf66', width=2),
            fillcolor='rgba(81, 207, 102, 0.3)'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['positions'].index,
            y=data['positions']['short'],
            mode='lines',
            name='Short',
            fill='tozeroy',
            line=dict(color='#ff6b6b', width=2),
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['positions'].index,
            y=data['positions']['net'],
            mode='lines',
            name='Net',
            line=dict(color='#00d4ff', width=3, dash='dot')
        ))
        
        fig.update_layout(
            title="Portfolio Positions",
            xaxis_title="Date",
            yaxis_title="Position Size",
            template="plotly_dark",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Trade P&L
        trades = data['trades']
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Individual Trade P&L", "Cumulative P&L"),
            vertical_spacing=0.12
        )
        
        # Individual trades
        colors = ['#51cf66' if pnl > 0 else '#ff6b6b' for pnl in trades['pnl']]
        fig.add_trace(
            go.Bar(
                x=trades['date'],
                y=trades['pnl'],
                name='Trade P&L',
                marker_color=colors
            ),
            row=1, col=1
        )
        
        # Cumulative P&L
        fig.add_trace(
            go.Scatter(
                x=trades['date'],
                y=trades['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='#00d4ff', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            hovermode='x unified',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            st.metric("Winning Trades", f"{(trades['pnl'] > 0).sum()}")
        with col3:
            st.metric("Avg Trade P&L", f"${trades['pnl'].mean():.2f}")
    
    with tab3:
        # Portfolio turnover
        positions_diff = data['positions']['net'].diff().abs()
        turnover = positions_diff.rolling(30).sum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=turnover.index,
            y=turnover.values,
            mode='lines',
            name='30D Turnover',
            line=dict(color='#ffd43b', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 212, 59, 0.3)'
        ))
        
        fig.update_layout(
            title="Portfolio Turnover (30-Day Rolling)",
            xaxis_title="Date",
            yaxis_title="Turnover",
            template="plotly_dark",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def plot_credit_analysis(data: Dict[str, Any]) -> None:
    """Plot credit market analysis."""
    tab1, tab2, tab3 = st.tabs([
        "Spread Evolution",
        "Spread Distribution",
        "Rating Analysis"
    ])
    
    with tab1:
        # Spread evolution
        fig = go.Figure()
        
        for col, color in zip(['IG', 'HY', 'BBB'],
                             ['#00d4ff', '#ff6b6b', '#ffd43b']):
            fig.add_trace(go.Scatter(
                x=data['spreads'].index,
                y=data['spreads'][col],
                mode='lines',
                name=col,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title="Credit Spread Evolution",
            xaxis_title="Date",
            yaxis_title="Spread (bps)",
            template="plotly_dark",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Spread distribution
        fig = go.Figure()
        
        for col, color in zip(['IG', 'HY', 'BBB'],
                             ['#00d4ff', '#ff6b6b', '#ffd43b']):
            fig.add_trace(go.Histogram(
                x=data['spreads'][col],
                name=col,
                marker_color=color,
                opacity=0.6,
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Credit Spread Distribution",
            xaxis_title="Spread (bps)",
            yaxis_title="Frequency",
            template="plotly_dark",
            barmode='overlay',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        for col_idx, (col, label) in enumerate(zip(['IG', 'HY', 'BBB'],
                                                   ['IG', 'HY', 'BBB'])):
            with [col1, col2, col3][col_idx]:
                st.metric(
                    f"{label} Avg Spread",
                    f"{data['spreads'][col].mean():.1f} bps",
                    delta=f"±{data['spreads'][col].std():.1f} bps"
                )
    
    with tab3:
        # Rating migration simulation
        ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        migration_matrix = np.random.dirichlet(np.ones(7), size=7)
        
        fig = go.Figure(data=go.Heatmap(
            z=migration_matrix,
            x=ratings,
            y=ratings,
            colorscale='RdYlGn_r',
            text=np.round(migration_matrix * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Probability (%)")
        ))
        
        fig.update_layout(
            title="Rating Migration Matrix (Simulated)",
            xaxis_title="To Rating",
            yaxis_title="From Rating",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def plot_risk_analytics(data: Dict[str, Any]) -> None:
    """Plot risk analytics."""
    tab1, tab2, tab3 = st.tabs([
        "VaR & CVaR",
        "Correlation Matrix",
        "Risk Decomposition"
    ])
    
    with tab1:
        # VaR and CVaR
        returns = data['returns']
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='#00d4ff',
            opacity=0.7
        ))
        
        # Multiple confidence levels
        for conf, color in [(0.95, '#ffd43b'), (0.99, '#ff6b6b')]:
            var = RiskMetrics.value_at_risk(returns, conf) * 100
            
            fig.add_vline(
                x=var,
                line_dash="dash",
                line_color=color,
                annotation_text=f"VaR ({int(conf*100)}%): {var:.2f}%",
                annotation_position="top"
            )
        
        fig.update_layout(
            title="Value at Risk Analysis",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics table
        col1, col2, col3 = st.columns(3)
        with col1:
            var_95 = RiskMetrics.value_at_risk(returns, 0.95) * 100
            st.metric("VaR (95%)", f"{var_95:.2f}%")
        with col2:
            cvar_95 = RiskMetrics.conditional_var(returns, 0.95) * 100
            st.metric("CVaR (95%)", f"{cvar_95:.2f}%")
        with col3:
            st.metric("Skewness", f"{returns.skew():.2f}")
    
    with tab2:
        # Correlation matrix (simulated multi-asset)
        assets = ['IG Corp', 'HY Corp', 'Equity', 'Rates', 'FX']
        corr_matrix = np.random.uniform(0.3, 0.9, (5, 5))
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=assets,
            y=assets,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Cross-Asset Correlation Matrix",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Risk decomposition
        risk_sources = ['Market Risk', 'Credit Risk', 'Liquidity Risk',
                       'Model Risk', 'Operational Risk']
        risk_values = np.random.uniform(10, 40, 5)
        risk_values = risk_values / risk_values.sum() * 100
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_sources,
            values=risk_values,
            hole=0.4,
            marker_colors=['#00d4ff', '#51cf66', '#ffd43b', '#ff6b6b', '#9775fa']
        )])
        
        fig.update_layout(
            title="Risk Decomposition",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application."""
    # Title
    st.title("📊 Deep RL Credit Alpha Dashboard")
    st.markdown("Real-time monitoring of reinforcement learning trading agents")
    
    # Sidebar controls
    controls = create_sidebar()
    
    # Load data
    with st.spinner("Loading data..."):
        data = generate_sample_data(controls['agent_type'])
    
    # Display metrics
    st.markdown("---")
    display_metrics(data)
    
    # Performance section
    st.markdown("---")
    st.header("📈 Performance Analysis")
    plot_performance(data, controls['benchmark'])
    
    # Trading activity section
    st.markdown("---")
    st.header("💼 Trading Activity")
    plot_trading_activity(data)
    
    # Credit analysis section
    st.markdown("---")
    st.header("💳 Credit Market Analysis")
    plot_credit_analysis(data)
    
    # Risk analytics section
    st.markdown("---")
    st.header("⚠️ Risk Analytics")
    plot_risk_analytics(data)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Deep Reinforcement Learning for Cross-Asset Alpha in Credit Markets</p>
        <p>Dashboard v1.0 | Last updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )
    
    # Auto-refresh
    if controls['refresh_interval']:
        st.rerun()


if __name__ == '__main__':
    main()
