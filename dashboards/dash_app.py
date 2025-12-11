"""
Interactive Plotly Dash dashboard for monitoring RL trading agents.

This dashboard provides real-time visualization of:
- Agent performance metrics
- Trading signals and positions
- Credit spread dynamics
- Risk metrics and P&L
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.risk_metrics import RiskMetrics

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

app.title = "Credit Alpha RL Dashboard"

# Color scheme
COLORS = {
    'background': '#1e1e1e',
    'card': '#2d2d2d',
    'text': '#ffffff',
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'danger': '#ff6b6b'
}


def create_header() -> dbc.Container:
    """Create dashboard header."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1(
                    "Deep RL Credit Alpha Dashboard",
                    className="text-center mb-2",
                    style={'color': COLORS['primary']}
                ),
                html.P(
                    "Real-time monitoring of reinforcement learning trading agents",
                    className="text-center text-muted"
                )
            ])
        ], className="mb-4")
    ], fluid=True)


def create_metrics_cards() -> dbc.Row:
    """Create KPI metric cards."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Return", className="card-title"),
                    html.H2(id="total-return", children="--",
                            style={'color': COLORS['success']}),
                    html.P("YTD Performance", className="text-muted")
                ])
            ], style={'backgroundColor': COLORS['card']})
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Sharpe Ratio", className="card-title"),
                    html.H2(id="sharpe-ratio", children="--",
                            style={'color': COLORS['primary']}),
                    html.P("Risk-Adjusted Return", className="text-muted")
                ])
            ], style={'backgroundColor': COLORS['card']})
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Max Drawdown", className="card-title"),
                    html.H2(id="max-drawdown", children="--",
                            style={'color': COLORS['danger']}),
                    html.P("Peak to Trough", className="text-muted")
                ])
            ], style={'backgroundColor': COLORS['card']})
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Win Rate", className="card-title"),
                    html.H2(id="win-rate", children="--",
                            style={'color': COLORS['warning']}),
                    html.P("Profitable Trades", className="text-muted")
                ])
            ], style={'backgroundColor': COLORS['card']})
        ], width=3)
    ], className="mb-4")


def create_control_panel() -> dbc.Card:
    """Create control panel for user inputs."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Control Panel")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Agent Type"),
                    dcc.Dropdown(
                        id='agent-selector',
                        options=[
                            {'label': 'DQN', 'value': 'dqn'},
                            {'label': 'PPO', 'value': 'ppo'},
                            {'label': 'Constrained PPO', 'value': 'constrained'},
                        ],
                        value='ppo',
                        clearable=False,
                        style={'color': '#000'}
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=datetime.now() - timedelta(days=365),
                        end_date=datetime.now(),
                        display_format='YYYY-MM-DD'
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Benchmark"),
                    dcc.Dropdown(
                        id='benchmark-selector',
                        options=[
                            {'label': 'iBoxx USD Corporate', 'value': 'iboxx'},
                            {'label': 'CDX.NA.IG', 'value': 'cdx'},
                            {'label': 'Buy & Hold', 'value': 'bnh'},
                        ],
                        value='iboxx',
                        clearable=False,
                        style={'color': '#000'}
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Update"),
                    html.Br(),
                    dbc.Button(
                        "Refresh Data",
                        id="refresh-button",
                        color="primary",
                        className="w-100"
                    )
                ], width=2)
            ])
        ])
    ], className="mb-4", style={'backgroundColor': COLORS['card']})


def create_performance_section() -> dbc.Card:
    """Create performance visualization section."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Performance Analysis")),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id='cumulative-returns-graph'),
                    label="Cumulative Returns"
                ),
                dbc.Tab(
                    dcc.Graph(id='drawdown-graph'),
                    label="Drawdown"
                ),
                dbc.Tab(
                    dcc.Graph(id='rolling-metrics-graph'),
                    label="Rolling Metrics"
                ),
                dbc.Tab(
                    dcc.Graph(id='returns-distribution-graph'),
                    label="Returns Distribution"
                )
            ])
        ])
    ], className="mb-4", style={'backgroundColor': COLORS['card']})


def create_trading_section() -> dbc.Card:
    """Create trading activity visualization section."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Trading Activity")),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id='positions-graph'),
                    label="Positions"
                ),
                dbc.Tab(
                    dcc.Graph(id='trade-pnl-graph'),
                    label="Trade P&L"
                ),
                dbc.Tab(
                    dcc.Graph(id='turnover-graph'),
                    label="Turnover"
                )
            ])
        ])
    ], className="mb-4", style={'backgroundColor': COLORS['card']})


def create_credit_section() -> dbc.Card:
    """Create credit-specific analysis section."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Credit Market Analysis")),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id='spread-evolution-graph'),
                    label="Spread Evolution"
                ),
                dbc.Tab(
                    dcc.Graph(id='spread-distribution-graph'),
                    label="Spread Distribution"
                ),
                dbc.Tab(
                    dcc.Graph(id='rating-analysis-graph'),
                    label="Rating Analysis"
                )
            ])
        ])
    ], className="mb-4", style={'backgroundColor': COLORS['card']})


def create_risk_section() -> dbc.Card:
    """Create risk metrics visualization section."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Risk Analytics")),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id='var-cvar-graph'),
                    label="VaR & CVaR"
                ),
                dbc.Tab(
                    dcc.Graph(id='correlation-matrix-graph'),
                    label="Correlation Matrix"
                ),
                dbc.Tab(
                    dcc.Graph(id='rolling-correlation-graph'),
                    label="Rolling Correlation"
                )
            ])
        ])
    ], className="mb-4", style={'backgroundColor': COLORS['card']})


# Layout
app.layout = dbc.Container([
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0
    ),
    
    create_header(),
    create_metrics_cards(),
    create_control_panel(),
    
    dbc.Row([
        dbc.Col([
            create_performance_section(),
            create_trading_section()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            create_credit_section(),
            create_risk_section()
        ], width=12)
    ])
], fluid=True, style={'backgroundColor': COLORS['background']})


# Helper function to generate sample data
def generate_sample_data(agent_type: str, days: int = 365) -> Dict[str, Any]:
    """Generate sample data for demonstration."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(42)
    
    # Generate returns
    daily_returns = np.random.normal(0.0005, 0.01, days)
    if agent_type == 'ppo':
        daily_returns += 0.0002  # Slight positive drift
    
    cumulative = (1 + pd.Series(daily_returns, index=dates)).cumprod()
    
    return {
        'returns': pd.Series(daily_returns, index=dates),
        'cumulative': cumulative,
        'positions': pd.DataFrame({
            'long': np.random.uniform(0, 1, days),
            'short': np.random.uniform(-0.5, 0, days)
        }, index=dates),
        'spreads': pd.DataFrame({
            'IG': np.random.uniform(80, 150, days),
            'HY': np.random.uniform(300, 600, days)
        }, index=dates)
    }


# Callbacks
@app.callback(
    [Output('total-return', 'children'),
     Output('sharpe-ratio', 'children'),
     Output('max-drawdown', 'children'),
     Output('win-rate', 'children')],
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('agent-selector', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_metrics(n_clicks: Optional[int], n_intervals: int,
                   agent: str, start_date: str, end_date: str) -> tuple:
    """Update KPI metrics."""
    data = generate_sample_data(agent)
    returns = data['returns']
    
    # Calculate metrics
    total_return = (data['cumulative'].iloc[-1] - 1) * 100
    sharpe = RiskMetrics.sharpe_ratio(returns, risk_free_rate=0.02)
    max_dd = float(abs(RiskMetrics.max_drawdown(data['cumulative']))) * 100
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    return (
        f"{total_return:.2f}%",
        f"{sharpe:.2f}",
        f"-{max_dd:.2f}%",
        f"{win_rate:.1f}%"
    )


@app.callback(
    Output('cumulative-returns-graph', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('agent-selector', 'value'),
     State('benchmark-selector', 'value')]
)
def update_cumulative_returns(n_clicks: Optional[int], n_intervals: int,
                              agent: str, benchmark: str) -> go.Figure:
    """Update cumulative returns chart."""
    data = generate_sample_data(agent)
    benchmark_data = generate_sample_data('bnh')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['cumulative'].index,
        y=(data['cumulative'] - 1) * 100,
        mode='lines',
        name=f'{agent.upper()} Agent',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=benchmark_data['cumulative'].index,
        y=(benchmark_data['cumulative'] - 1) * 100,
        mode='lines',
        name='Benchmark',
        line=dict(color=COLORS['secondary'], width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig


@app.callback(
    Output('drawdown-graph', 'figure'),
    [Input('refresh-button', 'n_clicks')],
    [State('agent-selector', 'value')]
)
def update_drawdown(n_clicks: Optional[int], agent: str) -> go.Figure:
    """Update drawdown chart."""
    data = generate_sample_data(agent)
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
        line=dict(color=COLORS['danger'], width=2),
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.update_layout(
        title='Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig


@app.callback(
    Output('positions-graph', 'figure'),
    [Input('refresh-button', 'n_clicks')],
    [State('agent-selector', 'value')]
)
def update_positions(n_clicks: Optional[int], agent: str) -> go.Figure:
    """Update positions chart."""
    data = generate_sample_data(agent)
    positions = data['positions']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=positions.index,
        y=positions['long'],
        mode='lines',
        name='Long Positions',
        fill='tozeroy',
        line=dict(color=COLORS['success'], width=2),
        fillcolor='rgba(81, 207, 102, 0.3)'
    ))
    
    fig.add_trace(go.Scatter(
        x=positions.index,
        y=positions['short'],
        mode='lines',
        name='Short Positions',
        fill='tozeroy',
        line=dict(color=COLORS['danger'], width=2),
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.update_layout(
        title='Portfolio Positions Over Time',
        xaxis_title='Date',
        yaxis_title='Position Size',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig


@app.callback(
    Output('spread-evolution-graph', 'figure'),
    [Input('refresh-button', 'n_clicks')],
    [State('agent-selector', 'value')]
)
def update_spreads(n_clicks: Optional[int], agent: str) -> go.Figure:
    """Update credit spreads chart."""
    data = generate_sample_data(agent)
    spreads = data['spreads']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spreads.index,
        y=spreads['IG'],
        mode='lines',
        name='Investment Grade',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=spreads.index,
        y=spreads['HY'],
        mode='lines',
        name='High Yield',
        line=dict(color=COLORS['warning'], width=2)
    ))
    
    fig.update_layout(
        title='Credit Spread Evolution',
        xaxis_title='Date',
        yaxis_title='Spread (bps)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig


@app.callback(
    Output('var-cvar-graph', 'figure'),
    [Input('refresh-button', 'n_clicks')],
    [State('agent-selector', 'value')]
)
def update_var_cvar(n_clicks: Optional[int], agent: str) -> go.Figure:
    """Update VaR and CVaR chart."""
    data = generate_sample_data(agent)
    returns = data['returns']
    
    var_95 = RiskMetrics.value_at_risk(returns, confidence_level=0.95)
    cvar_95 = RiskMetrics.conditional_var(returns, confidence_level=0.95)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Returns Distribution',
        marker_color=COLORS['primary'],
        opacity=0.7
    ))
    
    fig.add_vline(
        x=var_95 * 100,
        line_dash="dash",
        line_color=COLORS['warning'],
        annotation_text=f"VaR (95%): {var_95*100:.2f}%"
    )
    
    fig.add_vline(
        x=cvar_95 * 100,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text=f"CVaR (95%): {cvar_95*100:.2f}%"
    )
    
    fig.update_layout(
        title='Value at Risk Analysis',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
