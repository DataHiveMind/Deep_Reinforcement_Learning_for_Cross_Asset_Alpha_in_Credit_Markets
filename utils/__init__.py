"""
Utilities Package for Credit Alpha RL

Provides risk metrics, volatility models, and plotting utilities for
portfolio analysis and backtesting.
"""

from .plotting import (
    CreditPlotter,
    PerformancePlotter,
    RiskPlotter,
    TradingPlotter,
    create_performance_report,
)
from .risk_metrics import CreditRiskMetrics, RiskMetrics, calculate_portfolio_metrics
from .volatility_models import (
    CreditSpreadVolatility,
    EWMAVolatility,
    GARCHVolatility,
    GarmanKlassVolatility,
    ParkinsonVolatility,
    VolatilityForecaster,
    YangZhangVolatility,
    realized_volatility,
)

__all__ = [
    # Risk metrics
    'RiskMetrics',
    'CreditRiskMetrics',
    'calculate_portfolio_metrics',

    # Volatility models
    'EWMAVolatility',
    'GARCHVolatility',
    'ParkinsonVolatility',
    'GarmanKlassVolatility',
    'YangZhangVolatility',
    'CreditSpreadVolatility',
    'VolatilityForecaster',
    'realized_volatility',

    # Plotting
    'PerformancePlotter',
    'RiskPlotter',
    'CreditPlotter',
    'TradingPlotter',
    'create_performance_report',
]
