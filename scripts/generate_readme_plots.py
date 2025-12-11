"""
Generate plots for README.md documentation.

This script creates sample visualizations that can be used in the project README
to showcase the capabilities of the Deep RL Credit Alpha system.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.plotting import (
    CreditPlotter,
    PerformancePlotter,
    RiskPlotter,
    TradingPlotter,
    create_performance_report,
)


def create_plots_directory():
    """Create directory for saving plots."""
    plots_dir = project_root / 'plots'
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def generate_sample_data(days: int = 500) -> dict:
    """
    Generate realistic sample data for plotting.
    
    Args:
        days: Number of days to simulate
        
    Returns:
        Dictionary containing sample data
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate returns (strategy outperforms benchmark)
    strategy_returns = np.random.normal(0.0008, 0.012, days)
    benchmark_returns = np.random.normal(0.0004, 0.015, days)
    
    strategy_returns = pd.Series(strategy_returns, index=dates, name='Strategy')
    benchmark_returns = pd.Series(benchmark_returns, index=dates, name='Benchmark')
    
    # Generate multi-strategy returns
    multi_strategy = pd.DataFrame({
        'DQN': np.random.normal(0.0006, 0.013, days),
        'PPO': np.random.normal(0.0008, 0.011, days),
        'Constrained PPO': np.random.normal(0.0007, 0.009, days)
    }, index=dates)
    
    # Generate credit spreads
    spreads = pd.DataFrame({
        'AAA': 50 + np.cumsum(np.random.normal(0, 1, days)),
        'AA': 80 + np.cumsum(np.random.normal(0, 1.5, days)),
        'A': 120 + np.cumsum(np.random.normal(0, 2, days)),
        'BBB': 180 + np.cumsum(np.random.normal(0, 3, days)),
        'BB': 350 + np.cumsum(np.random.normal(0, 5, days)),
        'B': 550 + np.cumsum(np.random.normal(0, 8, days))
    }, index=dates)
    spreads = spreads.clip(lower=20)  # Floor spreads at 20 bps
    
    # Generate positions
    positions = pd.DataFrame({
        'IG_Corp': np.random.uniform(0.2, 0.8, days),
        'HY_Corp': np.random.uniform(0.1, 0.5, days),
        'CDX': np.random.uniform(-0.3, 0.3, days)
    }, index=dates)
    
    # Generate prices
    prices = pd.DataFrame({
        'IG_Corp': 100 + np.cumsum(np.random.normal(0, 0.5, days)),
        'HY_Corp': 100 + np.cumsum(np.random.normal(0, 1, days)),
        'CDX': 100 + np.cumsum(np.random.normal(0, 0.3, days))
    }, index=dates)
    
    # Generate trades
    n_trades = 100
    trade_indices = np.sort(np.random.choice(len(dates), n_trades, replace=False))
    trades = pd.DataFrame({
        'pnl': np.random.normal(500, 1500, n_trades),
        'size': np.random.uniform(10000, 100000, n_trades)
    }, index=dates[trade_indices])
    
    # Generate ratings series
    ratings = pd.Series({
        'AAA': 'AAA', 'AA': 'AA', 'A': 'A',
        'BBB': 'BBB', 'BB': 'BB', 'B': 'B'
    })
    
    return {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'multi_strategy': multi_strategy,
        'spreads': spreads,
        'positions': positions,
        'prices': prices,
        'trades': trades,
        'ratings': ratings
    }


def generate_all_plots(save_dir: Path):
    """
    Generate all plots for README.
    
    Args:
        save_dir: Directory to save plots
    """
    print("Generating sample data...")
    data = generate_sample_data()
    
    print("\n📊 Generating Performance Plots...")
    
    # 1. Cumulative returns
    print("  - Cumulative returns")
    PerformancePlotter.plot_cumulative_returns(
        returns=data['strategy_returns'],
        benchmark=data['benchmark_returns'],
        title="Strategy vs Benchmark Performance",
        save_path=str(save_dir / 'cumulative_returns.png')
    )
    
    # 2. Multi-strategy comparison
    print("  - Multi-strategy comparison")
    PerformancePlotter.plot_cumulative_returns(
        returns=data['multi_strategy'],
        title="Multi-Agent Performance Comparison",
        save_path=str(save_dir / 'multi_strategy_returns.png')
    )
    
    # 3. Drawdown
    print("  - Drawdown analysis")
    PerformancePlotter.plot_drawdown(
        returns=data['strategy_returns'],
        title="Strategy Drawdown",
        save_path=str(save_dir / 'drawdown.png')
    )
    
    # 4. Rolling metrics
    print("  - Rolling metrics")
    PerformancePlotter.plot_rolling_metrics(
        returns=data['strategy_returns'],
        metrics=['sharpe', 'volatility', 'return'],
        window=60,
        save_path=str(save_dir / 'rolling_metrics.png')
    )
    
    # 5. Returns distribution
    print("  - Returns distribution")
    PerformancePlotter.plot_returns_distribution(
        returns=data['strategy_returns'],
        save_path=str(save_dir / 'returns_distribution.png')
    )
    
    print("\n⚠️ Generating Risk Plots...")
    
    # 6. VaR and CVaR
    print("  - VaR and CVaR")
    RiskPlotter.plot_var_cvar(
        returns=data['strategy_returns'],
        confidence_levels=[0.95, 0.99],
        save_path=str(save_dir / 'var_cvar.png')
    )
    
    # 7. Correlation matrix
    print("  - Correlation matrix")
    RiskPlotter.plot_correlation_matrix(
        returns_df=data['multi_strategy'],
        save_path=str(save_dir / 'correlation_matrix.png')
    )
    
    # 8. Rolling correlation
    print("  - Rolling correlation")
    RiskPlotter.plot_rolling_correlation(
        returns1=data['strategy_returns'],
        returns2=data['benchmark_returns'],
        window=60,
        label1='Strategy',
        label2='Benchmark',
        save_path=str(save_dir / 'rolling_correlation.png')
    )
    
    print("\n💳 Generating Credit Market Plots...")
    
    # 9. Spread evolution
    print("  - Spread evolution")
    CreditPlotter.plot_spread_evolution(
        spreads_df=data['spreads'],
        title="Credit Spread Evolution by Rating",
        save_path=str(save_dir / 'spread_evolution.png')
    )
    
    # 10. Spread distribution
    print("  - Spread distribution")
    CreditPlotter.plot_spread_distribution(
        spreads=data['spreads']['BBB'],
        save_path=str(save_dir / 'spread_distribution.png')
    )
    
    # 11. Spread by rating
    print("  - Spread by rating")
    CreditPlotter.plot_spread_by_rating(
        spreads_df=data['spreads'],
        ratings=data['ratings'],
        save_path=str(save_dir / 'spread_by_rating.png')
    )
    
    print("\n💼 Generating Trading Activity Plots...")
    
    # 12. Positions
    print("  - Positions over time")
    TradingPlotter.plot_positions(
        positions=data['positions'],
        prices=data['prices'],
        save_path=str(save_dir / 'positions.png')
    )
    
    # 13. Trade P&L
    print("  - Trade P&L")
    TradingPlotter.plot_trade_pnl(
        trades_df=data['trades'],
        save_path=str(save_dir / 'trade_pnl.png')
    )
    
    # 14. Comprehensive performance report
    print("  - Comprehensive performance report")
    create_performance_report(
        returns=data['strategy_returns'],
        benchmark=data['benchmark_returns'],
        positions=data['positions'],
        save_path=str(save_dir / 'performance_report.png')
    )
    
    print(f"\n✅ All plots saved to: {save_dir}")
    print(f"\nGenerated {14} plots for README.md:")
    print("  • cumulative_returns.png")
    print("  • multi_strategy_returns.png")
    print("  • drawdown.png")
    print("  • rolling_metrics.png")
    print("  • returns_distribution.png")
    print("  • var_cvar.png")
    print("  • correlation_matrix.png")
    print("  • rolling_correlation.png")
    print("  • spread_evolution.png")
    print("  • spread_distribution.png")
    print("  • spread_by_rating.png")
    print("  • positions.png")
    print("  • trade_pnl.png")
    print("  • performance_report.png")


def main():
    """Main execution function."""
    print("=" * 60)
    print("README Plot Generator")
    print("Deep RL Credit Alpha System")
    print("=" * 60)
    
    # Create plots directory
    plots_dir = create_plots_directory()
    print(f"\nSaving plots to: {plots_dir}")
    
    # Generate all plots
    try:
        generate_all_plots(plots_dir)
        print("\n" + "=" * 60)
        print("✅ Success! All plots generated.")
        print("=" * 60)
        print("\nUsage in README.md:")
        print("  ![Cumulative Returns](plots/cumulative_returns.png)")
        print("  ![Drawdown](plots/drawdown.png)")
        print("  ... etc.")
        
    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
