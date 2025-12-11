"""
Example: How to use plotting functions with save_path parameter.

This script demonstrates how to generate and save plots from your backtesting results
for use in documentation, presentations, or reports.
"""

import sys
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
)


def example_basic_usage():
    """Example: Basic plot with save."""
    print("Example 1: Basic cumulative returns plot")
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    
    # Create and save plot
    PerformancePlotter.plot_cumulative_returns(
        returns=returns,
        title="Strategy Performance",
        save_path='plots/example_cumulative_returns.png'  # Saves automatically!
    )
    
    print("✅ Saved to: plots/example_cumulative_returns.png\n")


def example_with_benchmark():
    """Example: Plot with benchmark comparison."""
    print("Example 2: Strategy vs Benchmark with save")
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    strategy_returns = pd.Series(np.random.normal(0.0008, 0.01, len(dates)), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0004, 0.012, len(dates)), index=dates)
    
    # Create and save plot
    PerformancePlotter.plot_cumulative_returns(
        returns=strategy_returns,
        benchmark=benchmark_returns,
        title="RL Agent vs Benchmark",
        save_path='plots/example_vs_benchmark.png'
    )
    
    print("✅ Saved to: plots/example_vs_benchmark.png\n")


def example_risk_metrics():
    """Example: Risk analysis plots."""
    print("Example 3: VaR and CVaR visualization")
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    
    # Create and save VaR/CVaR plot
    RiskPlotter.plot_var_cvar(
        returns=returns,
        confidence_levels=[0.95, 0.99],
        save_path='plots/example_var_cvar.png'
    )
    
    print("✅ Saved to: plots/example_var_cvar.png\n")


def example_credit_spreads():
    """Example: Credit spread analysis."""
    print("Example 4: Credit spread evolution")
    
    # Generate sample spread data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    spreads = pd.DataFrame({
        'AAA': 50 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'A': 100 + np.cumsum(np.random.normal(0, 2, len(dates))),
        'BBB': 180 + np.cumsum(np.random.normal(0, 3, len(dates))),
        'BB': 350 + np.cumsum(np.random.normal(0, 5, len(dates)))
    }, index=dates)
    
    # Create and save plot
    CreditPlotter.plot_spread_evolution(
        spreads_df=spreads,
        title="Credit Spread Dynamics",
        save_path='plots/example_spreads.png'
    )
    
    print("✅ Saved to: plots/example_spreads.png\n")


def example_trading_positions():
    """Example: Trading positions visualization."""
    print("Example 5: Portfolio positions")
    
    # Generate sample position data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    positions = pd.DataFrame({
        'IG_Corp': np.random.uniform(0.2, 0.8, len(dates)),
        'HY_Corp': np.random.uniform(0.1, 0.5, len(dates)),
        'CDX': np.random.uniform(-0.3, 0.3, len(dates))
    }, index=dates)
    
    # Create and save plot
    TradingPlotter.plot_positions(
        positions=positions,
        save_path='plots/example_positions.png'
    )
    
    print("✅ Saved to: plots/example_positions.png\n")


def example_batch_generation():
    """Example: Generate multiple plots in batch."""
    print("Example 6: Batch plot generation")
    
    # Generate sample data once
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.0008, 0.01, len(dates)), index=dates)
    
    # Create output directory
    output_dir = Path('plots/batch_example')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate multiple plots
    plots = {
        'cumulative': PerformancePlotter.plot_cumulative_returns,
        'drawdown': PerformancePlotter.plot_drawdown,
        'returns_dist': PerformancePlotter.plot_returns_distribution,
        'var_cvar': RiskPlotter.plot_var_cvar
    }
    
    for name, plot_func in plots.items():
        save_path = output_dir / f'{name}.png'
        plot_func(returns=returns, save_path=str(save_path))
        print(f"  ✅ Saved: {save_path}")
    
    print()


def main():
    """Run all examples."""
    print("=" * 70)
    print("Plotting Examples with Save Functionality")
    print("=" * 70)
    print()
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Run examples
    example_basic_usage()
    example_with_benchmark()
    example_risk_metrics()
    example_credit_spreads()
    example_trading_positions()
    example_batch_generation()
    
    print("=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print()
    print("📁 Check the 'plots/' directory for generated images.")
    print()
    print("💡 Tips:")
    print("  • High resolution: All plots saved at 300 DPI")
    print("  • PNG format: Perfect for README.md and documentation")
    print("  • Tight layout: No whitespace wasted")
    print()
    print("🔗 Usage in README.md:")
    print("  ![Performance](plots/cumulative_returns.png)")
    print("  ![Risk Analysis](plots/var_cvar.png)")
    print()


if __name__ == '__main__':
    main()
