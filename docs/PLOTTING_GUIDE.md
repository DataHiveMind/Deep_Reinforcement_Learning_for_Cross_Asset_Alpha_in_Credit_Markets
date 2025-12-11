# Plotting Functions - Quick Reference

## Overview

All plotting functions in `utils/plotting.py` now support the `save_path` parameter for easy plot export.

## Usage Pattern

```python
from utils.plotting import PerformancePlotter

# Create and save plot in one step
PerformancePlotter.plot_cumulative_returns(
    returns=your_returns_series,
    save_path='plots/my_plot.png'  # ← Automatically saves at 300 DPI
)
```

## Available Plot Functions

### PerformancePlotter

- ✅ `plot_cumulative_returns(returns, benchmark=None, save_path=None)`
- ✅ `plot_drawdown(returns, save_path=None)`
- ✅ `plot_rolling_metrics(returns, metrics=['sharpe', 'volatility'], window=252, save_path=None)`
- ✅ `plot_returns_distribution(returns, bins=50, save_path=None)`

### RiskPlotter

- ✅ `plot_var_cvar(returns, confidence_levels=[0.95, 0.99], save_path=None)`
- ✅ `plot_correlation_matrix(returns_df, method='pearson', save_path=None)`
- ✅ `plot_rolling_correlation(returns1, returns2, window=60, save_path=None)`

### CreditPlotter

- ✅ `plot_spread_evolution(spreads_df, save_path=None)`
- ✅ `plot_spread_distribution(spreads, bins=30, save_path=None)`
- ✅ `plot_spread_by_rating(spreads_df, ratings, save_path=None)`

### TradingPlotter

- ✅ `plot_positions(positions, prices=None, save_path=None)`
- ✅ `plot_trade_pnl(trades_df, save_path=None)`

### Utility Functions

- ✅ `create_performance_report(returns, benchmark=None, positions=None, save_path=None)`

## Generate All Plots for README

### Option 1: Use the automated script

```bash
python scripts/generate_readme_plots.py
```

This creates 14 comprehensive plots in the `plots/` directory.

### Option 2: Use example patterns

```bash
python examples/plotting_examples.py
```

This shows individual examples of how to create and save each plot type.

### Option 3: Manual generation

```python
from utils.plotting import PerformancePlotter
import pandas as pd

# Your actual backtest results
returns = pd.read_csv('results/backtest_returns.csv')['returns']

# Generate plot
PerformancePlotter.plot_cumulative_returns(
    returns=returns,
    title="My Strategy Performance",
    save_path='plots/my_cumulative_returns.png'
)
```

## Plot Specifications

- **Format**: PNG
- **Resolution**: 300 DPI (publication quality)
- **Layout**: Tight (no wasted whitespace)
- **Size**: Configurable via `figsize` parameter

## Directory Structure

```text
plots/
├── .gitkeep
├── cumulative_returns.png
├── drawdown.png
├── rolling_metrics.png
├── var_cvar.png
└── ... (other plots)
```

## Using in README.md

### Basic usage

```markdown
![Cumulative Returns](plots/cumulative_returns.png)
```

### With sizing

```markdown
<img src="plots/cumulative_returns.png" alt="Cumulative Returns" width="800"/>
```

### Side-by-side

```markdown
| Performance | Risk Analysis |
|-------------|---------------|
| ![Performance](plots/cumulative_returns.png) | ![Risk](plots/var_cvar.png) |
```

## Tips

1. **Always create plots directory first:**

   ```python
   from pathlib import Path
   Path('plots').mkdir(exist_ok=True)
   ```

2. **Use descriptive filenames:**

   ```python
   save_path='plots/ppo_agent_vs_benchmark_2024.png'
   ```

3. **Organize by category:**

   ```python
   save_path='plots/performance/cumulative_returns.png'
   save_path='plots/risk/var_analysis.png'
   ```

4. **High DPI for presentations:**

   All plots are automatically saved at 300 DPI - perfect for papers and presentations!

5. **Version control:**

   - Add `plots/*.png` to `.gitignore` if you regenerate plots frequently
   - Or commit them if they're final results for documentation

## Example Workflow

```python
# 1. Run your backtest
from backtests.backtest_runner import BacktestRunner

results = BacktestRunner.run(agent='ppo', env='credit_spread')

# 2. Generate plots
from utils.plotting import PerformancePlotter, RiskPlotter

PerformancePlotter.plot_cumulative_returns(
    returns=results['returns'],
    save_path='plots/ppo_performance.png'
)

RiskPlotter.plot_var_cvar(
    returns=results['returns'],
    save_path='plots/ppo_risk.png'
)

# 3. Add to README.md
# See the plots appear automatically!
```

## Troubleshooting

**Issue**: Plot not saving

- ✅ **Solution**: Ensure directory exists first: `Path('plots').mkdir(exist_ok=True)`

**Issue**: Low quality plots

- ✅ **Solution**: Already fixed! All plots save at 300 DPI automatically

**Issue**: Too much whitespace

- ✅ **Solution**: Already handled via `bbox_inches='tight'`

**Issue**: Want different format (PDF, SVG)

- ✅ **Solution**: Just change extension: `save_path='plots/my_plot.pdf'`

## Next Steps

1. Run `python scripts/generate_readme_plots.py` to create sample plots
2. Review the generated plots in the `plots/` directory
3. Update your README.md with the plot references
4. Customize titles, colors, and labels as needed
