"""
Plotting Utilities for Credit Trading Analysis

This module provides comprehensive visualization tools for backtesting results,
portfolio performance, risk metrics, and credit market analysis.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PerformancePlotter:
    """Plot portfolio performance and returns."""

    @staticmethod
    def plot_cumulative_returns(
        returns: Union[pd.Series, pd.DataFrame],
        benchmark: Optional[pd.Series] = None,
        title: str = "Cumulative Returns",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot cumulative returns over time.

        Args:
            returns: Returns series or DataFrame (multiple strategies)
            benchmark: Optional benchmark returns
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if isinstance(returns, pd.Series):
            cumulative = (1 + returns).cumprod()
            ax.plot(cumulative.index, cumulative.values, label='Strategy', linewidth=2)
        else:
            for col in returns.columns:
                cumulative = (1 + returns[col]).cumprod()
                ax.plot(cumulative.index, cumulative.values, label=col, linewidth=2)

        if benchmark is not None:
            cumulative_bench = (1 + benchmark).cumprod()
            ax.plot(cumulative_bench.index, cumulative_bench.values,
                    label='Benchmark', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_drawdown(
        returns: pd.Series,
        title: str = "Drawdown",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot drawdown over time.

        Args:
            returns: Returns series
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        ax.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_rolling_metrics(
        returns: pd.Series,
        metrics: List[str] = ['sharpe', 'volatility'],
        window: int = 252,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot rolling performance metrics.

        Args:
            returns: Returns series
            metrics: List of metrics to plot
            window: Rolling window size
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric == 'sharpe':
                rolling_metric = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
                ax.set_ylabel('Sharpe Ratio')
                title = f'Rolling {window}-day Sharpe Ratio'
            elif metric == 'volatility':
                rolling_metric = returns.rolling(window).std() * np.sqrt(252)
                ax.set_ylabel('Volatility')
                title = f'Rolling {window}-day Volatility'
            elif metric == 'return':
                rolling_metric = returns.rolling(window).mean() * 252
                ax.set_ylabel('Return')
                title = f'Rolling {window}-day Annualized Return'
            else:
                continue

            ax.plot(rolling_metric.index, rolling_metric.values, linewidth=2)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        axes[-1].set_xlabel('Date')
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        bins: int = 50,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot returns distribution with statistics.

        Args:
            returns: Returns series
            bins: Number of histogram bins
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(returns.dropna(), bins=bins, alpha=0.7, color='blue', edgecolor='black')

        # Add statistics
        mean = returns.mean()
        std = returns.std()
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        ax.axvline(mean + std, color='orange', linestyle='--', linewidth=1, label=f'±1 Std: {std:.4f}')
        ax.axvline(mean - std, color='orange', linestyle='--', linewidth=1)

        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig


class RiskPlotter:
    """Plot risk metrics and analysis."""

    @staticmethod
    def plot_var_cvar(
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot VaR and CVaR on returns distribution.

        Args:
            returns: Returns series
            confidence_levels: List of confidence levels
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(returns.dropna(), bins=50, alpha=0.6, color='blue', edgecolor='black')

        # Add VaR and CVaR lines
        colors = ['red', 'darkred']
        for conf, color in zip(confidence_levels, colors):
            var = np.percentile(returns, (1 - conf) * 100)
            cvar = returns[returns <= var].mean()

            ax.axvline(var, color=color, linestyle='--', linewidth=2,
                       label=f'VaR {conf*100}%: {var:.4f}')
            ax.axvline(cvar, color=color, linestyle=':', linewidth=2,
                       label=f'CVaR {conf*100}%: {cvar:.4f}')

        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        ax.set_title('Value at Risk (VaR) and Conditional VaR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_correlation_matrix(
        returns_df: pd.DataFrame,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            returns_df: DataFrame of returns (multiple assets)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        corr_matrix = returns_df.corr(method=method)

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, ax=ax,
                    cbar_kws={'label': 'Correlation'})

        ax.set_title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_rolling_correlation(
        returns1: pd.Series,
        returns2: pd.Series,
        window: int = 60,
        label1: str = 'Asset 1',
        label2: str = 'Asset 2',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot rolling correlation between two return series.

        Args:
            returns1: First returns series
            returns2: Second returns series
            window: Rolling window size
            label1: Label for first asset
            label2: Label for second asset
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        rolling_corr = returns1.rolling(window).corr(returns2)

        ax.plot(rolling_corr.index, rolling_corr.values, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5)
        ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Rolling {window}-day Correlation: {label1} vs {label2}')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig


class CreditPlotter:
    """Plot credit-specific analytics."""

    @staticmethod
    def plot_spread_evolution(
        spreads_df: pd.DataFrame,
        title: str = "Credit Spread Evolution",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot credit spread evolution over time.

        Args:
            spreads_df: DataFrame with spreads (time x assets)
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for col in spreads_df.columns:
            ax.plot(spreads_df.index, spreads_df[col], label=col, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Spread (bps)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_spread_distribution(
        spreads: pd.Series,
        bins: int = 30,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot credit spread distribution.

        Args:
            spreads: Credit spread series
            bins: Number of bins
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(spreads.dropna(), bins=bins, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(spreads.median(), color='red', linestyle='--',
                    linewidth=2, label=f'Median: {spreads.median():.2f}')
        ax1.set_xlabel('Spread (bps)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Spread Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(spreads.dropna(), vert=True)
        ax2.set_ylabel('Spread (bps)')
        ax2.set_title('Spread Box Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_spread_by_rating(
        spreads_df: pd.DataFrame,
        ratings: pd.Series,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot spreads grouped by credit rating.

        Args:
            spreads_df: DataFrame of spreads
            ratings: Series of credit ratings
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Group by rating
        rating_groups = {}
        for rating in ratings.unique():
            cols = ratings[ratings == rating].index
            rating_groups[rating] = spreads_df[cols].mean(axis=1)

        for rating, spread_series in rating_groups.items():
            ax.plot(spread_series.index, spread_series.values, label=rating, linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Average Spread (bps)')
        ax.set_title('Credit Spreads by Rating')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig


class TradingPlotter:
    """Plot trading signals and positions."""

    @staticmethod
    def plot_positions(
        positions: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot positions over time.

        Args:
            positions: DataFrame of positions
            prices: Optional price data
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if prices is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

            # Plot prices
            for col in prices.columns:
                ax1.plot(prices.index, prices[col], label=col, alpha=0.7)
            ax1.set_ylabel('Price')
            ax1.set_title('Prices')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Plot positions
            for col in positions.columns:
                ax2.plot(positions.index, positions[col], label=col, alpha=0.7)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Position')
            ax2.set_title('Positions')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        else:
            fig, ax = plt.subplots(figsize=figsize)

            for col in positions.columns:
                ax.plot(positions.index, positions[col], label=col, alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Position')
            ax.set_title('Positions Over Time')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_trade_pnl(
        trades_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot individual trade P&L.

        Args:
            trades_df: DataFrame with trade info (must have 'pnl' column)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Cumulative P&L
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax1.plot(cumulative_pnl.index, cumulative_pnl.values, linewidth=2)
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative P&L')
        ax1.set_title('Cumulative P&L')
        ax1.grid(True, alpha=0.3)

        # P&L distribution
        ax2.hist(trades_df['pnl'].dropna(), bins=30, alpha=0.7,
                 color='green', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('P&L per Trade')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Trade P&L Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def create_performance_report(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    positions: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive performance report.

    Args:
        returns: Strategy returns
        benchmark: Optional benchmark returns
        positions: Optional positions data
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    cumulative = (1 + returns).cumprod()
    ax1.plot(cumulative.index, cumulative.values, label='Strategy', linewidth=2)
    if benchmark is not None:
        cumulative_bench = (1 + benchmark).cumprod()
        ax1.plot(cumulative_bench.index, cumulative_bench.values,
                 label='Benchmark', linestyle='--', linewidth=2)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Cumulative Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Drawdown')
    ax2.grid(True, alpha=0.3)

    # Returns distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(returns.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Returns Distribution')
    ax3.grid(True, alpha=0.3)

    # Rolling Sharpe
    ax4 = fig.add_subplot(gs[2, 0])
    rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_title('Rolling 60-day Sharpe Ratio')
    ax4.grid(True, alpha=0.3)

    # Monthly returns heatmap
    ax5 = fig.add_subplot(gs[2, 1])
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    if len(monthly_returns) > 0:
        monthly_pivot = monthly_returns.to_frame('ret')
        monthly_pivot['year'] = monthly_pivot.index.year
        monthly_pivot['month'] = monthly_pivot.index.month
        monthly_pivot = monthly_pivot.pivot(index='year', columns='month', values='ret')
        sns.heatmap(monthly_pivot, annot=True, fmt='.2%', cmap='RdYlGn',
                    center=0, ax=ax5, cbar_kws={'label': 'Monthly Return'})
        ax5.set_title('Monthly Returns')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
