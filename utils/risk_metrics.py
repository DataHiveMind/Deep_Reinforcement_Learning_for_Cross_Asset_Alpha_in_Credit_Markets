"""
Risk Metrics Utilities for Credit Trading Strategies

This module provides comprehensive risk analytics for portfolio evaluation,
including Sharpe ratio, maximum drawdown, Value at Risk (VaR), CVaR, and
credit-specific risk metrics.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class RiskMetrics:
    """Calculate various risk metrics for portfolio returns."""

    @staticmethod
    def sharpe_ratio(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array or Series of returns
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        if np.std(excess_returns) == 0:
            return 0.0

        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def sortino_ratio(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sortino ratio (downside deviation).

        Args:
            returns: Array or Series of returns
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_deviation

    @staticmethod
    def max_drawdown(
        returns: Union[np.ndarray, pd.Series],
        return_series: bool = False
    ) -> Union[float, Tuple[float, pd.Series]]:
        """
        Calculate maximum drawdown from returns.

        Args:
            returns: Array or Series of returns
            return_series: If True, also return drawdown series

        Returns:
            Maximum drawdown (negative value) or tuple of (max_dd, drawdown_series)
        """
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()

        if return_series:
            return max_dd, drawdown
        return max_dd

    @staticmethod
    def calmar_ratio(
        returns: Union[np.ndarray, pd.Series],
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Array or Series of returns
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        annualized_return = np.mean(returns) * periods_per_year
        max_dd = RiskMetrics.max_drawdown(returns)

        if max_dd == 0:
            return 0.0

        return annualized_return / abs(max_dd)

    @staticmethod
    def value_at_risk(
        returns: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array or Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'cornish_fisher'

        Returns:
            VaR as a positive number (loss)
        """
        if len(returns) < 2:
            return 0.0

        if method == 'historical':
            return -np.percentile(returns, (1 - confidence_level) * 100)

        elif method == 'parametric':
            mu = np.mean(returns)
            sigma = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            return -(mu + z_score * sigma)

        elif method == 'cornish_fisher':
            mu = np.mean(returns)
            sigma = np.std(returns)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)

            z = stats.norm.ppf(1 - confidence_level)
            z_cf = (z + (z**2 - 1) * skew / 6 +
                    (z**3 - 3 * z) * kurt / 24 -
                    (2 * z**3 - 5 * z) * skew**2 / 36)

            return -(mu + z_cf * sigma)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    @staticmethod
    def conditional_var(
        returns: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Array or Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            CVaR as a positive number (expected loss beyond VaR)
        """
        if len(returns) < 2:
            return 0.0

        var = RiskMetrics.value_at_risk(returns, confidence_level, 'historical')
        returns_beyond_var = returns[returns <= -var]

        if len(returns_beyond_var) == 0:
            return var

        return -np.mean(returns_beyond_var)

    @staticmethod
    def information_ratio(
        returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Information Ratio (active return / tracking error).

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized Information Ratio
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns)

        if tracking_error == 0:
            return 0.0

        return np.sqrt(periods_per_year) * np.mean(active_returns) / tracking_error

    @staticmethod
    def omega_ratio(
        returns: Union[np.ndarray, pd.Series],
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio (probability-weighted gains / losses).

        Args:
            returns: Array or Series of returns
            threshold: Threshold return (default 0)

        Returns:
            Omega ratio
        """
        if len(returns) < 2:
            return 0.0

        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]

        if len(returns_below) == 0 or np.sum(returns_below) == 0:
            return np.inf if len(returns_above) > 0 else 0.0

        return np.sum(returns_above) / np.sum(returns_below)


class CreditRiskMetrics:
    """Credit-specific risk metrics for bond and spread trading."""

    @staticmethod
    def spread_duration(
        spread: float,
        duration: float,
        spread_change: float = 0.01
    ) -> float:
        """
        Calculate spread duration (sensitivity to spread changes).

        Args:
            spread: Current credit spread (in bps or decimal)
            duration: Modified duration
            spread_change: Spread change for sensitivity (default 1bp = 0.01)

        Returns:
            Spread duration
        """
        return -duration * spread_change / (1 + spread)

    @staticmethod
    def credit_var(
        spreads: Union[np.ndarray, pd.Series],
        notional: float,
        duration: float,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate credit-specific Value at Risk.

        Args:
            spreads: Historical spread changes
            notional: Notional amount
            duration: Modified duration
            confidence_level: Confidence level

        Returns:
            Credit VaR in currency units
        """
        spread_var = RiskMetrics.value_at_risk(spreads, confidence_level)
        return notional * duration * spread_var

    @staticmethod
    def jump_risk_metric(
        returns: Union[np.ndarray, pd.Series],
        threshold: float = 3.0
    ) -> Tuple[float, int]:
        """
        Identify jump risk in credit spreads.

        Args:
            returns: Array or Series of returns
            threshold: Number of standard deviations for jump identification

        Returns:
            Tuple of (jump frequency, number of jumps)
        """
        if len(returns) < 2:
            return 0.0, 0

        mean = np.mean(returns)
        std = np.std(returns)

        jumps = np.abs(returns - mean) > (threshold * std)
        n_jumps = np.sum(jumps)
        jump_frequency = n_jumps / len(returns)

        return jump_frequency, n_jumps

    @staticmethod
    def transition_probability(
        ratings: pd.Series,
        states: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Calculate credit rating transition probability matrix.

        Args:
            ratings: Series of credit ratings over time
            states: List of possible rating states (auto-detect if None)

        Returns:
            Transition probability matrix
        """
        if states is None:
            states = sorted(ratings.unique())

        n_states = len(states)
        transition_matrix = np.zeros((n_states, n_states))

        for i, current_state in enumerate(states):
            current_mask = ratings == current_state
            next_ratings = ratings[current_mask].shift(-1)

            for j, next_state in enumerate(states):
                transition_matrix[i, j] = np.sum(next_ratings == next_state)

        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums

        return pd.DataFrame(transition_matrix, index=states, columns=states)


def calculate_portfolio_metrics(
    returns: Union[np.ndarray, pd.Series],
    benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate comprehensive portfolio metrics.

    Args:
        returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary of risk metrics
    """
    metrics = {}

    # Return metrics
    metrics['total_return'] = (1 + pd.Series(returns)).prod() - 1
    metrics['annualized_return'] = np.mean(returns) * periods_per_year
    metrics['volatility'] = np.std(returns) * np.sqrt(periods_per_year)

    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = RiskMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year)
    metrics['sortino_ratio'] = RiskMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown metrics
    metrics['max_drawdown'] = RiskMetrics.max_drawdown(returns)
    metrics['calmar_ratio'] = RiskMetrics.calmar_ratio(returns, periods_per_year)

    # Risk metrics
    metrics['var_95'] = RiskMetrics.value_at_risk(returns, 0.95)
    metrics['cvar_95'] = RiskMetrics.conditional_var(returns, 0.95)
    metrics['var_99'] = RiskMetrics.value_at_risk(returns, 0.99)
    metrics['cvar_99'] = RiskMetrics.conditional_var(returns, 0.99)

    # Distribution metrics
    metrics['skewness'] = stats.skew(returns)
    metrics['kurtosis'] = stats.kurtosis(returns)

    # Benchmark comparison
    if benchmark_returns is not None:
        metrics['information_ratio'] = RiskMetrics.information_ratio(
            returns, benchmark_returns, periods_per_year
        )
        metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        metrics['alpha'] = metrics['annualized_return'] - (
            risk_free_rate + metrics['beta'] * (np.mean(benchmark_returns) * periods_per_year - risk_free_rate)
        )

    return metrics
