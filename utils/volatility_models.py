"""
Volatility Models for Credit Markets

This module implements various volatility models including GARCH, EWMA,
and credit-specific volatility estimators for risk management and trading.
"""

import warnings
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from arch import arch_model


class EWMAVolatility:
    """Exponentially Weighted Moving Average volatility model."""

    def __init__(self, lambda_param: float = 0.94):
        """
        Initialize EWMA volatility model.

        Args:
            lambda_param: Decay factor (typically 0.94 for daily data)
        """
        self.lambda_param = lambda_param
        self.volatility_series = None

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> 'EWMAVolatility':
        """
        Fit EWMA model to returns.

        Args:
            returns: Array or Series of returns

        Returns:
            Self for method chaining
        """
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns

        # Initialize with sample variance
        variance = np.var(returns[:30]) if len(returns) > 30 else np.var(returns)
        variances = [variance]

        # Calculate EWMA variance
        for ret in returns[1:]:
            variance = self.lambda_param * variance + (1 - self.lambda_param) * ret**2
            variances.append(variance)

        self.volatility_series = pd.Series(np.sqrt(variances), index=returns.index)
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Forecast volatility for future steps.

        Args:
            steps: Number of steps to forecast

        Returns:
            Array of forecasted volatilities
        """
        if self.volatility_series is None:
            raise ValueError("Model must be fitted before prediction")

        current_vol = self.volatility_series.iloc[-1]
        long_run_vol = self.volatility_series.mean()

        forecasts = []
        for h in range(1, steps + 1):
            # EWMA forecast converges to long-run average
            weight = self.lambda_param ** h
            forecast = weight * current_vol + (1 - weight) * long_run_vol
            forecasts.append(forecast)

        return np.array(forecasts)

    def get_current_volatility(self) -> float:
        """Get the most recent volatility estimate."""
        if self.volatility_series is None:
            raise ValueError("Model must be fitted first")
        return self.volatility_series.iloc[-1]


class GARCHVolatility:
    """GARCH(1,1) volatility model using arch package."""

    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model.

        Args:
            p: Order of GARCH terms
            q: Order of ARCH terms
        """
        self.p = p
        self.q = q
        self.model = None
        self.results = None

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series],
        dist: str = 't'
    ) -> 'GARCHVolatility':
        """
        Fit GARCH model to returns.

        Args:
            returns: Array or Series of returns
            dist: Distribution ('normal', 't', 'skewt')

        Returns:
            Self for method chaining
        """
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        returns_pct = returns * 100  # Scale for numerical stability

        self.model = arch_model(
            returns_pct,
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=dist  # type: ignore[arg-type]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results = self.model.fit(disp='off', show_warning=False)

        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Forecast volatility for future steps.

        Args:
            steps: Number of steps to forecast

        Returns:
            Array of forecasted volatilities
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.results.forecast(horizon=steps)
        variance_forecast = forecast.variance.values[-1, :]

        return np.sqrt(variance_forecast) / 100  # Scale back

    def get_conditional_volatility(self) -> pd.Series:
        """Get the conditional volatility series."""
        if self.results is None:
            raise ValueError("Model must be fitted first")
        vol_series = self.results.conditional_volatility / 100
        return pd.Series(vol_series) if isinstance(vol_series, np.ndarray) else vol_series

    def get_parameters(self) -> dict:
        """Get fitted model parameters."""
        if self.results is None:
            raise ValueError("Model must be fitted first")
        return self.results.params.to_dict()


class ParkinsonVolatility:
    """Parkinson (High-Low) volatility estimator."""

    @staticmethod
    def calculate(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Parkinson volatility using high-low range.

        Args:
            high: High prices
            low: Low prices
            window: Rolling window size

        Returns:
            Series of volatility estimates
        """
        high = pd.Series(high) if isinstance(high, np.ndarray) else high
        low = pd.Series(low) if isinstance(low, np.ndarray) else low

        # Parkinson formula
        hl_ratio = np.log(high / low)
        parkinson_vol = np.sqrt(hl_ratio**2 / (4 * np.log(2)))

        # Apply rolling window
        return parkinson_vol.rolling(window=window).mean()


class GarmanKlassVolatility:
    """Garman-Klass volatility estimator (uses OHLC data)."""

    @staticmethod
    def calculate(
        open_price: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility.

        Args:
            open_price: Opening prices
            high: High prices
            low: Low prices
            close: Closing prices
            window: Rolling window size

        Returns:
            Series of volatility estimates
        """
        open_price = pd.Series(open_price) if isinstance(open_price, np.ndarray) else open_price
        high = pd.Series(high) if isinstance(high, np.ndarray) else high
        low = pd.Series(low) if isinstance(low, np.ndarray) else low
        close = pd.Series(close) if isinstance(close, np.ndarray) else close

        # Garman-Klass formula
        hl = np.log(high / low)
        co = np.log(close / open_price)

        gk_vol = np.sqrt(0.5 * hl**2 - (2 * np.log(2) - 1) * co**2)

        # Apply rolling window
        return gk_vol.rolling(window=window).mean()


class YangZhangVolatility:
    """Yang-Zhang volatility estimator (drift-independent OHLC)."""

    @staticmethod
    def calculate(
        open_price: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Yang-Zhang volatility.

        Args:
            open_price: Opening prices
            high: High prices
            low: Low prices
            close: Closing prices
            window: Rolling window size

        Returns:
            Series of volatility estimates
        """
        open_price = pd.Series(open_price) if isinstance(open_price, np.ndarray) else open_price
        high = pd.Series(high) if isinstance(high, np.ndarray) else high
        low = pd.Series(low) if isinstance(low, np.ndarray) else low
        close = pd.Series(close) if isinstance(close, np.ndarray) else close

        # Log returns
        log_ho = np.log(high / open_price)
        log_lo = np.log(low / open_price)
        log_co = np.log(close / open_price)

        log_oc = np.log(open_price / close.shift(1))
        log_cc = np.log(close / close.shift(1))

        # Rogers-Satchell
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        # Overnight and close-to-close
        log_oc_series = pd.Series(log_oc) if isinstance(log_oc, np.ndarray) else log_oc
        log_cc_series = pd.Series(log_cc) if isinstance(log_cc, np.ndarray) else log_cc
        rs_series = pd.Series(rs) if isinstance(rs, np.ndarray) else rs
        
        overnight_var = log_oc_series.rolling(window=window).var()
        close_var = log_cc_series.rolling(window=window).var()
        rs_var = rs_series.rolling(window=window).mean()

        # Yang-Zhang
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * close_var + (1 - k) * rs_var

        return np.sqrt(yz_var)


class CreditSpreadVolatility:
    """Credit spread-specific volatility models."""

    @staticmethod
    def spread_volatility(
        spreads: Union[np.ndarray, pd.Series],
        method: str = 'ewma',
        window: int = 20,
        lambda_param: float = 0.94
    ) -> pd.Series:
        """
        Calculate credit spread volatility.

        Args:
            spreads: Credit spread series
            method: 'ewma', 'rolling', or 'expanding'
            window: Window size for rolling method
            lambda_param: Decay parameter for EWMA

        Returns:
            Volatility series
        """
        spreads = pd.Series(spreads) if isinstance(spreads, np.ndarray) else spreads
        spread_changes = spreads.diff().dropna()

        if method == 'ewma':
            vol_model = EWMAVolatility(lambda_param=lambda_param)
            vol_model.fit(spread_changes)
            if vol_model.volatility_series is not None:
                return vol_model.volatility_series
            else:
                raise ValueError("EWMA volatility calculation failed")

        elif method == 'rolling':
            return spread_changes.rolling(window=window).std()

        elif method == 'expanding':
            return spread_changes.expanding().std()

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def regime_conditional_volatility(
        returns: Union[np.ndarray, pd.Series],
        regimes: Union[np.ndarray, pd.Series]
    ) -> dict:
        """
        Calculate volatility conditional on market regimes.

        Args:
            returns: Return series
            regimes: Regime labels (e.g., 'high_vol', 'low_vol')

        Returns:
            Dictionary of regime-specific volatilities
        """
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        regimes = pd.Series(regimes) if isinstance(regimes, np.ndarray) else regimes

        regime_vols = {}
        for regime in regimes.unique():
            regime_returns = returns[regimes == regime]
            regime_vols[regime] = regime_returns.std()

        return regime_vols

    @staticmethod
    def cross_sectional_volatility(
        spread_matrix: pd.DataFrame,
        axis: int = 0
    ) -> pd.Series:
        """
        Calculate cross-sectional volatility across multiple credits.

        Args:
            spread_matrix: DataFrame with spreads (assets x time)
            axis: 0 for time-series vol, 1 for cross-sectional vol

        Returns:
            Volatility series
        """
        return spread_matrix.std(axis=axis)  # type: ignore[arg-type]


class VolatilityForecaster:
    """Ensemble volatility forecasting with multiple models."""

    def __init__(self):
        """Initialize volatility forecaster."""
        self.models = {}
        self.weights = {}

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series],
        models: Optional[list] = None
    ) -> 'VolatilityForecaster':
        """
        Fit multiple volatility models.

        Args:
            returns: Return series
            models: List of model names ['ewma', 'garch', 'historical']

        Returns:
            Self for method chaining
        """
        if models is None:
            models = ['ewma', 'garch', 'historical']

        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns

        # Fit EWMA
        if 'ewma' in models:
            ewma = EWMAVolatility(lambda_param=0.94)
            ewma.fit(returns)
            self.models['ewma'] = ewma

        # Fit GARCH
        if 'garch' in models:
            try:
                garch = GARCHVolatility(p=1, q=1)
                garch.fit(returns)
                self.models['garch'] = garch
            except Exception as e:
                print(f"GARCH fitting failed: {e}")

        # Historical volatility
        if 'historical' in models:
            self.models['historical'] = returns.rolling(window=20).std()

        # Equal weights by default
        n_models = len(self.models)
        self.weights = {name: 1.0 / n_models for name in self.models.keys()}

        return self

    def predict(self, steps: int = 1, method: str = 'ensemble') -> np.ndarray:
        """
        Forecast volatility.

        Args:
            steps: Number of steps ahead
            method: 'ensemble' or specific model name

        Returns:
            Volatility forecast array
        """
        if method == 'ensemble':
            forecasts = []
            for name, model in self.models.items():
                if name == 'historical':
                    # Use last historical vol as forecast
                    forecast = np.repeat(model.iloc[-1], steps)
                else:
                    forecast = model.predict(steps)
                forecasts.append(self.weights[name] * forecast)

            return np.sum(forecasts, axis=0)

        elif method in self.models:
            if method == 'historical':
                return np.repeat(self.models[method].iloc[-1], steps)
            return self.models[method].predict(steps)

        else:
            raise ValueError(f"Unknown method: {method}")

    def set_weights(self, weights: dict) -> None:
        """
        Set custom model weights.

        Args:
            weights: Dictionary of model weights (must sum to 1)
        """
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
        self.weights = weights


def realized_volatility(
    returns: Union[np.ndarray, pd.Series],
    periods_per_day: int = 1
) -> float:
    """
    Calculate realized volatility (annualized).

    Args:
        returns: Intraday or daily returns
        periods_per_day: Number of periods per day (e.g., 390 for minute data)

    Returns:
        Annualized realized volatility
    """
    returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
    rv = np.sqrt(np.sum(returns**2) * 252 * periods_per_day)
    return rv
