"""
Data Manager for Credit Market RL Project
==========================================
This module provides comprehensive data management for reinforcement learning tasks
in credit markets, including:
- Raw market data download from yfinance
- Data cleaning and preprocessing
- Technical and credit-specific feature engineering
- Train/validation/test splitting
- ArcticDB time-series database integration

Dependencies:
    - pandas, numpy: Data manipulation
    - yfinance: Market data source
    - arcticdb: Time-series database
    - ta: Technical analysis library
    - yaml: Configuration parsing
    - sklearn: Preprocessing utilities
"""

# Standard library imports
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arcticdb as adb
import numpy as np

# Third-party imports
import pandas as pd
import yaml
import yfinance as yf

# Technical analysis
try:
    import ta
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands
except ImportError:
    warnings.warn("ta library not installed. Technical indicators will be limited.")
    ta = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Comprehensive data manager for credit market reinforcement learning.
    
    This class handles the complete data pipeline from raw market data download
    to processed, feature-engineered datasets ready for RL training.
    
    Attributes:
        config_path: Path to YAML configuration file
        arctic_lib: ArcticDB library instance for time-series storage
        tickers: List of all tickers to download
        train_data: Processed training dataset
        val_data: Processed validation dataset
        test_data: Processed test dataset
    """
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        """
        Initialize DataManager with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Extract data configuration (needed before ArcticDB init)
        self.data_config = self.config['data']
        self.time_config = self.config['time']
        
        # Initialize ArcticDB connection
        self.arctic_lib = None
        if self.data_config['arctic']['enabled']:
            self._init_arcticdb()
        
        # Build ticker list
        self.tickers = self._build_ticker_list()
        
        # Placeholders for processed data
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        logger.info(f"DataManager initialized with {len(self.tickers)} tickers")
    
    def _load_config(self) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _init_arcticdb(self):
        """Initialize ArcticDB connection for time-series storage."""
        try:
            arctic_config = self.data_config['arctic']
            storage_path = arctic_config['storage_path']
            library_name = arctic_config['library']
            
            # Create storage directory if it doesn't exist
            storage_path_obj = Path(storage_path)
            storage_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Initialize Arctic with LMDB backend
            # Use absolute path for LMDB URI
            abs_path = storage_path_obj.resolve()
            arctic = adb.Arctic(f"lmdb://{abs_path}")
            
            # Get or create library
            self.arctic_lib = arctic.get_library(library_name, create_if_missing=True)
            
            logger.info(f"ArcticDB initialized at {abs_path}/{library_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ArcticDB: {e}")
            logger.error("Traceback: ", exc_info=True)
            logger.warning("Continuing without database.")
            self.arctic_lib = None
    
    def _build_ticker_list(self) -> List[str]:
        """Build complete list of tickers from configuration."""
        tickers = []
        
        # Credit instruments
        credit = self.data_config['credit_instruments']
        tickers.extend(credit.get('investment_grade', []))
        tickers.extend(credit.get('high_yield', []))
        
        # Treasury instruments
        treasury = self.data_config['treasury_instruments']
        tickers.extend(treasury.get('short_term', []))
        tickers.extend(treasury.get('intermediate_term', []))
        tickers.extend(treasury.get('long_term', []))
        
        # Equity volatility
        tickers.extend(self.data_config.get('equity_volatility', []))
        
        # Regime indicators
        tickers.extend(self.data_config.get('regime_indicators', []))
        
        # Remove duplicates and filter out synthetic instruments
        tickers = [t for t in list(set(tickers)) if not t.startswith('CDX')]
        
        return tickers
    
    def download_raw_data(self, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         force_download: bool = False) -> pd.DataFrame:
        """
        Download raw market data from yfinance.
        
        Args:
            start_date: Start date (YYYY-MM-DD). If None, uses earliest training date
            end_date: End date (YYYY-MM-DD). If None, uses today
            force_download: If True, skip cache and force fresh download
            
        Returns:
            DataFrame with multi-index columns (ticker, OHLCV)
        """
        # Determine date range
        if start_date is None:
            start_date = self.time_config['train_start_date']
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Downloading data for {len(self.tickers)} tickers from {start_date} to {end_date}")
        
        # Check ArcticDB cache if not forcing download
        if not force_download and self.arctic_lib is not None:
            try:
                cached_data = self._load_from_arctic('raw_data')
                if cached_data is not None:
                    logger.info("Loaded data from ArcticDB cache")
                    self.raw_data = cached_data
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Download data from yfinance
        try:
            data = yf.download(
                tickers=self.tickers,
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=False,
                group_by='ticker',
                threads=True,
                progress=True
            )
            
            if data is None or data.empty:
                raise ValueError("Downloaded data is empty")
            
            logger.info(f"Downloaded {len(data)} rows of data")
            
            # Store in ArcticDB
            if self.arctic_lib is not None:
                self._save_to_arctic(data, 'raw_data')
            
            self.raw_data = data
            return data
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise
    
    def clean_data(self, data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Clean raw data by handling duplicates, missing values, and anomalies.
        
        Args:
            data: Raw market data
            
        Returns:
            Cleaned DataFrame
        """
        if data is None:
            raise ValueError("Cannot clean None data")
        
        logger.info("Cleaning data...")
        
        df = data.copy()
        
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"Removed {len(data) - len(df)} duplicate rows")
        
        # Handle missing values
        initial_missing = df.isna().sum().sum()
        
        # Forward fill missing values (carry last known price forward)
        df = df.ffill()
        
        # Backward fill any remaining NaNs at the start
        df = df.bfill()
        
        # Drop any remaining NaNs (shouldn't be any)
        df = df.dropna()
        
        final_missing = df.isna().sum().sum()
        logger.info(f"Handled {initial_missing} missing values, {final_missing} remaining")
        
        # Remove outliers (prices that jump >50% in a single day - likely data errors)
        if isinstance(df.columns, pd.MultiIndex):
            # Multi-ticker format
            for ticker in df.columns.get_level_values(0).unique():
                if 'Close' in df[ticker].columns:
                    returns = df[ticker]['Close'].pct_change()
                    outliers = (returns.abs() > 0.5)
                    if outliers.any():
                        logger.warning(f"Found {outliers.sum()} outliers in {ticker}, removing...")
                        df.loc[outliers, ticker] = np.nan
        else:
            # Single ticker format
            if 'Close' in df.columns:
                returns = df['Close'].pct_change()
                outliers = (returns.abs() > 0.5)
                if outliers.any():
                    logger.warning(f"Found {outliers.sum()} outliers, removing...")
                    df.loc[outliers] = np.nan
        
        # Re-fill any NaNs created by outlier removal
        df = df.ffill().bfill()
        
        logger.info(f"Cleaned data shape: {df.shape}")
        
        return df
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer technical and credit-specific features.
        
        Args:
            data: Cleaned market data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        df = data.copy()
        
        # Process each ticker
        if isinstance(df.columns, pd.MultiIndex):
            tickers = df.columns.get_level_values(0).unique()
            
            for ticker in tickers:
                ticker_data = df[ticker].copy()
                
                # Skip if Close price not available
                if 'Close' not in ticker_data.columns:
                    continue
                
                close_prices = ticker_data['Close']
                
                # Technical indicators
                df[(ticker, 'returns')] = close_prices.pct_change()
                df[(ticker, 'log_returns')] = np.log(close_prices / close_prices.shift(1))
                df[(ticker, 'volatility')] = df[(ticker, 'returns')].rolling(window=20).std()
                
                # Rolling Sharpe (assuming 252 trading days, 2% risk-free rate)
                excess_returns = df[(ticker, 'returns')] - (0.02 / 252)
                rolling_std = df[(ticker, 'returns')].rolling(window=252).std()
                df[(ticker, 'rolling_sharpe')] = (excess_returns.rolling(window=252).mean() / rolling_std) * np.sqrt(252)
                
                # RSI
                if ta is not None:
                    try:
                        rsi = RSIIndicator(close=close_prices, window=14)
                        df[(ticker, 'rsi')] = rsi.rsi()
                    except Exception:
                        df[(ticker, 'rsi')] = self._calculate_rsi(close_prices, window=14)
                else:
                    df[(ticker, 'rsi')] = self._calculate_rsi(close_prices, window=14)
                
                # MACD
                if ta is not None:
                    try:
                        macd = MACD(close=close_prices)
                        df[(ticker, 'macd')] = macd.macd_diff()
                    except Exception:
                        df[(ticker, 'macd')] = self._calculate_macd(close_prices)
                else:
                    df[(ticker, 'macd')] = self._calculate_macd(close_prices)
                
                # Bollinger Bands
                if ta is not None:
                    try:
                        bb = BollingerBands(close=close_prices, window=20, window_dev=2)
                        df[(ticker, 'bb_upper')] = bb.bollinger_hband()
                        df[(ticker, 'bb_lower')] = bb.bollinger_lband()
                        df[(ticker, 'bb_width')] = bb.bollinger_wband()
                    except Exception:
                        bb_upper, bb_lower, bb_width = self._calculate_bollinger_bands(close_prices)
                        df[(ticker, 'bb_upper')] = bb_upper
                        df[(ticker, 'bb_lower')] = bb_lower
                        df[(ticker, 'bb_width')] = bb_width
                else:
                    bb_upper, bb_lower, bb_width = self._calculate_bollinger_bands(close_prices)
                    df[(ticker, 'bb_upper')] = bb_upper
                    df[(ticker, 'bb_lower')] = bb_lower
                    df[(ticker, 'bb_width')] = bb_width
                
                # Spread Z-Score (price relative to moving average)
                ma_20 = close_prices.rolling(window=20).mean()
                ma_std = close_prices.rolling(window=20).std()
                df[(ticker, 'spread_z_score')] = (close_prices - ma_20) / ma_std
                
                # Credit-specific features (estimated)
                # Duration approximation (inverse of yield proxy)
                if 'High' in ticker_data.columns and 'Low' in ticker_data.columns:
                    # Average price as proxy for bond price
                    avg_price = (ticker_data['High'] + ticker_data['Low']) / 2
                    # Estimate duration (rough approximation)
                    df[(ticker, 'duration_proxy')] = 100 / avg_price  # Simple inverse relationship
                    
                    # Convexity proxy (second derivative)
                    df[(ticker, 'convexity_proxy')] = avg_price.pct_change().diff()
                
                # Carry (approximated as yield over short rate)
                # Use returns as proxy for carry
                df[(ticker, 'carry_proxy')] = df[(ticker, 'returns')].rolling(window=21).mean() * 252
                
                # Momentum indicators
                df[(ticker, 'momentum_20')] = close_prices / close_prices.shift(20) - 1
                df[(ticker, 'momentum_60')] = close_prices / close_prices.shift(60) - 1
                
                # Percentile rank
                df[(ticker, 'price_percentile')] = close_prices.rolling(window=252).apply(
                    lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
                )
        
        # Cross-asset features (correlations between instruments)
        df = self._add_cross_asset_features(df)
        
        # Macro regime features
        df = self._add_regime_features(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD manually."""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_diff = macd_line - signal_line
        return macd_diff
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2):
        """Calculate Bollinger Bands manually."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        width = (upper_band - lower_band) / sma
        return upper_band, lower_band, width
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset correlation and relationship features."""
        try:
            # Calculate rolling correlations between key instruments
            window = self.time_config.get('correlation_window', 126)
            
            # Define key pairs for correlation
            pairs = [
                ('LQD', 'SPY', 'equity_credit_corr'),  # IG Credit vs Equity
                ('HYG', 'SPY', 'hy_equity_corr'),       # HY Credit vs Equity
                ('LQD', 'IEF', 'credit_treasury_corr'), # Credit vs Treasury
                ('^VIX', 'HYG', 'vix_hy_corr'),         # VIX vs HY
            ]
            
            for ticker1, ticker2, feature_name in pairs:
                try:
                    if (ticker1, 'Close') in df.columns and (ticker2, 'Close') in df.columns:
                        returns1 = df[(ticker1, 'Close')].pct_change()
                        returns2 = df[(ticker2, 'Close')].pct_change()
                        corr = returns1.rolling(window=window).corr(returns2)
                        df[('CrossAsset', feature_name)] = corr
                except Exception:
                    pass
            
            # Spread ratios
            if ('HYG', 'Close') in df.columns and ('LQD', 'Close') in df.columns:
                df[('CrossAsset', 'hy_ig_ratio')] = df[('HYG', 'Close')] / df[('LQD', 'Close')]
            
            if ('LQD', 'Close') in df.columns and ('IEF', 'Close') in df.columns:
                df[('CrossAsset', 'credit_treasury_ratio')] = df[('LQD', 'Close')] / df[('IEF', 'Close')]
                
        except Exception as e:
            logger.warning(f"Error adding cross-asset features: {e}")
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macro regime indicators."""
        try:
            # VIX percentile
            if ('^VIX', 'Close') in df.columns:
                vix = df[('^VIX', 'Close')]
                df[('Regime', 'vix_percentile')] = vix.rolling(window=252).apply(
                    lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
                )
            
            # Yield curve slope (10Y - 2Y proxy)
            if ('^TNX', 'Close') in df.columns and ('^IRX', 'Close') in df.columns:
                df[('Regime', 'yield_curve_slope')] = df[('^TNX', 'Close')] - df[('^IRX', 'Close')]
            
            # Credit cycle phase (simplified based on spread trends)
            if ('HYG', 'spread_z_score') in df.columns:
                spread_z = df[('HYG', 'spread_z_score')]
                df[('Regime', 'credit_cycle_phase')] = pd.cut(
                    spread_z, bins=[-np.inf, -1, 0, 1, np.inf], labels=[0, 1, 2, 3]
                ).astype(float)
                
        except Exception as e:
            logger.warning(f"Error adding regime features: {e}")
        
        return df
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets based on time periods.
        
        Args:
            data: Processed data with features
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("Splitting data into train/val/test sets...")
        
        # Get date ranges from config
        train_start = pd.to_datetime(self.time_config['train_start_date'])
        train_end = pd.to_datetime(self.time_config['train_end_date'])
        val_start = pd.to_datetime(self.time_config['val_start_date'])
        val_end = pd.to_datetime(self.time_config['val_end_date'])
        test_start = pd.to_datetime(self.time_config['test_start_date'])
        test_end = self.time_config['test_end_date']
        
        if test_end is None:
            test_end = data.index.max()
        else:
            test_end = pd.to_datetime(test_end)
        
        # Split data
        train_data = data.loc[train_start:train_end]
        val_data = data.loc[val_start:val_end]
        test_data = data.loc[test_start:test_end]
        
        logger.info(f"Train set: {len(train_data)} rows ({train_start.date()} to {train_end.date()})")
        logger.info(f"Val set: {len(val_data)} rows ({val_start.date()} to {val_end.date()})")
        logger.info(f"Test set: {len(test_data)} rows ({test_start.date()} to {test_end.date()})")
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        return train_data, val_data, test_data
    
    def _save_to_arctic(self, data: pd.DataFrame, symbol: str):
        """Save data to ArcticDB, flattening MultiIndex columns if necessary."""
        if self.arctic_lib is None:
            return
        
        try:
            # Flatten MultiIndex columns if present (ArcticDB doesn't support them)
            data_to_save = data.copy()
            if isinstance(data_to_save.columns, pd.MultiIndex):
                data_to_save.columns = ['_'.join(map(str, col)).strip() for col in data_to_save.columns.values]
            
            self.arctic_lib.write(symbol, data_to_save)
            logger.info(f"Saved {symbol} to ArcticDB with shape {data_to_save.shape}")
        except Exception as e:
            logger.error(f"Failed to save to ArcticDB: {e}")
    
    def _load_from_arctic(self, symbol: str, restore_multiindex: bool = True) -> Optional[pd.DataFrame]:
        """Load data from ArcticDB, optionally restoring MultiIndex columns."""
        if self.arctic_lib is None:
            return None
        
        try:
            result = self.arctic_lib.read(symbol)
            data = result.data
            
            # Ensure we return a pandas DataFrame
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)  # type: ignore
            
            logger.info(f"Loaded {symbol} from ArcticDB with shape {data.shape}")
            
            # Restore MultiIndex columns if they were flattened
            if restore_multiindex and '_' in str(data.columns[0]):
                # Try to split flattened column names back into MultiIndex
                try:
                    new_columns = [tuple(col.split('_', 1)) for col in data.columns]
                    data.columns = pd.MultiIndex.from_tuples(new_columns)
                except Exception:
                    # If restoration fails, just return with flat columns
                    pass
            
            return data
        except Exception as e:
            logger.debug(f"Symbol {symbol} not found in ArcticDB: {e}")
            return None
    
    def save_processed_data(self):
        """Save all processed datasets to ArcticDB."""
        if self.arctic_lib is None:
            logger.warning("ArcticDB not initialized. Cannot save processed data.")
            return
        
        datasets = {
            'processed_data': self.processed_data,
            'train_data': self.train_data,
            'val_data': self.val_data,
            'test_data': self.test_data
        }
        
        for name, data in datasets.items():
            if data is not None:
                self._save_to_arctic(data, name)
    
    def load_processed_data(self) -> bool:
        """
        Load processed data from ArcticDB.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if self.arctic_lib is None:
            return False
        
        try:
            self.processed_data = self._load_from_arctic('processed_data')
            self.train_data = self._load_from_arctic('train_data')
            self.val_data = self._load_from_arctic('val_data')
            self.test_data = self._load_from_arctic('test_data')
            
            if all([
                self.processed_data is not None, 
                self.train_data is not None,
                self.val_data is not None, 
                self.test_data is not None
            ]):
                logger.info("Successfully loaded all processed data from ArcticDB")
                return True
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
        
        return False
    
    def run_pipeline(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run complete data pipeline: download -> clean -> engineer -> split -> save.
        
        Args:
            force_download: If True, skip cache and download fresh data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("=" * 80)
        logger.info("Starting complete data pipeline")
        logger.info("=" * 80)
        
        # Try to load processed data from cache
        if not force_download and self.load_processed_data():
            logger.info("Using cached processed data")
            # Type assertion: load_processed_data only returns True if all data is loaded
            assert self.train_data is not None and self.val_data is not None and self.test_data is not None
            return self.train_data, self.val_data, self.test_data
        
        # Step 1: Download raw data
        raw_data = self.download_raw_data(force_download=force_download)
        
        # Step 2: Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Step 3: Engineer features
        featured_data = self.engineer_features(cleaned_data)
        self.processed_data = featured_data
        
        # Step 4: Split data
        train_data, val_data, test_data = self.split_data(featured_data)
        
        # Step 5: Save to ArcticDB
        self.save_processed_data()
        
        logger.info("=" * 80)
        logger.info("Data pipeline complete!")
        logger.info("=" * 80)
        
        return train_data, val_data, test_data


# Convenience functions for backwards compatibility
def get_data_manager(config_path: str = "configs/base.yaml") -> DataManager:
    """
    Factory function to create DataManager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DataManager instance
    """
    return DataManager(config_path=config_path)


if __name__ == "__main__":
    # Example usage
    print("Initializing DataManager...")
    dm = DataManager(config_path="configs/base.yaml")
    
    print("\nRunning data pipeline...")
    train, val, test = dm.run_pipeline(force_download=False)
    
    print("\nPipeline complete!")
    print(f"Train shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
