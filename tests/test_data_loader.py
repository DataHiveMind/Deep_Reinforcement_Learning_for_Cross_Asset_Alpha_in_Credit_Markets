"""
Comprehensive test suite for DataManager class.

Tests cover:
- Configuration loading
- ArcticDB initialization
- Data downloading from yfinance
- Data cleaning operations
- Feature engineering
- Data splitting
- ArcticDB persistence
"""

# Add parent directory to path for imports
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DataManager


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary directory with test configuration."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create minimal test configuration
    test_config = {
        'data': {
            'source': {
                'provider': 'yfinance',
                'cache_dir': str(tmp_path / 'data' / 'raw'),
                'use_cache': False
            },
            'arctic': {
                'enabled': True,
                'library': 'test_credit_data',
                'storage_backend': 'lmdb',
                'storage_path': str(tmp_path / 'data' / 'arctic_db'),
                'compression': 'lz4'
            },
            'credit_instruments': {
                'investment_grade': ['LQD', 'VCIT'],
                'high_yield': ['HYG', 'JNK']
            },
            'treasury_instruments': {
                'short_term': ['SHY'],
                'intermediate': ['IEF'],
                'long_term': ['TLT']
            },
            'equity_volatility': ['SPY', 'VXX'],
            'regime_indicators': ['^VIX', '^IRX', '^TNX'],
            'features': {
                'technical_indicators': True,
                'credit_specific': True,
                'cross_asset': True,
                'macro_regimes': True
            }
        },
        'time': {
            'train': {
                'start': '2020-01-01',
                'end': '2021-12-31'
            },
            'val': {
                'start': '2022-01-01',
                'end': '2022-12-31'
            },
            'test': {
                'start': '2023-01-01',
                'end': '2023-12-31'
            },
            'lookback_window': 252,
            'rebalance_frequency': 21
        }
    }
    
    config_path = config_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    return config_path


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['LQD', 'HYG', 'SPY', 'TLT', '^VIX']
    
    # Generate realistic price series
    np.random.seed(42)
    data = {}
    
    for ticker in tickers:
        # Create semi-realistic price series
        base_price = 100 if not ticker.startswith('^') else 20
        returns = np.random.randn(len(dates)) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))
        
        data[ticker] = {
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Adj Close': prices
        }
    
    # Create MultiIndex DataFrame
    dfs = []
    for ticker, ticker_data in data.items():
        df = pd.DataFrame(ticker_data, index=dates)
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        dfs.append(df)
    
    result = pd.concat(dfs, axis=1)
    result.index.name = 'Date'
    
    return result


class TestDataManagerInit:
    """Test DataManager initialization and configuration."""
    
    def test_init_with_valid_config(self, temp_config_dir):
        """Test initialization with valid configuration file."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        assert dm.config is not None
        assert 'data' in dm.config
        assert 'time' in dm.config
        assert dm.tickers is not None
        assert len(dm.tickers) > 0
    
    def test_config_loading(self, temp_config_dir):
        """Test configuration file loading."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        assert dm.config['data']['arctic']['enabled'] is True
        assert dm.config['data']['arctic']['library'] == 'test_credit_data'
        assert 'credit_instruments' in dm.config['data']
    
    def test_ticker_list_building(self, temp_config_dir):
        """Test building ticker list from config."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        # Should include all tickers from config
        assert 'LQD' in dm.tickers
        assert 'HYG' in dm.tickers
        assert 'SPY' in dm.tickers
        assert '^VIX' in dm.tickers
        
        # Should not have duplicates
        assert len(dm.tickers) == len(set(dm.tickers))
    
    def test_arcticdb_initialization(self, temp_config_dir):
        """Test ArcticDB initialization."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        # Should initialize if enabled in config
        if dm.config['data']['arctic']['enabled']:
            assert dm.arctic_lib is not None
    
    def test_init_without_arctic(self, temp_config_dir, tmp_path):
        """Test initialization with ArcticDB disabled."""
        # Modify config to disable arctic
        with open(temp_config_dir, 'r') as f:
            config = yaml.safe_load(f)
        
        config['data']['arctic']['enabled'] = False
        
        new_config_path = tmp_path / "config_no_arctic.yaml"
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f)
        
        dm = DataManager(config_path=str(new_config_path))
        assert dm.arctic_lib is None


class TestDataDownload:
    """Test data downloading functionality."""
    
    def test_download_with_mock(self, temp_config_dir, sample_price_data):
        """Test data download with mocked yfinance."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = sample_price_data
            
            data = dm.download_raw_data(
                start_date='2020-01-01',
                end_date='2023-12-31'
            )
            
            assert data is not None
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            mock_download.assert_called_once()
    
    def test_download_validation(self, temp_config_dir):
        """Test download data validation."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        with patch('yfinance.download') as mock_download:
            # Test empty DataFrame handling
            mock_download.return_value = pd.DataFrame()
            
            with pytest.raises(ValueError, match="Failed to download data"):
                dm.download_raw_data(
                    start_date='2020-01-01',
                    end_date='2023-12-31'
                )
    
    def test_download_date_handling(self, temp_config_dir, sample_price_data):
        """Test download with various date formats."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = sample_price_data
            
            # Test with string dates
            data = dm.download_raw_data(
                start_date='2020-01-01',
                end_date='2023-12-31'
            )
            assert data is not None


class TestDataCleaning:
    """Test data cleaning operations."""
    
    def test_clean_data_basic(self, temp_config_dir, sample_price_data):
        """Test basic data cleaning."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        
        assert cleaned is not None
        assert isinstance(cleaned, pd.DataFrame)
        assert not cleaned.empty
    
    def test_clean_data_removes_duplicates(self, temp_config_dir, sample_price_data):
        """Test duplicate removal."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        # Add duplicate rows
        data_with_dupes = pd.concat([sample_price_data, sample_price_data.iloc[:10]])
        original_len = len(sample_price_data)
        dupes_len = len(data_with_dupes)
        
        assert dupes_len > original_len  # Confirm duplicates added
        
        cleaned = dm.clean_data(data_with_dupes)
        
        # Should remove duplicates
        assert len(cleaned) == original_len
    
    def test_clean_data_handles_nans(self, temp_config_dir, sample_price_data):
        """Test NaN handling."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        # Introduce NaN values
        data_with_nans = sample_price_data.copy()
        data_with_nans.iloc[10:15, 0] = np.nan
        
        cleaned = dm.clean_data(data_with_nans)
        
        # Should fill or handle NaNs
        assert cleaned is not None
        # After ffill/bfill, most NaNs should be filled
        nan_ratio = cleaned.isna().sum().sum() / (cleaned.shape[0] * cleaned.shape[1])
        assert nan_ratio < 0.01  # Less than 1% NaNs remaining
    
    def test_clean_data_outlier_detection(self, temp_config_dir, sample_price_data):
        """Test outlier detection and handling."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        # Introduce outliers (>50% single-day move)
        data_with_outliers = sample_price_data.copy()
        col = data_with_outliers.columns[0]
        data_with_outliers.iloc[100, data_with_outliers.columns.get_loc(col)] *= 2.0
        
        cleaned = dm.clean_data(data_with_outliers)
        
        assert cleaned is not None


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_engineer_features_basic(self, temp_config_dir, sample_price_data):
        """Test basic feature engineering."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        assert features is not None
        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] > sample_price_data.shape[1]  # More columns after features
    
    def test_engineer_features_creates_returns(self, temp_config_dir, sample_price_data):
        """Test returns calculation."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        # Check for returns columns
        ticker = dm.tickers[0]
        assert (ticker, 'returns') in features.columns
        assert (ticker, 'log_returns') in features.columns
    
    def test_engineer_features_creates_volatility(self, temp_config_dir, sample_price_data):
        """Test volatility calculation."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        # Check for volatility columns
        ticker = dm.tickers[0]
        assert (ticker, 'volatility') in features.columns
    
    def test_engineer_features_technical_indicators(self, temp_config_dir, sample_price_data):
        """Test technical indicators creation."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        # Check for technical indicators
        ticker = dm.tickers[0]
        assert (ticker, 'rsi') in features.columns
        assert (ticker, 'macd') in features.columns
    
    def test_engineer_features_cross_asset(self, temp_config_dir, sample_price_data):
        """Test cross-asset features."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        # Check for cross-asset features
        assert ('CrossAsset', 'hy_ig_ratio') in features.columns


class TestDataSplitting:
    """Test data splitting functionality."""
    
    def test_split_data_basic(self, temp_config_dir, sample_price_data):
        """Test basic data splitting."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        train, val, test = dm.split_data(features)
        
        assert train is not None
        assert val is not None
        assert test is not None
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
    
    def test_split_data_no_overlap(self, temp_config_dir, sample_price_data):
        """Test that splits don't overlap."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        train, val, test = dm.split_data(features)
        
        # Check no temporal overlap
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()
    
    def test_split_data_preserves_features(self, temp_config_dir, sample_price_data):
        """Test that splits preserve all features."""
        dm = DataManager(config_path=str(temp_config_dir))
        dm.raw_data = sample_price_data.copy()
        
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        
        train, val, test = dm.split_data(features)
        
        # All splits should have same columns
        assert list(train.columns) == list(features.columns)
        assert list(val.columns) == list(features.columns)
        assert list(test.columns) == list(features.columns)


class TestArcticDBPersistence:
    """Test ArcticDB persistence operations."""
    
    def test_save_to_arctic(self, temp_config_dir, sample_price_data):
        """Test saving data to ArcticDB."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        if dm.arctic_lib is None:
            pytest.skip("ArcticDB not initialized")
        
        # Save sample data
        dm._save_to_arctic(sample_price_data, 'test_symbol')
        
        # Verify saved
        assert dm.arctic_lib.has_symbol('test_symbol')
    
    def test_load_from_arctic(self, temp_config_dir, sample_price_data):
        """Test loading data from ArcticDB."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        if dm.arctic_lib is None:
            pytest.skip("ArcticDB not initialized")
        
        # Save then load
        dm._save_to_arctic(sample_price_data, 'test_load')
        loaded = dm._load_from_arctic('test_load')
        
        assert loaded is not None
        assert isinstance(loaded, pd.DataFrame)
        pd.testing.assert_frame_equal(sample_price_data, loaded)
    
    def test_save_processed_data(self, temp_config_dir, sample_price_data):
        """Test saving processed splits to ArcticDB."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        if dm.arctic_lib is None:
            pytest.skip("ArcticDB not initialized")
        
        dm.raw_data = sample_price_data.copy()
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        train, val, test = dm.split_data(features)
        
        dm.train_data = train
        dm.val_data = val
        dm.test_data = test
        
        # Save all splits
        dm.save_processed_data()
        
        # Verify all saved
        assert dm.arctic_lib.has_symbol('train_data')
        assert dm.arctic_lib.has_symbol('val_data')
        assert dm.arctic_lib.has_symbol('test_data')
    
    def test_load_processed_data(self, temp_config_dir, sample_price_data):
        """Test loading processed data from ArcticDB."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        if dm.arctic_lib is None:
            pytest.skip("ArcticDB not initialized")
        
        # Prepare and save data
        dm.raw_data = sample_price_data.copy()
        cleaned = dm.clean_data(dm.raw_data)
        features = dm.engineer_features(cleaned)
        train, val, test = dm.split_data(features)
        
        dm.train_data = train
        dm.val_data = val
        dm.test_data = test
        dm.save_processed_data()
        
        # Create new instance and load
        dm2 = DataManager(config_path=str(temp_config_dir))
        success = dm2.load_processed_data()
        
        assert success is True
        assert dm2.train_data is not None
        assert dm2.val_data is not None
        assert dm2.test_data is not None
        
        pd.testing.assert_frame_equal(train, dm2.train_data)
        pd.testing.assert_frame_equal(val, dm2.val_data)
        pd.testing.assert_frame_equal(test, dm2.test_data)


class TestPipeline:
    """Test end-to-end pipeline functionality."""
    
    def test_run_pipeline_basic(self, temp_config_dir, sample_price_data):
        """Test basic pipeline execution."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = sample_price_data
            
            train, val, test = dm.run_pipeline(force_download=True)
            
            assert train is not None
            assert val is not None
            assert test is not None
    
    def test_run_pipeline_caching(self, temp_config_dir, sample_price_data):
        """Test pipeline caching functionality."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        if dm.arctic_lib is None:
            pytest.skip("ArcticDB not initialized")
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = sample_price_data
            
            # First run - should download
            train1, val1, test1 = dm.run_pipeline(force_download=True)
            call_count_1 = mock_download.call_count
            
            # Second run - should use cache
            train2, val2, test2 = dm.run_pipeline(force_download=False)
            call_count_2 = mock_download.call_count
            
            # Should not download again
            assert call_count_2 == call_count_1
            
            # Data should be identical
            pd.testing.assert_frame_equal(train1, train2)
    
    def test_run_pipeline_force_refresh(self, temp_config_dir, sample_price_data):
        """Test pipeline force refresh."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = sample_price_data
            
            # Run with force_download=True
            train, val, test = dm.run_pipeline(force_download=True)
            
            assert train is not None
            mock_download.assert_called()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            _dm = DataManager(config_path="nonexistent_config.yaml")
    
    def test_empty_dataframe_handling(self, temp_config_dir):
        """Test handling of empty DataFrame."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        empty_df = pd.DataFrame()
        
        # Should raise error for empty data
        with pytest.raises(ValueError):
            dm.clean_data(empty_df)
    
    def test_none_dataframe_handling(self, temp_config_dir):
        """Test handling of None DataFrame."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        # Should raise error for None data
        with pytest.raises(ValueError):
            dm.clean_data(None)
    
    def test_insufficient_data_for_features(self, temp_config_dir):
        """Test handling of insufficient data for feature calculation."""
        dm = DataManager(config_path=str(temp_config_dir))
        
        # Create very small dataset (< lookback window)
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        ticker = 'TEST'
        
        small_data = pd.DataFrame({
            (ticker, 'Close'): np.random.randn(10) * 10 + 100,
            (ticker, 'Volume'): np.random.randint(1000, 10000, 10)
        }, index=dates)
        
        cleaned = dm.clean_data(small_data)
        
        # Should still work but with NaNs for rolling features
        features = dm.engineer_features(cleaned)
        assert features is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
