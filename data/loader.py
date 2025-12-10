"""
This module provides functions to load and preprocess data for reinforcement learning tasks.
It includes functionality to read data from various sources, clean it, and prepare it for model training.

dependencies:
- pandas
- numpy
- sklearn
- custom_data_utils
- custom_preprocessing_utils
"""
# Standerd library imports
from typing import Tuple, Dict
from pathlib import Path
from dataclasses import dataclass

# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import arcticdb as adb
from sklearn.model_selection import train_test_split


@dataclass
class DataManger():
    """Class to manage data loading and preprocessing for reinforcement learning tasks."""
    portfolio: list = []
    start_data:str = "2000-01-01"
    end_data:str = "None"
    test_size: float = 0.2
    random_state: int = 42

    @property
    def get_raw_data(self)-> pd.DataFrame:
        """Fetch raw data from yfinance."""
        data = yf.download(
            tickers=self.portfolio,
            start=self.start_data,
            end=self.end_data,
            interval = "1d",
            auto_adjust =   False)

        return data

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into training and testing sets."""
        train_data, test_data = train_test_split(
            data, test_size=self.test_size, random_state=self.random_state
        )
        return train_data, test_data

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load, preprocess, and split the data."""
        raw_data = self.load_data()
        processed_data = self.preprocess_data(raw_data)
        train_data, test_data = self.split_data(processed_data)
        return train_data, test_data
    
    @property
    def pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data loading and preprocessing pipeline."""
        train_data, test_data = self.get_data()
        return train_data, test_data
