"""
Data Loader Module
Handles data ingestion, validation, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Load and preprocess data from various sources
    """

    def __init__(self):
        """Initialize data loader"""
        self.data = None
        self.data_info = {}

    def load_csv(self,
                 file_path: str,
                 target_column: str,
                 **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file

        Parameters:
        -----------
        file_path : str
            Path to CSV file
        target_column : str
            Name of target variable
        **kwargs
            Additional parameters for pd.read_csv

        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        print(f"\n{'='*60}")
        print(f"Loading Data from CSV")
        print(f"{'='*60}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.data = pd.read_csv(file_path, **kwargs)

        print(f"✓ Data loaded successfully")
        print(f"  Rows: {self.data.shape[0]}")
        print(f"  Columns: {self.data.shape[1]}")
        print(f"  Target: {target_column}")

        # Validate target column
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        self._analyze_data(target_column)

        return self.data

    def load_sample_dataset(self,
                             dataset_name: str = 'diabetes',
                             n_samples: int = 1000) -> Tuple[pd.DataFrame, str]:
        """
        Load sample dataset for demonstration

        Parameters:
        -----------
        dataset_name : str
            Name of sample dataset ('diabetes', 'breast_cancer', 'california_housing')
        n_samples : int
            Number of samples to generate (for synthetic data)

        Returns:
        --------
        Tuple[pd.DataFrame, str]
            (data, target_column_name)
        """
        from sklearn.datasets import (
            load_diabetes,
            load_breast_cancer,
            fetch_california_housing,
            make_classification,
            make_regression
        )

        print(f"\n{'='*60}")
        print(f"Loading Sample Dataset: {dataset_name}")
        print(f"{'='*60}")

        if dataset_name == 'diabetes':
            data_sklearn = load_diabetes()
            self.data = pd.DataFrame(data_sklearn.data, columns=data_sklearn.feature_names)
            self.data['target'] = data_sklearn.target
            target_column = 'target'

        elif dataset_name == 'breast_cancer':
            data_sklearn = load_breast_cancer()
            self.data = pd.DataFrame(data_sklearn.data, columns=data_sklearn.feature_names)
            self.data['target'] = data_sklearn.target
            target_column = 'target'

        elif dataset_name == 'california_housing':
            data_sklearn = fetch_california_housing()
            self.data = pd.DataFrame(data_sklearn.data, columns=data_sklearn.feature_names)
            self.data['target'] = data_sklearn.target
            target_column = 'target'

        elif dataset_name == 'synthetic_classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                random_state=42
            )
            self.data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            self.data['target'] = y
            target_column = 'target'

        elif dataset_name == 'synthetic_regression':
            X, y = make_regression(
                n_samples=n_samples,
                n_features=20,
                n_informative=15,
                random_state=42
            )
            self.data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            self.data['target'] = y
            target_column = 'target'

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"✓ Dataset loaded: {dataset_name}")
        print(f"  Rows: {self.data.shape[0]}")
        print(f"  Columns: {self.data.shape[1]}")

        self._analyze_data(target_column)

        return self.data, target_column

    def _analyze_data(self, target_column: str):
        """Analyze loaded data"""
        print(f"\n{'Data Analysis':-^60}")

        # Basic info
        print(f"\nData Types:")
        dtype_counts = self.data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")

        # Missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(self.data)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print(f"\n✓ No missing values")

        # Target distribution
        print(f"\nTarget Variable: {target_column}")
        if self.data[target_column].dtype in ['object', 'category', 'bool'] or \
           self.data[target_column].nunique() < 20:
            print("  Type: Categorical (Classification)")
            print(f"  Classes: {self.data[target_column].nunique()}")
            print(f"\n  Distribution:")
            for val, count in self.data[target_column].value_counts().items():
                pct = (count / len(self.data)) * 100
                print(f"    {val}: {count} ({pct:.1f}%)")
        else:
            print("  Type: Continuous (Regression)")
            print(f"  Mean: {self.data[target_column].mean():.2f}")
            print(f"  Std:  {self.data[target_column].std():.2f}")
            print(f"  Min:  {self.data[target_column].min():.2f}")
            print(f"  Max:  {self.data[target_column].max():.2f}")

        self.data_info = {
            'n_rows': self.data.shape[0],
            'n_cols': self.data.shape[1],
            'target': target_column,
            'missing_values': missing.sum(),
            'numeric_features': len(self.data.select_dtypes(include=[np.number]).columns)
        }

    def handle_missing_values(self,
                               strategy: str = 'median',
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values

        Parameters:
        -----------
        strategy : str
            Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns : Optional[List[str]]
            Specific columns to impute (None = all columns)

        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        if self.data is None:
            raise ValueError("No data loaded")

        print(f"\n{'='*60}")
        print(f"Handling Missing Values (strategy: {strategy})")
        print(f"{'='*60}")

        if columns is None:
            columns = self.data.columns[self.data.isnull().any()].tolist()

        if not columns:
            print("✓ No missing values to handle")
            return self.data

        for col in columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count == 0:
                continue

            if strategy == 'mean':
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif strategy == 'median':
                self.data[col].fillna(self.data[col].median(), inplace=True)
            elif strategy == 'mode':
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                self.data.dropna(subset=[col], inplace=True)

            print(f"✓ Imputed {missing_count} values in '{col}'")

        return self.data

    def encode_categorical(self,
                            columns: Optional[List[str]] = None,
                            method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables

        Parameters:
        -----------
        columns : Optional[List[str]]
            Columns to encode (None = all categorical)
        method : str
            Encoding method ('onehot', 'label')

        Returns:
        --------
        pd.DataFrame
            Data with encoded categorical variables
        """
        if self.data is None:
            raise ValueError("No data loaded")

        print(f"\n{'='*60}")
        print(f"Encoding Categorical Variables (method: {method})")
        print(f"{'='*60}")

        if columns is None:
            columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not columns:
            print("✓ No categorical variables to encode")
            return self.data

        for col in columns:
            if method == 'onehot':
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                self.data = pd.concat([self.data, dummies], axis=1)
                self.data.drop(columns=[col], inplace=True)
                print(f"✓ One-hot encoded '{col}' ({len(dummies.columns)} new columns)")

            elif method == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                print(f"✓ Label encoded '{col}'")

        return self.data

    def get_data(self) -> pd.DataFrame:
        """Return the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.copy()

    def get_info(self) -> Dict:
        """Return data information"""
        return self.data_info.copy()


if __name__ == "__main__":
    print("Data Loader Module - Ready for import")
