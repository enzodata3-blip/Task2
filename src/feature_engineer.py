"""
Feature Engineering Module
Creates interaction terms and polynomial features based on correlation analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Engineer features including interaction terms, polynomial features, and domain-specific transformations
    """

    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Initialize feature engineer

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Name of the target variable
        """
        self.data = data.copy()
        self.target_column = target_column
        self.engineered_data = data.copy()
        self.interaction_features = []
        self.polynomial_features = []
        self.created_features = []

    def create_interaction_terms(self,
                                  interaction_pairs: List[Tuple[str, str]],
                                  interaction_type: str = 'multiply') -> pd.DataFrame:
        """
        Create interaction terms from feature pairs

        Parameters:
        -----------
        interaction_pairs : List[Tuple[str, str]]
            List of feature pairs to create interactions
        interaction_type : str
            Type of interaction ('multiply', 'add', 'divide', 'subtract', 'all')

        Returns:
        --------
        pd.DataFrame
            Data with new interaction features
        """
        print(f"\n{'='*60}")
        print(f"Creating Interaction Terms")
        print(f"{'='*60}")

        for feat1, feat2 in interaction_pairs:
            if feat1 not in self.data.columns or feat2 not in self.data.columns:
                print(f"⚠️  Skipping {feat1} × {feat2}: Feature(s) not found")
                continue

            # Multiplication (most common interaction)
            if interaction_type in ['multiply', 'all']:
                interaction_name = f"{feat1}_X_{feat2}"
                self.engineered_data[interaction_name] = (
                    self.engineered_data[feat1] * self.engineered_data[feat2]
                )
                self.interaction_features.append(interaction_name)
                self.created_features.append(interaction_name)
                print(f"✓ Created: {interaction_name}")

            # Addition (combined effect)
            if interaction_type in ['add', 'all']:
                sum_name = f"{feat1}_PLUS_{feat2}"
                self.engineered_data[sum_name] = (
                    self.engineered_data[feat1] + self.engineered_data[feat2]
                )
                self.interaction_features.append(sum_name)
                self.created_features.append(sum_name)
                print(f"✓ Created: {sum_name}")

            # Division (ratio)
            if interaction_type in ['divide', 'all']:
                # Avoid division by zero
                ratio_name = f"{feat1}_DIV_{feat2}"
                denominator = self.engineered_data[feat2].replace(0, np.nan)
                self.engineered_data[ratio_name] = (
                    self.engineered_data[feat1] / denominator
                )
                # Fill NaN with median
                self.engineered_data[ratio_name].fillna(
                    self.engineered_data[ratio_name].median(),
                    inplace=True
                )
                self.interaction_features.append(ratio_name)
                self.created_features.append(ratio_name)
                print(f"✓ Created: {ratio_name}")

            # Subtraction (difference)
            if interaction_type in ['subtract', 'all']:
                diff_name = f"{feat1}_MINUS_{feat2}"
                self.engineered_data[diff_name] = (
                    self.engineered_data[feat1] - self.engineered_data[feat2]
                )
                self.interaction_features.append(diff_name)
                self.created_features.append(diff_name)
                print(f"✓ Created: {diff_name}")

        print(f"\nTotal interaction features created: {len(self.interaction_features)}")
        return self.engineered_data

    def create_polynomial_features(self,
                                    features: List[str],
                                    degree: int = 2,
                                    include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features (squared, cubed terms)

        Parameters:
        -----------
        features : List[str]
            Features to create polynomial terms from
        degree : int
            Degree of polynomial (2 = squared, 3 = cubed)
        include_bias : bool
            Whether to include bias term

        Returns:
        --------
        pd.DataFrame
            Data with polynomial features
        """
        print(f"\n{'='*60}")
        print(f"Creating Polynomial Features (degree={degree})")
        print(f"{'='*60}")

        for feature in features:
            if feature not in self.data.columns:
                print(f"⚠️  Skipping {feature}: Feature not found")
                continue

            for d in range(2, degree + 1):
                poly_name = f"{feature}_POW{d}"
                self.engineered_data[poly_name] = np.power(
                    self.engineered_data[feature], d
                )
                self.polynomial_features.append(poly_name)
                self.created_features.append(poly_name)
                print(f"✓ Created: {poly_name}")

        print(f"\nTotal polynomial features created: {len(self.polynomial_features)}")
        return self.engineered_data

    def create_domain_specific_features(self,
                                         custom_transformations: Dict[str, callable]) -> pd.DataFrame:
        """
        Create custom domain-specific features

        Parameters:
        -----------
        custom_transformations : Dict[str, callable]
            Dictionary mapping new feature name to transformation function

        Returns:
        --------
        pd.DataFrame
            Data with custom features

        Example:
        --------
        transformations = {
            'BMI_category': lambda df: pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                                               labels=['underweight', 'normal', 'overweight', 'obese'])
        }
        """
        print(f"\n{'='*60}")
        print(f"Creating Domain-Specific Features")
        print(f"{'='*60}")

        for feature_name, transform_func in custom_transformations.items():
            try:
                self.engineered_data[feature_name] = transform_func(self.engineered_data)
                self.created_features.append(feature_name)
                print(f"✓ Created: {feature_name}")
            except Exception as e:
                print(f"⚠️  Failed to create {feature_name}: {str(e)}")

        return self.engineered_data

    def create_log_features(self, features: List[str], handle_negatives: str = 'shift') -> pd.DataFrame:
        """
        Create log-transformed features (useful for skewed distributions)

        Parameters:
        -----------
        features : List[str]
            Features to log-transform
        handle_negatives : str
            How to handle negative values ('shift', 'absolute', 'skip')

        Returns:
        --------
        pd.DataFrame
            Data with log features
        """
        print(f"\n{'='*60}")
        print(f"Creating Log-Transformed Features")
        print(f"{'='*60}")

        for feature in features:
            if feature not in self.data.columns:
                print(f"⚠️  Skipping {feature}: Feature not found")
                continue

            log_name = f"{feature}_LOG"
            values = self.engineered_data[feature].copy()

            # Handle negative or zero values
            if handle_negatives == 'shift':
                min_val = values.min()
                if min_val <= 0:
                    values = values - min_val + 1
            elif handle_negatives == 'absolute':
                values = np.abs(values)
            elif handle_negatives == 'skip':
                if (values <= 0).any():
                    print(f"⚠️  Skipping {feature}: Contains non-positive values")
                    continue

            self.engineered_data[log_name] = np.log1p(values)
            self.created_features.append(log_name)
            print(f"✓ Created: {log_name}")

        return self.engineered_data

    def create_binned_features(self,
                                features: List[str],
                                n_bins: int = 5,
                                strategy: str = 'quantile') -> pd.DataFrame:
        """
        Create binned/discretized versions of continuous features

        Parameters:
        -----------
        features : List[str]
            Features to bin
        n_bins : int
            Number of bins
        strategy : str
            Binning strategy ('quantile', 'uniform', 'kmeans')

        Returns:
        --------
        pd.DataFrame
            Data with binned features
        """
        from sklearn.preprocessing import KBinsDiscretizer

        print(f"\n{'='*60}")
        print(f"Creating Binned Features (n_bins={n_bins}, strategy={strategy})")
        print(f"{'='*60}")

        for feature in features:
            if feature not in self.data.columns:
                print(f"⚠️  Skipping {feature}: Feature not found")
                continue

            binned_name = f"{feature}_BIN{n_bins}"

            try:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode='ordinal',
                    strategy=strategy
                )
                values = self.engineered_data[[feature]].copy()
                self.engineered_data[binned_name] = discretizer.fit_transform(values)
                self.created_features.append(binned_name)
                print(f"✓ Created: {binned_name}")
            except Exception as e:
                print(f"⚠️  Failed to create {binned_name}: {str(e)}")

        return self.engineered_data

    def remove_low_variance_features(self, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove features with low variance (likely not informative)

        Parameters:
        -----------
        threshold : float
            Variance threshold

        Returns:
        --------
        pd.DataFrame
            Data with low variance features removed
        """
        from sklearn.feature_selection import VarianceThreshold

        print(f"\n{'='*60}")
        print(f"Removing Low Variance Features (threshold={threshold})")
        print(f"{'='*60}")

        numeric_features = self.engineered_data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_features:
            numeric_features.remove(self.target_column)

        initial_count = len(numeric_features)

        selector = VarianceThreshold(threshold=threshold)
        data_numeric = self.engineered_data[numeric_features]

        try:
            selector.fit(data_numeric)
            selected_features = [
                feat for feat, selected in zip(numeric_features, selector.get_support())
                if selected
            ]

            removed_features = set(numeric_features) - set(selected_features)

            if removed_features:
                print(f"\nRemoving {len(removed_features)} low-variance features:")
                for feat in removed_features:
                    print(f"  - {feat}")
                self.engineered_data = self.engineered_data.drop(columns=list(removed_features))
            else:
                print("\nNo low-variance features found.")

            print(f"\nFeatures: {initial_count} → {len(selected_features)}")

        except Exception as e:
            print(f"⚠️  Variance filtering failed: {str(e)}")

        return self.engineered_data

    def get_feature_summary(self) -> Dict:
        """
        Get summary of feature engineering

        Returns:
        --------
        Dict
            Summary statistics
        """
        summary = {
            'original_features': len(self.data.columns) - 1,  # Exclude target
            'total_features': len(self.engineered_data.columns) - 1,
            'created_features': len(self.created_features),
            'interaction_features': len(self.interaction_features),
            'polynomial_features': len(self.polynomial_features),
            'feature_list': self.created_features
        }

        print(f"\n{'='*60}")
        print(f"FEATURE ENGINEERING SUMMARY")
        print(f"{'='*60}")
        print(f"Original features:    {summary['original_features']}")
        print(f"Created features:     {summary['created_features']}")
        print(f"  - Interactions:     {summary['interaction_features']}")
        print(f"  - Polynomials:      {summary['polynomial_features']}")
        print(f"Total features:       {summary['total_features']}")
        print(f"{'='*60}\n")

        return summary

    def get_engineered_data(self) -> pd.DataFrame:
        """Return the engineered dataset"""
        return self.engineered_data.copy()


if __name__ == "__main__":
    # Example usage
    print("Feature Engineer Module - Ready for import")
