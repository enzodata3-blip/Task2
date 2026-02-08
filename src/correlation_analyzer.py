"""
Correlation Analyzer Module
Analyzes feature correlations and suggests interaction terms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """
    Analyze correlations between features and identify potential interaction terms
    """

    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Initialize the correlation analyzer

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Name of the target variable
        """
        self.data = data.copy()
        self.target_column = target_column
        self.numeric_features = self._get_numeric_features()
        self.correlation_matrix = None
        self.target_correlations = None

    def _get_numeric_features(self) -> List[str]:
        """Extract numeric feature names (excluding target)"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        return numeric_cols

    def compute_correlations(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix

        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman', or 'kendall')

        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        print(f"\n{'='*60}")
        print(f"Computing {method.upper()} correlation matrix...")
        print(f"{'='*60}")

        data_numeric = self.data[self.numeric_features + [self.target_column]]
        self.correlation_matrix = data_numeric.corr(method=method)

        # Compute correlations with target
        self.target_correlations = self.correlation_matrix[self.target_column].sort_values(
            ascending=False, key=abs
        ).drop(self.target_column)

        print(f"\nTop 10 features correlated with {self.target_column}:")
        print("-" * 60)
        for feature, corr in self.target_correlations.head(10).items():
            print(f"  {feature:40s}: {corr:+.4f}")

        return self.correlation_matrix

    def identify_multicollinearity(self, threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated feature pairs (potential multicollinearity)

        Parameters:
        -----------
        threshold : float
            Correlation threshold for flagging multicollinearity

        Returns:
        --------
        List[Tuple[str, str, float]]
            List of (feature1, feature2, correlation) tuples
        """
        if self.correlation_matrix is None:
            self.compute_correlations()

        print(f"\n{'='*60}")
        print(f"Identifying Multicollinearity (threshold: {threshold})")
        print(f"{'='*60}")

        high_corr_pairs = []

        # Get upper triangle of correlation matrix
        corr_matrix = self.correlation_matrix.loc[self.numeric_features, self.numeric_features]
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)

        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if upper_triangle[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr_pairs:
            print(f"\nFound {len(high_corr_pairs)} highly correlated pairs:")
            print("-" * 60)
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feat1:30s} <-> {feat2:30s}: {corr:+.4f}")
        else:
            print("\nNo multicollinearity issues detected.")

        return high_corr_pairs

    def suggest_interaction_terms(self,
                                   min_correlation: float = 0.15,
                                   max_correlation: float = 0.85,
                                   top_n: int = 20) -> List[Tuple[str, str, float]]:
        """
        Suggest interaction terms based on correlation analysis

        Strategy:
        - Features moderately correlated with target (not too weak, not too strong)
        - Features with complementary information (not highly correlated with each other)

        Parameters:
        -----------
        min_correlation : float
            Minimum absolute correlation with target to consider
        max_correlation : float
            Maximum correlation between features to avoid redundancy
        top_n : int
            Number of interaction terms to suggest

        Returns:
        --------
        List[Tuple[str, str, float]]
            List of (feature1, feature2, combined_score) tuples
        """
        if self.correlation_matrix is None:
            self.compute_correlations()

        print(f"\n{'='*60}")
        print(f"Suggesting Interaction Terms")
        print(f"{'='*60}")
        print(f"Criteria:")
        print(f"  - Each feature abs(corr) with target >= {min_correlation}")
        print(f"  - Feature pair abs(corr) with each other <= {max_correlation}")

        # Filter features based on target correlation
        candidate_features = [
            feat for feat, corr in self.target_correlations.items()
            if abs(corr) >= min_correlation
        ]

        print(f"\nCandidate features (n={len(candidate_features)}): {', '.join(candidate_features[:10])}...")

        interaction_scores = []

        # Generate all pairs of candidate features
        for i, feat1 in enumerate(candidate_features):
            for feat2 in candidate_features[i+1:]:
                # Check correlation between features
                feat_corr = abs(self.correlation_matrix.loc[feat1, feat2])

                if feat_corr <= max_correlation:
                    # Score based on:
                    # 1. Individual correlations with target
                    # 2. Low correlation between features (complementary info)
                    target_corr1 = abs(self.target_correlations[feat1])
                    target_corr2 = abs(self.target_correlations[feat2])

                    # Combined score: sum of target correlations, penalized by feature correlation
                    score = (target_corr1 + target_corr2) * (1 - feat_corr/max_correlation)

                    interaction_scores.append((feat1, feat2, score))

        # Sort by score and take top N
        interaction_scores.sort(key=lambda x: x[2], reverse=True)
        top_interactions = interaction_scores[:top_n]

        print(f"\nTop {min(top_n, len(top_interactions))} Suggested Interaction Terms:")
        print("-" * 60)
        for i, (feat1, feat2, score) in enumerate(top_interactions, 1):
            corr1 = self.target_correlations[feat1]
            corr2 = self.target_correlations[feat2]
            feat_corr = self.correlation_matrix.loc[feat1, feat2]
            print(f"{i:2d}. {feat1} × {feat2}")
            print(f"     Target corr: {feat1}={corr1:+.4f}, {feat2}={corr2:+.4f}")
            print(f"     Feature corr: {feat_corr:+.4f}, Score: {score:.4f}\n")

        return top_interactions

    def plot_correlation_heatmap(self,
                                  figsize: Tuple[int, int] = (14, 12),
                                  save_path: Optional[str] = None):
        """
        Plot correlation heatmap

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save the plot
        """
        if self.correlation_matrix is None:
            self.compute_correlations()

        plt.figure(figsize=figsize)

        # Use only numeric features for cleaner visualization
        corr_subset = self.correlation_matrix.loc[
            self.numeric_features + [self.target_column],
            self.numeric_features + [self.target_column]
        ]

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_subset, dtype=bool), k=1)

        sns.heatmap(
            corr_subset,
            mask=mask,
            annot=True if len(corr_subset) <= 15 else False,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'}
        )

        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nCorrelation heatmap saved to: {save_path}")

        plt.show()

    def plot_target_correlations(self,
                                  top_n: int = 15,
                                  figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None):
        """
        Plot bar chart of feature correlations with target

        Parameters:
        -----------
        top_n : int
            Number of top features to display
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save the plot
        """
        if self.target_correlations is None:
            self.compute_correlations()

        plt.figure(figsize=figsize)

        # Get top N by absolute correlation
        top_corr = self.target_correlations.head(top_n)

        # Create color map (positive = blue, negative = red)
        colors = ['#2E86AB' if x > 0 else '#A23B72' for x in top_corr.values]

        plt.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.8)
        plt.yticks(range(len(top_corr)), top_corr.index)
        plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Features Correlated with {self.target_column}',
                  fontsize=14, fontweight='bold', pad=15)
        plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        plt.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, v in enumerate(top_corr.values):
            plt.text(v, i, f' {v:.3f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nTarget correlation plot saved to: {save_path}")

        plt.show()

    def generate_report(self) -> Dict:
        """
        Generate comprehensive correlation analysis report

        Returns:
        --------
        Dict
            Report dictionary with analysis results
        """
        if self.correlation_matrix is None:
            self.compute_correlations()

        multicollinearity = self.identify_multicollinearity()
        interactions = self.suggest_interaction_terms()

        report = {
            'n_features': len(self.numeric_features),
            'target_column': self.target_column,
            'top_correlations': self.target_correlations.head(10).to_dict(),
            'multicollinearity_pairs': len(multicollinearity),
            'suggested_interactions': len(interactions),
            'interaction_details': [
                {'feature1': f1, 'feature2': f2, 'score': score}
                for f1, f2, score in interactions
            ]
        }

        print(f"\n{'='*60}")
        print(f"CORRELATION ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total numeric features: {report['n_features']}")
        print(f"Target variable: {report['target_column']}")
        print(f"Multicollinearity issues: {report['multicollinearity_pairs']}")
        print(f"Suggested interactions: {report['suggested_interactions']}")
        print(f"{'='*60}\n")

        return report


if __name__ == "__main__":
    # Example usage
    print("Correlation Analyzer Module - Ready for import")
