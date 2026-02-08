"""
Advanced Feature Selection with Correlation-Based Variable Reintroduction
Adds variables back into the model by finding those that correlate strongly
with existing high-performing features
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from correlation_analyzer import CorrelationAnalyzer
from feature_engineer import FeatureEngineer
from model_optimizer import ModelOptimizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureSelector:
    """
    Advanced feature selection using correlation-based reintroduction
    """

    def __init__(self, data, target, correlation_matrix):
        self.data = data
        self.target = target
        self.corr_matrix = correlation_matrix
        self.selected_features = []
        self.feature_pool = []
        self.reintroduced_features = []

    def initialize_feature_pool(self, min_target_correlation=0.10):
        """
        Initialize pool of candidate features based on minimum target correlation
        """
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numeric_features:
            numeric_features.remove(self.target)

        target_corrs = self.corr_matrix[self.target].drop(self.target)

        # Start with features that have minimum correlation with target
        for feat in numeric_features:
            if abs(target_corrs[feat]) >= min_target_correlation:
                self.selected_features.append(feat)
            else:
                self.feature_pool.append(feat)

        print(f"\n{'='*80}")
        print(f"INITIAL FEATURE POOL")
        print(f"{'='*80}")
        print(f"Selected features (direct target correlation >= {min_target_correlation}): {len(self.selected_features)}")
        print(f"Feature pool (candidates for reintroduction): {len(self.feature_pool)}")
        print(f"\nSelected features:")
        for feat in self.selected_features:
            print(f"  • {feat:20s} → target: {target_corrs[feat]:+.4f}")

        if self.feature_pool:
            print(f"\nFeature pool (weak/indirect relationship with target):")
            for feat in self.feature_pool:
                print(f"  • {feat:20s} → target: {target_corrs[feat]:+.4f}")

        return self.selected_features, self.feature_pool

    def find_correlated_features(self, reference_features, top_n=3, min_correlation=0.30):
        """
        Find features from the pool that correlate strongly with reference features

        Strategy: If a feature correlates strongly with a high-performing feature,
        it might capture similar/complementary information worth adding back
        """
        print(f"\n{'='*80}")
        print(f"FINDING CORRELATED FEATURES TO REINTRODUCE")
        print(f"{'='*80}")
        print(f"Reference features: {reference_features}")
        print(f"Minimum correlation threshold: {min_correlation}")

        candidates = []

        for pool_feat in self.feature_pool:
            max_correlation = 0
            best_reference = None

            # Check correlation with each reference feature
            for ref_feat in reference_features:
                corr = abs(self.corr_matrix.loc[pool_feat, ref_feat])
                if corr > max_correlation:
                    max_correlation = corr
                    best_reference = ref_feat

            if max_correlation >= min_correlation:
                target_corr = self.corr_matrix.loc[pool_feat, self.target]
                # Score: correlation with reference feature + indirect target correlation
                score = max_correlation + abs(target_corr) * 0.5
                candidates.append({
                    'feature': pool_feat,
                    'best_reference': best_reference,
                    'correlation_with_reference': max_correlation,
                    'target_correlation': target_corr,
                    'score': score
                })

        # Sort by score
        if len(candidates) > 0:
            candidates_df = pd.DataFrame(candidates).sort_values('score', ascending=False)
        else:
            candidates_df = pd.DataFrame()

        if len(candidates_df) > 0:
            print(f"\nFound {len(candidates_df)} candidates for reintroduction:")
            print("-" * 80)
            print(f"{'Feature':<15} {'Best Ref':<15} {'Ref Corr':>10} {'Target Corr':>12} {'Score':>10}")
            print("-" * 80)
            for _, row in candidates_df.head(top_n).iterrows():
                print(f"{row['feature']:<15} {row['best_reference']:<15} {row['correlation_with_reference']:>10.4f} {row['target_correlation']:>12.4f} {row['score']:>10.4f}")

            # Add top N to selected features
            top_candidates = candidates_df.head(top_n)['feature'].tolist()
            self.reintroduced_features.extend(top_candidates)
            self.selected_features.extend(top_candidates)

            # Remove from pool
            for feat in top_candidates:
                if feat in self.feature_pool:
                    self.feature_pool.remove(feat)

            print(f"\n✓ Reintroduced {len(top_candidates)} features")
        else:
            print(f"\n⚠ No candidates found with correlation >= {min_correlation}")

        return candidates_df

    def create_interactions_with_reintroduced(self):
        """
        Create interaction terms specifically between:
        1. Original high-correlation features
        2. Reintroduced features
        This captures the indirect relationships through the reintroduced variables
        """
        print(f"\n{'='*80}")
        print(f"CREATING INTERACTION TERMS WITH REINTRODUCED FEATURES")
        print(f"{'='*80}")

        engineer = FeatureEngineer(self.data, self.target)

        # Get top correlated features (exclude reintroduced)
        target_corrs = self.corr_matrix[self.target].drop(self.target)
        top_features = [f for f in self.selected_features if f not in self.reintroduced_features]
        top_features = sorted(top_features, key=lambda x: abs(target_corrs[x]), reverse=True)[:3]

        # Create interactions between top features and reintroduced features
        interaction_pairs = []
        print(f"\nCreating interactions between:")
        print(f"  Top features: {top_features}")
        print(f"  Reintroduced features: {self.reintroduced_features}")
        print()

        for top_feat in top_features:
            for reintro_feat in self.reintroduced_features:
                interaction_pairs.append((top_feat, reintro_feat))
                print(f"  • {top_feat} × {reintro_feat}")

        if interaction_pairs:
            data_with_interactions = engineer.create_interaction_terms(
                interaction_pairs=interaction_pairs,
                interaction_type='multiply'
            )
        else:
            data_with_interactions = self.data.copy()

        # Also create standard interactions among top features
        print(f"\nCreating standard interactions among top features...")
        standard_pairs = []
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                standard_pairs.append((feat1, feat2))
                print(f"  • {feat1} × {feat2}")

        if standard_pairs:
            data_with_interactions = engineer.create_interaction_terms(
                interaction_pairs=standard_pairs,
                interaction_type='multiply'
            )

        summary = engineer.get_feature_summary()

        return data_with_interactions, engineer.created_features

    def visualize_correlation_structure(self):
        """
        Visualize the correlation structure showing:
        1. Selected features
        2. Reintroduced features
        3. Feature pool
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Correlation-Based Feature Reintroduction Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Full correlation matrix
        ax1 = axes[0, 0]
        all_features = self.selected_features + self.feature_pool + [self.target]
        corr_subset = self.corr_matrix.loc[all_features, all_features]
        sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={'label': 'Correlation'}, ax=ax1, annot_kws={'size': 7})
        ax1.set_title('Full Correlation Matrix', fontsize=12, fontweight='bold')
        ax1.tick_params(labelsize=8)

        # 2. Correlation with target (color-coded)
        ax2 = axes[0, 1]
        target_corrs = self.corr_matrix[self.target].drop(self.target).sort_values(key=abs, ascending=True)

        colors = []
        for feat in target_corrs.index:
            if feat in self.reintroduced_features:
                colors.append('orangered')  # Reintroduced
            elif feat in self.selected_features:
                colors.append('steelblue')  # Selected
            else:
                colors.append('gray')  # Pool

        ax2.barh(range(len(target_corrs)), target_corrs.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(target_corrs)))
        ax2.set_yticklabels(target_corrs.index, fontsize=8)
        ax2.set_xlabel('Correlation with Target', fontsize=10)
        ax2.set_title('Feature Correlations (Blue=Selected, Red=Reintroduced, Gray=Pool)',
                      fontsize=11, fontweight='bold')
        ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)

        # 3. Correlation network: reintroduced features with selected features
        ax3 = axes[1, 0]
        ax3.axis('off')
        ax3.set_title('Reintroduced Features → Selected Features',
                      fontsize=12, fontweight='bold')

        if self.reintroduced_features:
            network_data = []
            for reintro_feat in self.reintroduced_features:
                for sel_feat in [f for f in self.selected_features if f not in self.reintroduced_features]:
                    corr = self.corr_matrix.loc[reintro_feat, sel_feat]
                    if abs(corr) > 0.3:
                        network_data.append({
                            'Reintroduced': reintro_feat,
                            'Selected': sel_feat,
                            'Correlation': corr
                        })

            if network_data:
                network_df = pd.DataFrame(network_data).sort_values('Correlation', key=abs, ascending=False)

                table_data = []
                for _, row in network_df.head(15).iterrows():
                    table_data.append([
                        row['Reintroduced'],
                        '→',
                        row['Selected'],
                        f"{row['Correlation']:+.4f}"
                    ])

                table = ax3.table(cellText=table_data,
                                colLabels=['Reintroduced', '', 'Selected Feature', 'Corr'],
                                cellLoc='left',
                                loc='center',
                                colWidths=[0.3, 0.05, 0.3, 0.15])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)

                # Style header
                for i in range(4):
                    table[(0, i)].set_facecolor('#4472C4')
                    table[(0, i)].set_text_props(weight='bold', color='white')
            else:
                ax3.text(0.5, 0.5, 'No strong correlations found',
                        ha='center', va='center', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'No features reintroduced',
                    ha='center', va='center', fontsize=12)

        # 4. Feature selection summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title('Feature Selection Summary', fontsize=12, fontweight='bold')

        summary_text = f"""
FEATURE SELECTION BREAKDOWN

Initial Selection (Direct Target Correlation):
  • Count: {len([f for f in self.selected_features if f not in self.reintroduced_features])}
  • Features: {', '.join([f for f in self.selected_features if f not in self.reintroduced_features][:5])}...

Reintroduced (Indirect via Correlation):
  • Count: {len(self.reintroduced_features)}
  • Features: {', '.join(self.reintroduced_features) if self.reintroduced_features else 'None'}

Remaining in Pool:
  • Count: {len(self.feature_pool)}
  • Features: {', '.join(self.feature_pool) if self.feature_pool else 'None'}

TOTAL FEATURES: {len(self.selected_features)}

RATIONALE:
Reintroduced features may not correlate directly
with the target, but they correlate strongly with
features that DO predict the target. This captures
indirect relationships and provides complementary
information that can improve model performance.
        """

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('results/plots/advanced_feature_selection.png',
                    dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: results/plots/advanced_feature_selection.png")
        plt.show()


def main():
    """
    Main workflow for advanced feature selection
    """
    print("\n" + "="*80)
    print(" " * 20 + "ADVANCED FEATURE SELECTION WITH REINTRODUCTION")
    print("="*80)

    # Load data
    print("\n▸ Loading data...")
    loader = DataLoader()
    data, target = loader.load_sample_dataset('diabetes')

    # Compute correlation matrix
    print("\n▸ Computing correlation matrix...")
    analyzer = CorrelationAnalyzer(data, target)
    corr_matrix = analyzer.compute_correlations(method='pearson')

    # Initialize advanced selector
    selector = AdvancedFeatureSelector(data, target, corr_matrix)

    # Step 1: Initialize feature pools
    # Use lower threshold to have more features in the pool for demonstration
    selected, pool = selector.initialize_feature_pool(min_target_correlation=0.25)

    # Step 2: Identify features to reintroduce
    # Strategy: Find features in pool that correlate strongly with top-performing features
    top_performers = analyzer.target_correlations.head(3).index.tolist()
    print(f"\n▸ Top performing features: {top_performers}")

    candidates = selector.find_correlated_features(
        reference_features=top_performers,
        top_n=2,  # Reintroduce top 2 candidates
        min_correlation=0.25
    )

    # Step 3: Create interactions with reintroduced features
    print(f"\n▸ Creating enhanced feature set...")
    data_enhanced, interaction_features = selector.create_interactions_with_reintroduced()

    # Step 4: Visualize correlation structure
    print(f"\n▸ Generating visualizations...")
    selector.visualize_correlation_structure()

    # Step 5: Train and compare models
    print(f"\n▸ Training models...")

    # Model 1: Baseline (only direct high-correlation features)
    baseline_features = [f for f in selected if f not in selector.reintroduced_features]
    print(f"\nBaseline features: {baseline_features}")

    # Model 2: With reintroduced features
    reintroduced_features = selected
    print(f"With reintroduced: {reintroduced_features}")

    # Model 3: With reintroduced + interactions
    all_features = [f for f in data_enhanced.columns if f != target]
    print(f"With interactions: {len(all_features)} total features")

    # Train models
    task_type = 'regression'
    optimizer = ModelOptimizer(task_type=task_type, test_size=0.2, random_state=42)

    results = []

    # Baseline
    data_baseline = data[baseline_features + [target]]
    optimizer.prepare_data(data_baseline, target, scale_features=True)
    baseline_model = optimizer.train_baseline_model(model_type='auto')
    results.append({
        'Model': 'Baseline',
        'Features': len(baseline_features),
        'Train R²': optimizer.baseline_results['train_r2'],
        'Test R²': optimizer.baseline_results['test_r2'],
        'Test RMSE': optimizer.baseline_results['test_rmse'],
        'Test MAE': optimizer.baseline_results['test_mae']
    })

    # With reintroduced
    data_reintro = data[reintroduced_features + [target]]
    optimizer_reintro = ModelOptimizer(task_type=task_type, test_size=0.2, random_state=42)
    optimizer_reintro.prepare_data(data_reintro, target, scale_features=True)
    reintro_model = optimizer_reintro.train_baseline_model(model_type='auto')
    results.append({
        'Model': '+ Reintroduced',
        'Features': len(reintroduced_features),
        'Train R²': optimizer_reintro.baseline_results['train_r2'],
        'Test R²': optimizer_reintro.baseline_results['test_r2'],
        'Test RMSE': optimizer_reintro.baseline_results['test_rmse'],
        'Test MAE': optimizer_reintro.baseline_results['test_mae']
    })

    # With interactions
    optimizer_full = ModelOptimizer(task_type=task_type, test_size=0.2, random_state=42)
    optimizer_full.prepare_data(data_enhanced, target, scale_features=True)
    full_model = optimizer_full.train_baseline_model(model_type='auto')
    results.append({
        'Model': '+ Interactions',
        'Features': len(all_features),
        'Train R²': optimizer_full.baseline_results['train_r2'],
        'Test R²': optimizer_full.baseline_results['test_r2'],
        'Test RMSE': optimizer_full.baseline_results['test_rmse'],
        'Test MAE': optimizer_full.baseline_results['test_mae']
    })

    # Display results
    results_df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"{'MODEL COMPARISON':^80}")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    # Calculate improvements
    baseline_score = results[0]['Test R²']
    reintro_improvement = ((results[1]['Test R²'] - baseline_score) / baseline_score) * 100
    full_improvement = ((results[2]['Test R²'] - baseline_score) / baseline_score) * 100

    print(f"\n{'='*80}")
    print(f"IMPROVEMENTS vs. BASELINE")
    print(f"{'='*80}")
    print(f"With reintroduced features: {reintro_improvement:+.2f}%")
    print(f"With interactions:          {full_improvement:+.2f}%")
    print(f"{'='*80}\n")

    # Save results
    results_df.to_csv('results/advanced_feature_selection_results.csv', index=False)
    print("✓ Results saved: results/advanced_feature_selection_results.csv")

    return {
        'results': results_df,
        'reintroduced_features': selector.reintroduced_features,
        'improvement_reintro': reintro_improvement,
        'improvement_full': full_improvement
    }


if __name__ == "__main__":
    results = main()
    print(f"\n✓ Analysis complete!")
    print(f"Reintroduced features: {results['reintroduced_features']}")
    print(f"Final improvement: {results['improvement_full']:+.2f}%")
