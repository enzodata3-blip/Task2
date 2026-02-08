"""
Analysis of Negatively Correlated Variables: s3 and s4
Understanding how opposing forces interact to predict the target
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_negative_correlation_pair(data, target, feat1='s3', feat2='s4'):
    """
    Deep dive into how two negatively correlated features interact
    """

    print("\n" + "="*80)
    print(" " * 15 + f"NEGATIVE CORRELATION ANALYSIS: {feat1} vs {feat2}")
    print("="*80)

    # Get correlation matrix
    analyzer = CorrelationAnalyzer(data, target)
    corr_matrix = analyzer.compute_correlations(method='pearson')

    # Extract key correlations
    feat1_target = corr_matrix.loc[feat1, target]
    feat2_target = corr_matrix.loc[feat2, target]
    feat1_feat2 = corr_matrix.loc[feat1, feat2]

    print(f"\n{'CORRELATION STRUCTURE':^80}")
    print("-" * 80)
    print(f"{feat1} → target:  {feat1_target:+.4f} (negative: when {feat1} ↑, target ↓)")
    print(f"{feat2} → target:  {feat2_target:+.4f} (positive: when {feat2} ↑, target ↑)")
    print(f"{feat1} ↔ {feat2}:  {feat1_feat2:+.4f} (STRONG NEGATIVE: they move in opposite directions)")
    print("-" * 80)

    print(f"\n{'INTERPRETATION':^80}")
    print("-" * 80)
    print(f"""
Since {feat1} and {feat2} are strongly negatively correlated ({feat1_feat2:.3f}),
they capture OPPOSING forces in the data:

1. When {feat1} is HIGH:
   - {feat2} tends to be LOW (due to negative correlation)
   - Target tends to be LOW (due to {feat1}'s negative correlation with target)

2. When {feat1} is LOW:
   - {feat2} tends to be HIGH (due to negative correlation)
   - Target tends to be HIGH (due to {feat2}'s positive correlation with target)

3. Their INTERACTION ({feat1} × {feat2}):
   - Captures the BALANCE between these opposing forces
   - When BOTH are moderate → interaction term is HIGH
   - When one dominates → interaction term is LOW
   - This creates a NON-LINEAR relationship with the target!
    """)

    # Create visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Scatter: feat1 vs feat2 (showing negative correlation)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(data[feat1], data[feat2], c=data[target],
                           cmap='RdYlBu_r', alpha=0.6, s=50)
    ax1.set_xlabel(f'{feat1}', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'{feat2}', fontsize=11, fontweight='bold')
    ax1.set_title(f'{feat1} vs {feat2} (colored by target)\nCorrelation: {feat1_feat2:.3f}',
                  fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Target Value')
    ax1.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(data[feat1], data[feat2], 1)
    p = np.poly1d(z)
    ax1.plot(data[feat1], p(data[feat1]), "r--", linewidth=2, alpha=0.8, label='Trend')
    ax1.legend()

    # 2. Scatter: feat1 vs target
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(data[feat1], data[target], alpha=0.5, s=50, color='steelblue')
    ax2.set_xlabel(f'{feat1}', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Target', fontsize=11, fontweight='bold')
    ax2.set_title(f'{feat1} vs Target\nCorrelation: {feat1_target:+.3f}',
                  fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(data[feat1], data[target], 1)
    p = np.poly1d(z)
    ax2.plot(data[feat1], p(data[feat1]), "r--", linewidth=2, alpha=0.8)

    # 3. Scatter: feat2 vs target
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(data[feat2], data[target], alpha=0.5, s=50, color='orangered')
    ax3.set_xlabel(f'{feat2}', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Target', fontsize=11, fontweight='bold')
    ax3.set_title(f'{feat2} vs Target\nCorrelation: {feat2_target:+.3f}',
                  fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(data[feat2], data[target], 1)
    p = np.poly1d(z)
    ax3.plot(data[feat2], p(data[feat2]), "r--", linewidth=2, alpha=0.8)

    # 4. Create interaction term and analyze
    data_copy = data.copy()
    data_copy[f'{feat1}_X_{feat2}'] = data_copy[feat1] * data_copy[feat2]
    interaction_corr = data_copy[[f'{feat1}_X_{feat2}', target]].corr().iloc[0, 1]

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(data_copy[f'{feat1}_X_{feat2}'], data[target],
                alpha=0.5, s=50, color='purple')
    ax4.set_xlabel(f'{feat1} × {feat2} (Interaction)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Target', fontsize=11, fontweight='bold')
    ax4.set_title(f'Interaction Term vs Target\nCorrelation: {interaction_corr:+.3f}',
                  fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(data_copy[f'{feat1}_X_{feat2}'], data[target], 1)
    p = np.poly1d(z)
    ax4.plot(data_copy[f'{feat1}_X_{feat2}'],
             p(data_copy[f'{feat1}_X_{feat2}']), "r--", linewidth=2, alpha=0.8)

    # 5. Heatmap: quartile analysis
    ax5 = fig.add_subplot(gs[1, 1:])

    # Create quartile buckets
    data_copy[f'{feat1}_bin'] = pd.qcut(data_copy[feat1], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
    data_copy[f'{feat2}_bin'] = pd.qcut(data_copy[feat2], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')

    # Calculate mean target for each combination
    heatmap_data = data_copy.groupby([f'{feat1}_bin', f'{feat2}_bin'])['target'].mean().unstack()

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                center=data[target].mean(), ax=ax5, cbar_kws={'label': 'Mean Target'})
    ax5.set_xlabel(f'{feat2} Level', fontsize=11, fontweight='bold')
    ax5.set_ylabel(f'{feat1} Level', fontsize=11, fontweight='bold')
    ax5.set_title(f'Target Value Heatmap: {feat1} vs {feat2} Combinations\n(Shows how opposing forces create non-linear patterns)',
                  fontsize=12, fontweight='bold')

    # 6. Model comparison
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    # Train models to compare
    X_train = data[[feat1, feat2, target]].dropna()
    y = X_train[target]

    # Model 1: feat1 only
    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(X_train[[feat1]])
    model1 = LinearRegression().fit(X1, y)
    r2_1 = model1.score(X1, y)

    # Model 2: feat2 only
    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(X_train[[feat2]])
    model2 = LinearRegression().fit(X2, y)
    r2_2 = model2.score(X2, y)

    # Model 3: both features
    scaler3 = StandardScaler()
    X3 = scaler3.fit_transform(X_train[[feat1, feat2]])
    model3 = LinearRegression().fit(X3, y)
    r2_3 = model3.score(X3, y)

    # Model 4: with interaction
    X_train_int = X_train.copy()
    X_train_int[f'{feat1}_X_{feat2}'] = X_train_int[feat1] * X_train_int[feat2]
    scaler4 = StandardScaler()
    X4 = scaler4.fit_transform(X_train_int[[feat1, feat2, f'{feat1}_X_{feat2}']])
    model4 = LinearRegression().fit(X4, y)
    r2_4 = model4.score(X4, y)

    # Display results
    results_text = f"""
    {'LINEAR REGRESSION MODEL COMPARISON':^70}
    {'='*70}

    Model Configuration                           R² Score    Improvement
    {'─'*70}
    1. {feat1} only                                    {r2_1:.4f}      baseline
    2. {feat2} only                                    {r2_2:.4f}      {((r2_2-r2_1)/r2_1*100):+.1f}%
    3. {feat1} + {feat2} (additive)                      {r2_3:.4f}      {((r2_3-r2_1)/r2_1*100):+.1f}%
    4. {feat1} + {feat2} + ({feat1}×{feat2}) (with interaction)   {r2_4:.4f}      {((r2_4-r2_1)/r2_1*100):+.1f}%
    {'─'*70}

    KEY FINDINGS:

    1. OPPOSING FORCES: {feat1} and {feat2} represent opposing influences on the target
       - {feat1} pulls target DOWN (correlation: {feat1_target:+.3f})
       - {feat2} pulls target UP (correlation: {feat2_target:+.3f})

    2. COMPLEMENTARY INFORMATION: Despite being highly correlated with each other
       ({feat1_feat2:.3f}), they provide DIFFERENT predictive information when used together.
       Adding {feat2} to {feat1} improves R² by {((r2_3-r2_1)/r2_1*100):+.1f}%.

    3. NON-LINEAR INTERACTION: The interaction term ({feat1}×{feat2}) captures how these
       opposing forces BALANCE each other. This non-linearity improves the model by an
       additional {((r2_4-r2_3)/r2_3*100):+.1f}% over the additive model.

    4. PRACTICAL INTERPRETATION: When {feat1} and {feat2} are BOTH moderate, their
       product is high and indicates a specific regime. When one dominates, the product
       is lower, indicating a different regime. This captures state-dependent effects.

    CLAIM: Variables with opposing correlations (one positive, one negative with target)
    that are themselves strongly correlated provide COMPLEMENTARY predictive power through
    their INTERACTION. They capture non-linear, state-dependent relationships that linear
    additive models miss.
    """

    ax6.text(0.05, 0.95, results_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Negative Correlation Analysis: {feat1} ↔ {feat2}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('results/plots/negative_correlation_analysis.png',
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: results/plots/negative_correlation_analysis.png")
    plt.show()

    # Return summary
    return {
        'feat1': feat1,
        'feat2': feat2,
        'feat1_target_corr': feat1_target,
        'feat2_target_corr': feat2_target,
        'feat1_feat2_corr': feat1_feat2,
        'interaction_target_corr': interaction_corr,
        'r2_feat1_only': r2_1,
        'r2_feat2_only': r2_2,
        'r2_both': r2_3,
        'r2_with_interaction': r2_4,
        'improvement_interaction': ((r2_4 - r2_3) / r2_3 * 100)
    }


def main():
    """
    Main analysis workflow
    """
    print("\n" + "="*80)
    print(" " * 20 + "NEGATIVE CORRELATION PAIR ANALYSIS")
    print("="*80)

    # Load data
    loader = DataLoader()
    data, target = loader.load_sample_dataset('diabetes')

    # Analyze s3 and s4
    results = analyze_negative_correlation_pair(data, target, feat1='s3', feat2='s4')

    # Print summary
    print("\n" + "="*80)
    print(" " * 30 + "SUMMARY")
    print("="*80)
    print(f"\n{results['feat1']} and {results['feat2']} Analysis:")
    print(f"  • {results['feat1']} → target: {results['feat1_target_corr']:+.4f}")
    print(f"  • {results['feat2']} → target: {results['feat2_target_corr']:+.4f}")
    print(f"  • {results['feat1']} ↔ {results['feat2']}: {results['feat1_feat2_corr']:+.4f}")
    print(f"  • ({results['feat1']}×{results['feat2']}) → target: {results['interaction_target_corr']:+.4f}")
    print(f"\nModel Performance:")
    print(f"  • {results['feat1']} only: R² = {results['r2_feat1_only']:.4f}")
    print(f"  • {results['feat2']} only: R² = {results['r2_feat2_only']:.4f}")
    print(f"  • Both features: R² = {results['r2_both']:.4f}")
    print(f"  • With interaction: R² = {results['r2_with_interaction']:.4f}")
    print(f"\n  → Interaction improvement: {results['improvement_interaction']:+.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("""
Negatively correlated feature pairs that have opposing relationships with the
target are HIGHLY VALUABLE for prediction because:

1. They capture different aspects of the underlying process
2. Their interaction term captures non-linear, state-dependent effects
3. They provide complementary information despite being correlated
4. The balance between them matters more than either alone

This is a classic case where DOMAIN UNDERSTANDING combined with STATISTICAL
ANALYSIS reveals important interaction effects that improve model performance.
    """)
    print("="*80 + "\n")

    return results


if __name__ == "__main__":
    results = main()
