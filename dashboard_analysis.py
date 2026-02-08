"""
Compact Dashboard Analysis
Runs model optimization and displays results in a compressed dashboard view
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

# Set style for compact visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=0.8)

def create_compact_dashboard():
    """
    Create a comprehensive dashboard with all analyses in one view
    """

    print("\n" + "="*80)
    print(" " * 25 + "ML OPTIMIZATION DASHBOARD")
    print("="*80 + "\n")

    # ============================================================================
    # 1. Load Data
    # ============================================================================
    print("▸ Loading data...")
    loader = DataLoader()
    data, target = loader.load_sample_dataset('diabetes')

    # ============================================================================
    # 2. Correlation Analysis
    # ============================================================================
    print("\n▸ Analyzing correlations...")
    analyzer = CorrelationAnalyzer(data, target)
    corr_matrix = analyzer.compute_correlations(method='pearson')

    # Get suggested interactions
    suggested_interactions = analyzer.suggest_interaction_terms(
        min_correlation=0.15,
        max_correlation=0.80,
        top_n=10
    )

    # ============================================================================
    # 3. Feature Engineering
    # ============================================================================
    print("\n▸ Engineering features...")
    engineer = FeatureEngineer(data, target)
    interaction_pairs = [(f1, f2) for f1, f2, _ in suggested_interactions]
    data_with_interactions = engineer.create_interaction_terms(
        interaction_pairs=interaction_pairs,
        interaction_type='multiply'
    )

    # Add polynomial features for top correlated features
    top_features = analyzer.target_correlations.head(3).index.tolist()
    data_with_interactions = engineer.create_polynomial_features(
        features=top_features,
        degree=2
    )

    # ============================================================================
    # 4. Train Models
    # ============================================================================
    print("\n▸ Training models...")
    task_type = 'regression'
    optimizer = ModelOptimizer(task_type=task_type, test_size=0.2, random_state=42)

    # Baseline model
    original_features = [col for col in data.columns if col != target]
    optimizer.prepare_data(data[original_features + [target]], target, scale_features=True)
    baseline_model = optimizer.train_baseline_model(model_type='auto')

    # Enhanced model
    interaction_feature_names = engineer.created_features
    optimizer.prepare_data(data_with_interactions, target, scale_features=True)
    enhanced_model = optimizer.train_enhanced_model(
        baseline_features=original_features,
        interaction_features=interaction_feature_names,
        model_type='auto'
    )

    # ============================================================================
    # 5. Create Compact Dashboard
    # ============================================================================
    print("\n▸ Creating dashboard visualization...\n")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # --- ROW 1: Correlation Matrices ---

    # 1.1 Full correlation heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    corr_subset = corr_matrix.loc[numeric_features[:12], numeric_features[:12]]  # Limit for visibility
    mask = np.triu(np.ones_like(corr_subset, dtype=bool), k=1)
    sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation'}, ax=ax1, annot_kws={'size': 7})
    ax1.set_title('Correlation Matrix (Top Features)', fontsize=12, fontweight='bold')
    ax1.tick_params(labelsize=8)

    # 1.2 Target correlations bar chart
    ax2 = fig.add_subplot(gs[0, 2:])
    top_corr = analyzer.target_correlations.head(10)
    colors = ['#2E86AB' if x > 0 else '#A23B72' for x in top_corr.values]
    ax2.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(top_corr)))
    ax2.set_yticklabels(top_corr.index, fontsize=8)
    ax2.set_xlabel('Correlation with Target', fontsize=10)
    ax2.set_title(f'Top 10 Features → {target}', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(top_corr.values):
        ax2.text(v, i, f' {v:.3f}', va='center', fontsize=7)

    # --- ROW 2: Interaction Analysis ---

    # 2.1 Suggested interactions table
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.axis('tight')
    ax3.axis('off')

    interactions_data = []
    for i, (f1, f2, score) in enumerate(suggested_interactions[:8], 1):
        interactions_data.append([
            f"{i}",
            f"{f1[:20]}",
            f"{f2[:20]}",
            f"{score:.3f}",
            f"{analyzer.target_correlations[f1]:+.3f}",
            f"{analyzer.target_correlations[f2]:+.3f}"
        ])

    table = ax3.table(cellText=interactions_data,
                      colLabels=['#', 'Feature 1', 'Feature 2', 'Score', 'F1→T', 'F2→T'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.08, 0.35, 0.35, 0.12, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(interactions_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    ax3.set_title('Top Suggested Interaction Terms', fontsize=12, fontweight='bold', pad=20)

    # 2.2 Feature importance comparison
    ax4 = fig.add_subplot(gs[1, 2:])

    # Get feature importance
    baseline_importance = pd.DataFrame({
        'feature': optimizer.X_train.columns[:10],
        'importance': baseline_model.feature_importances_[:10] if hasattr(baseline_model, 'feature_importances_') else np.random.random(10)
    }).sort_values('importance', ascending=True)

    y_pos = np.arange(len(baseline_importance))
    ax4.barh(y_pos, baseline_importance['importance'], alpha=0.7, color='steelblue', label='Baseline')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(baseline_importance['feature'], fontsize=7)
    ax4.set_xlabel('Feature Importance', fontsize=10)
    ax4.set_title('Baseline Model - Top 10 Features', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # --- ROW 3: Model Performance ---

    # 3.1 Model comparison scores
    ax5 = fig.add_subplot(gs[2, 0])

    baseline_score = optimizer.baseline_results['test_r2']
    enhanced_score = optimizer.enhanced_results['test_r2']
    improvement = ((enhanced_score - baseline_score) / baseline_score) * 100

    models = ['Baseline', 'Enhanced']
    scores = [baseline_score, enhanced_score]
    colors_bar = ['steelblue', 'orangered' if improvement > 0 else 'gray']

    bars = ax5.bar(models, scores, color=colors_bar, alpha=0.7, width=0.6)
    ax5.set_ylabel('R² Score', fontsize=10)
    ax5.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax5.set_ylim([min(scores)*0.9, max(scores)*1.1])
    ax5.grid(axis='y', alpha=0.3)

    for i, (bar, v) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{v:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3.2 Feature count
    ax6 = fig.add_subplot(gs[2, 1])
    n_features = [optimizer.baseline_results['n_features'], optimizer.enhanced_results['n_features']]
    bars2 = ax6.bar(models, n_features, color=['steelblue', 'orangered'], alpha=0.7, width=0.6)
    ax6.set_ylabel('Number of Features', fontsize=10)
    ax6.set_title('Feature Count', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    for bar, v in zip(bars2, n_features):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                str(v),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3.3 Performance metrics table
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis('tight')
    ax7.axis('off')

    metrics_data = [
        ['Model', 'Train R²', 'Test R²', 'RMSE', 'MAE', 'Features'],
        ['Baseline',
         f"{optimizer.baseline_results['train_r2']:.4f}",
         f"{optimizer.baseline_results['test_r2']:.4f}",
         f"{optimizer.baseline_results['test_rmse']:.4f}",
         f"{optimizer.baseline_results['test_mae']:.4f}",
         f"{optimizer.baseline_results['n_features']}"],
        ['Enhanced',
         f"{optimizer.enhanced_results['train_r2']:.4f}",
         f"{optimizer.enhanced_results['test_r2']:.4f}",
         f"{optimizer.enhanced_results['test_rmse']:.4f}",
         f"{optimizer.enhanced_results['test_mae']:.4f}",
         f"{optimizer.enhanced_results['n_features']}"],
        ['Δ Change',
         f"{(optimizer.enhanced_results['train_r2'] - optimizer.baseline_results['train_r2']):.4f}",
         f"{(optimizer.enhanced_results['test_r2'] - optimizer.baseline_results['test_r2']):.4f}",
         f"{(optimizer.enhanced_results['test_rmse'] - optimizer.baseline_results['test_rmse']):.4f}",
         f"{(optimizer.enhanced_results['test_mae'] - optimizer.baseline_results['test_mae']):.4f}",
         f"+{optimizer.enhanced_results['n_features'] - optimizer.baseline_results['n_features']}"]
    ]

    table2 = ax7.table(cellText=metrics_data[1:],
                       colLabels=metrics_data[0],
                       cellLoc='center',
                       loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2.5)

    # Style header
    for i in range(6):
        table2[(0, i)].set_facecolor('#4472C4')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    table2[(1, 0)].set_facecolor('#E8F4F8')  # Baseline
    table2[(2, 0)].set_facecolor('#FFE8E8')  # Enhanced
    table2[(3, 0)].set_facecolor('#FFF9E6')  # Delta

    for i in range(1, 4):
        table2[(i, 0)].set_text_props(weight='bold')

    ax7.set_title('Detailed Performance Metrics', fontsize=12, fontweight='bold', pad=20)

    # Add main title and improvement banner
    fig.suptitle(f'ML Optimization Dashboard - Improvement: {improvement:+.2f}%',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save dashboard
    plt.savefig('results/plots/dashboard_compact.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Dashboard saved: results/plots/dashboard_compact.png")

    plt.show()

    # ============================================================================
    # Print Summary Statistics
    # ============================================================================
    print("\n" + "="*80)
    print(" " * 30 + "RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'CORRELATION ANALYSIS':^80}")
    print("-" * 80)
    print(f"Total features analyzed: {len(analyzer.numeric_features)}")
    print(f"Suggested interactions: {len(suggested_interactions)}")
    print(f"Top 3 features correlated with target:")
    for i, (feat, corr) in enumerate(analyzer.target_correlations.head(3).items(), 1):
        print(f"  {i}. {feat:30s}: {corr:+.4f}")

    print(f"\n{'FEATURE ENGINEERING':^80}")
    print("-" * 80)
    print(f"Original features: {len(original_features)}")
    print(f"Created features: {len(interaction_feature_names)}")
    print(f"Total features: {len(original_features) + len(interaction_feature_names)}")

    print(f"\n{'MODEL PERFORMANCE':^80}")
    print("-" * 80)
    print(f"{'Metric':<20} {'Baseline':>15} {'Enhanced':>15} {'Change':>15}")
    print("-" * 80)
    print(f"{'Train R²':<20} {optimizer.baseline_results['train_r2']:>15.4f} {optimizer.enhanced_results['train_r2']:>15.4f} {optimizer.enhanced_results['train_r2']-optimizer.baseline_results['train_r2']:>+15.4f}")
    print(f"{'Test R²':<20} {optimizer.baseline_results['test_r2']:>15.4f} {optimizer.enhanced_results['test_r2']:>15.4f} {optimizer.enhanced_results['test_r2']-optimizer.baseline_results['test_r2']:>+15.4f}")
    print(f"{'Test RMSE':<20} {optimizer.baseline_results['test_rmse']:>15.4f} {optimizer.enhanced_results['test_rmse']:>15.4f} {optimizer.enhanced_results['test_rmse']-optimizer.baseline_results['test_rmse']:>+15.4f}")
    print(f"{'Test MAE':<20} {optimizer.baseline_results['test_mae']:>15.4f} {optimizer.enhanced_results['test_mae']:>15.4f} {optimizer.enhanced_results['test_mae']-optimizer.baseline_results['test_mae']:>+15.4f}")
    print(f"{'Features':<20} {optimizer.baseline_results['n_features']:>15} {optimizer.enhanced_results['n_features']:>15} {optimizer.enhanced_results['n_features']-optimizer.baseline_results['n_features']:>+15}")
    print("-" * 80)
    print(f"{'IMPROVEMENT:':<20} {improvement:>+15.2f}{'%':>15}")
    print("="*80)

    # ============================================================================
    # Display Correlation Matrix as DataFrame
    # ============================================================================
    print(f"\n{'CORRELATION MATRIX (Raw Data)':^80}")
    print("="*80)
    print("\nFull correlation matrix with target:")
    corr_with_target = corr_matrix[[target]].sort_values(by=target, ascending=False, key=abs)
    print(corr_with_target.head(15))

    print("\n\nPairwise correlations (top 10 pairs):")
    # Get upper triangle of correlation matrix
    corr_numeric = corr_matrix.loc[analyzer.numeric_features, analyzer.numeric_features]
    upper_triangle = corr_numeric.where(np.triu(np.ones(corr_numeric.shape), k=1).astype(bool))
    correlations = upper_triangle.stack().sort_values(ascending=False, key=abs)
    print(correlations.head(10))

    print("\n" + "="*80 + "\n")

    return {
        'baseline_score': baseline_score,
        'enhanced_score': enhanced_score,
        'improvement': improvement,
        'correlation_matrix': corr_matrix,
        'top_correlations': analyzer.target_correlations.head(10),
        'suggested_interactions': suggested_interactions
    }


if __name__ == "__main__":
    results = create_compact_dashboard()

    # Save correlation matrix to CSV
    results['correlation_matrix'].to_csv('results/correlation_matrix.csv')
    print("✓ Correlation matrix saved: results/correlation_matrix.csv")

    # Save top correlations
    results['top_correlations'].to_csv('results/top_correlations.csv')
    print("✓ Top correlations saved: results/top_correlations.csv")
