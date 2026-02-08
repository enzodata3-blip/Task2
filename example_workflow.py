"""
Complete Example Workflow
Demonstrates the full machine learning optimization pipeline with interaction terms
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from correlation_analyzer import CorrelationAnalyzer
from feature_engineer import FeatureEngineer
from model_optimizer import ModelOptimizer
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Main workflow demonstrating human-in-the-loop model optimization
    """

    print("\n" + "="*80)
    print(" "*20 + "ML MODEL OPTIMIZATION WITH INTERACTION TERMS")
    print("="*80)

    # ============================================================================
    # STEP 1: Load Data
    # ============================================================================
    print("\n" + "▸"*40)
    print("STEP 1: Data Loading")
    print("▸"*40)

    loader = DataLoader()

    # Option 1: Load your own CSV
    # data, target = loader.load_csv('data/your_dataset.csv', 'target_column'), 'target_column'

    # Option 2: Load sample dataset (for demonstration)
    data, target = loader.load_sample_dataset('diabetes')  # or 'breast_cancer', 'california_housing'

    # Handle missing values if any
    data = loader.handle_missing_values(strategy='median')

    # ============================================================================
    # STEP 2: Correlation Analysis
    # ============================================================================
    print("\n" + "▸"*40)
    print("STEP 2: Correlation Analysis")
    print("▸"*40)

    analyzer = CorrelationAnalyzer(data, target)

    # Compute correlation matrix
    corr_matrix = analyzer.compute_correlations(method='pearson')

    # Identify multicollinearity issues
    multicollinearity = analyzer.identify_multicollinearity(threshold=0.85)

    # Suggest interaction terms based on correlation
    suggested_interactions = analyzer.suggest_interaction_terms(
        min_correlation=0.15,
        max_correlation=0.80,
        top_n=10
    )

    # Generate visualizations
    analyzer.plot_correlation_heatmap(
        figsize=(12, 10),
        save_path='results/plots/correlation_heatmap.png'
    )

    analyzer.plot_target_correlations(
        top_n=15,
        save_path='results/plots/target_correlations.png'
    )

    # Generate report
    corr_report = analyzer.generate_report()

    # ============================================================================
    # STEP 3: Feature Engineering
    # ============================================================================
    print("\n" + "▸"*40)
    print("STEP 3: Feature Engineering")
    print("▸"*40)

    engineer = FeatureEngineer(data, target)

    # Extract feature pairs from suggested interactions
    interaction_pairs = [(feat1, feat2) for feat1, feat2, _ in suggested_interactions]

    # Create interaction terms (multiplication is most common)
    data_with_interactions = engineer.create_interaction_terms(
        interaction_pairs=interaction_pairs,
        interaction_type='multiply'  # Can also use 'all' for all types
    )

    # Optional: Create polynomial features for top correlated features
    top_features = analyzer.target_correlations.head(5).index.tolist()
    data_with_interactions = engineer.create_polynomial_features(
        features=top_features,
        degree=2
    )

    # Optional: Create log features for skewed distributions
    # data_with_interactions = engineer.create_log_features(
    #     features=['feature_name'],
    #     handle_negatives='shift'
    # )

    # Remove low variance features
    data_with_interactions = engineer.remove_low_variance_features(threshold=0.01)

    # Get feature engineering summary
    feature_summary = engineer.get_feature_summary()

    # ============================================================================
    # STEP 4: Model Training and Comparison
    # ============================================================================
    print("\n" + "▸"*40)
    print("STEP 4: Model Training & Comparison")
    print("▸"*40)

    # Determine task type
    task_type = 'regression' if data[target].nunique() > 20 else 'classification'
    print(f"\nDetected task type: {task_type.upper()}")

    optimizer = ModelOptimizer(task_type=task_type, test_size=0.2, random_state=42)

    # Prepare data (baseline - original features only)
    original_features = [col for col in data.columns if col != target]
    optimizer.prepare_data(data[original_features + [target]], target, scale_features=True)

    # Train baseline model (without interactions)
    print("\n" + "-"*60)
    baseline_model = optimizer.train_baseline_model(model_type='auto')

    # Get baseline feature importance
    baseline_importance = optimizer.get_feature_importance(
        baseline_model,
        optimizer.X_train.columns.tolist(),
        top_n=15
    )

    # Prepare enhanced data with interactions
    enhanced_data = data_with_interactions.copy()
    interaction_feature_names = engineer.created_features

    # Prepare enhanced model data
    optimizer.prepare_data(enhanced_data, target, scale_features=True)

    # Train enhanced model (with interactions)
    print("\n" + "-"*60)
    enhanced_model = optimizer.train_enhanced_model(
        baseline_features=original_features,
        interaction_features=interaction_feature_names,
        model_type='auto'
    )

    # Get enhanced feature importance
    enhanced_importance = optimizer.get_feature_importance(
        enhanced_model,
        [f for f in original_features + interaction_feature_names if f in optimizer.X_train.columns],
        top_n=15
    )

    # Compare models
    comparison = optimizer.compare_models()

    # Save comparison results
    comparison.to_csv('results/model_comparison.csv', index=False)
    print("\n✓ Comparison results saved to: results/model_comparison.csv")

    # Save models
    optimizer.save_models(
        baseline_path='results/models/baseline_model.joblib',
        enhanced_path='results/models/enhanced_model.joblib'
    )

    # ============================================================================
    # STEP 5: Human-in-the-Loop Analysis & Recommendations
    # ============================================================================
    print("\n" + "▸"*40)
    print("STEP 5: Analysis & Recommendations")
    print("▸"*40)

    # Calculate performance improvement
    if task_type == 'classification':
        baseline_score = optimizer.baseline_results['test_accuracy']
        enhanced_score = optimizer.enhanced_results['test_accuracy']
        metric_name = 'Accuracy'
    else:
        baseline_score = optimizer.baseline_results['test_r2']
        enhanced_score = optimizer.enhanced_results['test_r2']
        metric_name = 'R² Score'

    improvement = ((enhanced_score - baseline_score) / baseline_score) * 100

    print(f"\n{'PERFORMANCE SUMMARY':=^60}")
    print(f"\nBaseline {metric_name}:  {baseline_score:.4f}")
    print(f"Enhanced {metric_name}:  {enhanced_score:.4f}")
    print(f"Improvement:        {improvement:+.2f}%")

    # Recommendations
    print(f"\n{'RECOMMENDATIONS':=^60}")

    if improvement > 5:
        print("\n✓ STRONG IMPROVEMENT detected!")
        print("  → Interaction terms are beneficial for this problem")
        print("  → Consider using the enhanced model in production")
        print("  → Explore additional domain-specific interactions")

    elif improvement > 1:
        print("\n✓ MODERATE IMPROVEMENT detected")
        print("  → Interaction terms provide some benefit")
        print("  → Consider feature selection to reduce complexity")
        print("  → Monitor for overfitting with cross-validation")

    elif improvement > -1:
        print("\n→ MINIMAL CHANGE")
        print("  → Interactions do not add significant value")
        print("  → Consider using baseline model for simplicity")
        print("  → Explore other feature engineering approaches")

    else:
        print("\n⚠ PERFORMANCE DEGRADATION")
        print("  → Interaction terms may cause overfitting")
        print("  → Use baseline model or apply feature selection")
        print("  → Consider regularization techniques (L1/L2)")

    # Identify valuable interaction terms
    print(f"\n{'VALUABLE INTERACTION TERMS':=^60}")
    interaction_features_in_model = [
        f for f in interaction_feature_names
        if f in enhanced_importance['feature'].values
    ]

    if interaction_features_in_model:
        print(f"\nTop interaction features by importance:")
        for feat in interaction_features_in_model[:5]:
            if feat in enhanced_importance['feature'].values:
                imp = enhanced_importance[enhanced_importance['feature'] == feat]['importance'].values[0]
                print(f"  • {feat}: {imp:.4f}")
    else:
        print("\nNo interaction terms in top features - may need refinement")

    # ============================================================================
    # Final Summary
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"{'WORKFLOW COMPLETE':^80}")
    print(f"{'='*80}")

    print("\nGenerated Files:")
    print("  • results/plots/correlation_heatmap.png")
    print("  • results/plots/target_correlations.png")
    print("  • results/model_comparison.csv")
    print("  • results/models/baseline_model.joblib")
    print("  • results/models/enhanced_model.joblib")

    print("\nNext Steps:")
    print("  1. Review correlation heatmap for additional insights")
    print("  2. Experiment with different interaction types")
    print("  3. Try different model algorithms (RF, GBM, XGBoost)")
    print("  4. Perform cross-validation for robust evaluation")
    print("  5. Deploy the best-performing model")

    print("\n" + "="*80 + "\n")

    return {
        'baseline_score': baseline_score,
        'enhanced_score': enhanced_score,
        'improvement_pct': improvement,
        'n_original_features': len(original_features),
        'n_interaction_features': len(interaction_feature_names),
        'task_type': task_type
    }


if __name__ == "__main__":
    results = main()
    print(f"Optimization complete! Improvement: {results['improvement_pct']:+.2f}%")
