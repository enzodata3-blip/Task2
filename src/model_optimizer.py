"""
Model Optimizer Module
Trains and compares models with and without interaction terms
Human-in-the-loop optimization approach
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import warnings
import joblib
warnings.filterwarnings('ignore')


class ModelOptimizer:
    """
    Train and optimize machine learning models with human-in-the-loop approach
    Compares baseline vs. interaction-enhanced models
    """

    def __init__(self,
                 task_type: str = 'classification',
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize model optimizer

        Parameters:
        -----------
        task_type : str
            'classification' or 'regression'
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        """
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.baseline_model = None
        self.enhanced_model = None

        self.baseline_results = {}
        self.enhanced_results = {}

        self.scaler = StandardScaler()

    def prepare_data(self,
                     data: pd.DataFrame,
                     target_column: str,
                     scale_features: bool = True) -> Tuple:
        """
        Prepare data for training

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Target variable name
        scale_features : bool
            Whether to standardize features

        Returns:
        --------
        Tuple
            (X_train, X_test, y_train, y_test)
        """
        print(f"\n{'='*60}")
        print(f"Preparing Data for {self.task_type.upper()}")
        print(f"{'='*60}")

        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))

        # Select only numeric features
        X = X.select_dtypes(include=[np.number])

        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Target distribution:")
        print(y.value_counts() if self.task_type == 'classification' else y.describe())

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )

        # Scale features
        if scale_features:
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns,
                index=self.X_test.index
            )
            print("\n✓ Features scaled using StandardScaler")

        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_baseline_model(self,
                              model_type: str = 'auto',
                              **model_params) -> object:
        """
        Train baseline model (without interaction features)

        Parameters:
        -----------
        model_type : str
            Model type ('logistic', 'random_forest', 'gradient_boosting', 'auto')
        **model_params
            Additional parameters for the model

        Returns:
        --------
        object
            Trained model
        """
        print(f"\n{'='*60}")
        print(f"Training BASELINE Model")
        print(f"{'='*60}")

        # Select model based on task type
        if model_type == 'auto':
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    **model_params
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    **model_params
                )
        elif model_type == 'logistic':
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                **model_params
            )
        elif model_type == 'random_forest':
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )
        elif model_type == 'gradient_boosting':
            if self.task_type == 'classification':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"Model: {model.__class__.__name__}")

        # Train model
        self.baseline_model = model.fit(self.X_train, self.y_train)

        # Evaluate
        self.baseline_results = self._evaluate_model(
            self.baseline_model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            model_name="Baseline"
        )

        return self.baseline_model

    def train_enhanced_model(self,
                              baseline_features: List[str],
                              interaction_features: List[str],
                              model_type: str = 'auto',
                              **model_params) -> object:
        """
        Train enhanced model with interaction features

        Parameters:
        -----------
        baseline_features : List[str]
            Original features
        interaction_features : List[str]
            Interaction features to add
        model_type : str
            Model type ('logistic', 'random_forest', 'gradient_boosting', 'auto')
        **model_params
            Additional parameters for the model

        Returns:
        --------
        object
            Trained enhanced model
        """
        print(f"\n{'='*60}")
        print(f"Training ENHANCED Model (with Interactions)")
        print(f"{'='*60}")

        # Select features for enhanced model
        all_features = baseline_features + interaction_features
        available_features = [f for f in all_features if f in self.X_train.columns]

        X_train_enhanced = self.X_train[available_features]
        X_test_enhanced = self.X_test[available_features]

        print(f"Baseline features: {len(baseline_features)}")
        print(f"Interaction features: {len(interaction_features)}")
        print(f"Total features: {len(available_features)}")

        # Select model (same as baseline for fair comparison)
        if model_type == 'auto':
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    **model_params
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    **model_params
                )
        elif model_type == 'logistic':
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                **model_params
            )
        elif model_type == 'random_forest':
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )
        elif model_type == 'gradient_boosting':
            if self.task_type == 'classification':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    **model_params
                )

        print(f"Model: {model.__class__.__name__}")

        # Train model
        self.enhanced_model = model.fit(X_train_enhanced, self.y_train)

        # Evaluate
        self.enhanced_results = self._evaluate_model(
            self.enhanced_model,
            X_train_enhanced,
            X_test_enhanced,
            self.y_train,
            self.y_test,
            model_name="Enhanced"
        )

        return self.enhanced_model

    def _evaluate_model(self,
                        model: object,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        model_name: str = "Model") -> Dict:
        """
        Evaluate model performance

        Parameters:
        -----------
        model : object
            Trained model
        X_train, X_test : pd.DataFrame
            Training and test features
        y_train, y_test : pd.Series
            Training and test targets
        model_name : str
            Name for reporting

        Returns:
        --------
        Dict
            Evaluation metrics
        """
        print(f"\nEvaluating {model_name} Model...")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results = {
            'model_name': model_name,
            'n_features': X_train.shape[1]
        }

        if self.task_type == 'classification':
            # Classification metrics
            results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            results['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            results['test_precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            results['test_recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            results['test_f1'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

            # ROC AUC (if binary classification)
            if len(np.unique(y_test)) == 2:
                try:
                    y_test_proba = model.predict_proba(X_test)[:, 1]
                    results['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
                except:
                    results['test_roc_auc'] = None

            print(f"\n{model_name} Classification Results:")
            print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
            print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
            print(f"  Test Precision: {results['test_precision']:.4f}")
            print(f"  Test Recall:    {results['test_recall']:.4f}")
            print(f"  Test F1-Score:  {results['test_f1']:.4f}")
            if results.get('test_roc_auc'):
                print(f"  Test ROC-AUC:   {results['test_roc_auc']:.4f}")

        else:
            # Regression metrics
            results['train_r2'] = r2_score(y_train, y_train_pred)
            results['test_r2'] = r2_score(y_test, y_test_pred)
            results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
            results['test_mae'] = mean_absolute_error(y_test, y_test_pred)

            print(f"\n{model_name} Regression Results:")
            print(f"  Train R²:  {results['train_r2']:.4f}")
            print(f"  Test R²:   {results['test_r2']:.4f}")
            print(f"  Test RMSE: {results['test_rmse']:.4f}")
            print(f"  Test MAE:  {results['test_mae']:.4f}")

        return results

    def compare_models(self) -> pd.DataFrame:
        """
        Compare baseline vs. enhanced model performance

        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON")
        print(f"{'='*60}")

        comparison_df = pd.DataFrame([self.baseline_results, self.enhanced_results])

        # Calculate improvements
        if self.task_type == 'classification':
            key_metric = 'test_accuracy'
        else:
            key_metric = 'test_r2'

        baseline_score = self.baseline_results.get(key_metric, 0)
        enhanced_score = self.enhanced_results.get(key_metric, 0)
        improvement = ((enhanced_score - baseline_score) / baseline_score) * 100

        print(f"\n{'Model':<20} {'Features':<12} {key_metric:<15}")
        print("-" * 60)
        print(f"{self.baseline_results['model_name']:<20} {self.baseline_results['n_features']:<12} {baseline_score:.4f}")
        print(f"{self.enhanced_results['model_name']:<20} {self.enhanced_results['n_features']:<12} {enhanced_score:.4f}")
        print("-" * 60)
        print(f"\n{'IMPROVEMENT:':<35} {improvement:+.2f}%")

        if improvement > 0:
            print("\n✓ Enhanced model shows IMPROVEMENT!")
        elif improvement < -1:
            print("\n⚠ Enhanced model shows DEGRADATION - consider feature selection")
        else:
            print("\n→ Models perform similarly - interactions may not add value")

        print(f"{'='*60}\n")

        return comparison_df

    def get_feature_importance(self,
                                model: object,
                                feature_names: List[str],
                                top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from model

        Parameters:
        -----------
        model : object
            Trained model
        feature_names : List[str]
            Feature names
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            print("⚠ Model does not support feature importance")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 60)
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:<40s}: {row['importance']:.4f}")

        return importance_df

    def save_models(self, baseline_path: str, enhanced_path: str):
        """Save trained models"""
        if self.baseline_model:
            joblib.dump(self.baseline_model, baseline_path)
            print(f"✓ Baseline model saved: {baseline_path}")

        if self.enhanced_model:
            joblib.dump(self.enhanced_model, enhanced_path)
            print(f"✓ Enhanced model saved: {enhanced_path}")


if __name__ == "__main__":
    print("Model Optimizer Module - Ready for import")
