"""
Evaluation metrics for recommendation systems.

Implements the competition's Weighted RMSE metric and other evaluation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from recommenders.base import BaseRecommender


def compute_weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    movie_ids: np.ndarray,
    train_df: pd.DataFrame
) -> float:
    """
    Compute Weighted RMSE as defined in the competition.
    
    Formula: W-RMSE = sqrt(sum(w_i * (y_pred_i - y_true_i)^2) / sum(w_i))
    where w_i = 1 / sqrt(total_ratings_for_movie_i)
    
    Movies with many ratings are down-weighted to prevent disproportionate influence.
    
    Args:
        y_true: Array of true ratings
        y_pred: Array of predicted ratings
        movie_ids: Array of movie IDs corresponding to predictions
        train_df: Training DataFrame to compute movie rating counts
    
    Returns:
        Weighted RMSE value
    """
    # Compute rating counts per movie from training data
    movie_counts = train_df.groupby('movie_id').size().to_dict()
    
    # Compute weights for each prediction
    weights = np.array([1.0 / np.sqrt(movie_counts.get(movie_id, 1)) 
                       for movie_id in movie_ids])
    
    # Compute weighted squared errors
    squared_errors = (y_pred - y_true) ** 2
    weighted_squared_errors = weights * squared_errors
    
    # Compute weighted RMSE
    w_rmse = np.sqrt(weighted_squared_errors.sum() / weights.sum())
    
    return w_rmse


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute standard (unweighted) RMSE.
    
    Args:
        y_true: Array of true ratings
        y_pred: Array of predicted ratings
    
    Returns:
        RMSE value
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: Array of true ratings
        y_pred: Array of predicted ratings
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def evaluate_recommender(
    recommender: BaseRecommender,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    compute_all_metrics: bool = True
) -> Dict[str, float]:
    """
    Evaluate a recommender on test data with multiple metrics.
    
    Args:
        recommender: Fitted recommender instance
        test_df: Test DataFrame with columns [user_id, movie_id, rating]
        train_df: Training DataFrame (for computing weighted RMSE)
        compute_all_metrics: If True, compute RMSE, MAE in addition to W-RMSE
    
    Returns:
        Dictionary of metric names and values
    """
    if not recommender.is_fitted:
        raise RuntimeError("Recommender must be fitted before evaluation")
    
    print(f"Evaluating {recommender.__class__.__name__} on {len(test_df):,} samples...")
    
    # Get predictions
    y_pred = recommender.predict_dataframe(test_df)
    y_true = test_df['rating'].values
    movie_ids = test_df['movie_id'].values
    
    # Compute weighted RMSE (competition metric)
    w_rmse = compute_weighted_rmse(y_true, y_pred, movie_ids, train_df)
    
    metrics = {'weighted_rmse': w_rmse}
    
    if compute_all_metrics:
        metrics['rmse'] = compute_rmse(y_true, y_pred)
        metrics['mae'] = compute_mae(y_true, y_pred)
    
    return metrics


def print_evaluation_results(metrics: Dict[str, float]):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    print("\n=== Evaluation Results ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.6f}")


def cross_validate(
    recommender_class,
    train_df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    **recommender_kwargs
) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation on a recommender.
    
    Args:
        recommender_class: Recommender class (not instance)
        train_df: Full training DataFrame
        n_folds: Number of folds
        random_state: Random seed
        **recommender_kwargs: Hyperparameters to pass to recommender
    
    Returns:
        Dictionary with arrays of metrics for each fold
    """
    from sklearn.model_selection import KFold
    
    print(f"\n=== {n_folds}-Fold Cross-Validation ===")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_metrics = {'weighted_rmse': [], 'rmse': [], 'mae': []}
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df), 1):
        print(f"\nFold {fold_idx}/{n_folds}...")
        
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Train recommender
        recommender = recommender_class(**recommender_kwargs)
        recommender.fit(fold_train)
        
        # Evaluate
        metrics = evaluate_recommender(recommender, fold_val, fold_train)
        
        for metric_name, value in metrics.items():
            fold_metrics[metric_name].append(value)
        
        print(f"Fold {fold_idx} W-RMSE: {metrics['weighted_rmse']:.6f}")
    
    # Convert lists to arrays
    for key in fold_metrics:
        fold_metrics[key] = np.array(fold_metrics[key])
    
    # Print summary
    print(f"\n=== Cross-Validation Summary ===")
    for metric_name, values in fold_metrics.items():
        print(f"{metric_name.upper()}: {values.mean():.6f} (+/- {values.std():.6f})")
    
    return fold_metrics


def main():
    """Demo evaluation functionality."""
    from data_loader import DataLoader
    from recommenders.base import BaselineRecommender
    
    # Load data
    loader = DataLoader()
    train_df = loader.load_train_data(explicit_only=True)
    
    # Split train/test
    train_split, test_split = loader.get_train_test_split(test_size=0.2)
    
    # Train baseline
    baseline = BaselineRecommender()
    baseline.fit(train_split)
    
    # Evaluate
    metrics = evaluate_recommender(baseline, test_split, train_split)
    print_evaluation_results(metrics)


if __name__ == "__main__":
    main()
