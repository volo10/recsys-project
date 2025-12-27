"""
Base interface for recommendation algorithms.

All recommendation algorithms must inherit from BaseRecommender and implement
the abstract methods fit(), predict(), and predict_batch().
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation algorithms.
    
    This interface ensures all recommenders have consistent API for training
    and prediction, making it easy to swap algorithms.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize recommender with hyperparameters.
        
        Args:
            **kwargs: Algorithm-specific hyperparameters
        """
        self.is_fitted = False
        self.kwargs = kwargs
    
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> 'BaseRecommender':
        """
        Train the recommendation model on training data.
        
        Args:
            train_df: DataFrame with columns [user_id, movie_id, rating]
        
        Returns:
            Self (for method chaining)
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a single user-movie pair.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
        
        Returns:
            Predicted rating (typically in range [0.5, 5.0])
        """
        pass
    
    def predict_batch(self, pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Predict ratings for multiple user-movie pairs.
        
        Default implementation calls predict() for each pair.
        Subclasses can override for more efficient batch prediction.
        
        Args:
            pairs: List of (user_id, movie_id) tuples
        
        Returns:
            Array of predicted ratings
        """
        predictions = np.array([self.predict(user_id, movie_id) for user_id, movie_id in pairs])
        return predictions
    
    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for user-movie pairs in a DataFrame.
        
        Args:
            df: DataFrame with columns [user_id, movie_id]
        
        Returns:
            Array of predicted ratings
        """
        pairs = list(zip(df['user_id'], df['movie_id']))
        return self.predict_batch(pairs)
    
    def clip_predictions(self, predictions: np.ndarray, min_val: float = 0.5, max_val: float = 5.0) -> np.ndarray:
        """
        Clip predictions to valid rating range.
        
        Args:
            predictions: Array of predicted ratings
            min_val: Minimum valid rating
            max_val: Maximum valid rating
        
        Returns:
            Clipped predictions
        """
        return np.clip(predictions, min_val, max_val)
    
    def __repr__(self) -> str:
        """String representation of the recommender."""
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


class BaselineRecommender(BaseRecommender):
    """
    Simple baseline recommender that predicts global mean rating.
    
    This serves as a minimal working example and baseline for comparison.
    """
    
    def __init__(self):
        """Initialize baseline recommender."""
        super().__init__()
        self.global_mean = None
    
    def fit(self, train_df: pd.DataFrame) -> 'BaselineRecommender':
        """
        Compute global mean rating from training data.
        
        Args:
            train_df: DataFrame with columns [user_id, movie_id, rating]
        
        Returns:
            Self (for method chaining)
        """
        self.global_mean = train_df['rating'].mean()
        self.is_fitted = True
        print(f"Baseline fitted with global mean = {self.global_mean:.4f}")
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict global mean for any user-movie pair.
        
        Args:
            user_id: User identifier (ignored)
            movie_id: Movie identifier (ignored)
        
        Returns:
            Global mean rating
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.global_mean
    
    def predict_batch(self, pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Efficiently predict global mean for all pairs.
        
        Args:
            pairs: List of (user_id, movie_id) tuples
        
        Returns:
            Array filled with global mean
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return np.full(len(pairs), self.global_mean, dtype=np.float32)
