"""
Example: Adding a User-Mean Recommender

This file demonstrates how to add a new recommendation algorithm to the framework.
Copy this template to create your own recommender.
"""

from recommenders.base import BaseRecommender
import pandas as pd
import numpy as np
from typing import Dict


class UserMeanRecommender(BaseRecommender):
    """
    Predicts using per-user mean ratings.
    Falls back to global mean for unknown users.
    
    This is a simple example showing how to implement a new recommender.
    """
    
    def __init__(self):
        """Initialize the user-mean recommender."""
        super().__init__()
        self.user_means: Dict[int, float] = {}
        self.global_mean: float = None
    
    def fit(self, train_df: pd.DataFrame) -> 'UserMeanRecommender':
        """
        Compute mean rating for each user.
        
        Args:
            train_df: DataFrame with columns [user_id, movie_id, rating]
        
        Returns:
            Self (for method chaining)
        """
        print("Computing user means...")
        
        # Compute per-user mean
        self.user_means = train_df.groupby('user_id')['rating'].mean().to_dict()
        
        # Compute global mean as fallback
        self.global_mean = train_df['rating'].mean()
        
        self.is_fitted = True
        print(f"Fitted on {len(self.user_means)} users, global mean = {self.global_mean:.4f}")
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using user mean or global mean.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier (not used in this simple algorithm)
        
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Use user mean if available, otherwise global mean
        return self.user_means.get(user_id, self.global_mean)
    
    def predict_batch(self, pairs):
        """Efficient batch prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        predictions = np.array([
            self.user_means.get(user_id, self.global_mean)
            for user_id, _ in pairs
        ])
        return predictions


# To use this recommender:
# 1. Save this file as recommenders/user_mean.py
# 2. Add to main.py:
#    from recommenders.user_mean import UserMeanRecommender
#    RECOMMENDER_REGISTRY['user_mean'] = UserMeanRecommender
# 3. Run: python main.py --recommender user_mean --evaluate
