"""
Matrix Factorization with SVD and Biases.

This recommender uses Singular Value Decomposition to learn latent factors
for users and items, with additional bias terms for improved accuracy.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import Dict, Tuple
from recommenders.base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """
    Matrix Factorization using SVD with user and item biases.
    
    Prediction formula:
    r_ui = mu + b_u + b_i + q_i^T * p_u
    
    where:
    - mu: global mean rating
    - b_u: user bias
    - b_i: item bias  
    - q_i, p_u: latent factor vectors for item and user
    """
    
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        """
        Initialize SVD recommender.
        
        Args:
            n_factors: Number of latent factors (higher = more expressive but slower)
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.n_factors = n_factors
        self.random_state = random_state
        
        # Model parameters
        self.global_mean = None
        self.user_biases = {}
        self.item_biases = {}
        self.user_factors = None  # U matrix
        self.item_factors = None  # V^T matrix
        self.singular_values = None
        
        # Mappings
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}
    
    def fit(self, train_df: pd.DataFrame) -> 'SVDRecommender':
        """
        Train SVD model with biases.
        
        Args:
            train_df: DataFrame with columns [user_id, movie_id, rating]
        
        Returns:
            Self (for method chaining)
        """
        print(f"Training SVD with {self.n_factors} factors...")
        
        # Compute global mean
        self.global_mean = train_df['rating'].mean()
        
        # Create user and item mappings
        unique_users = train_df['user_id'].unique()
        unique_items = train_df['movie_id'].unique()
        
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        print(f"Users: {n_users:,}, Items: {n_items:,}")
        
        # Compute user biases (deviation from global mean)
        user_means = train_df.groupby('user_id')['rating'].mean()
        self.user_biases = (user_means - self.global_mean).to_dict()
        
        # Compute item biases (deviation from global mean)
        item_means = train_df.groupby('movie_id')['rating'].mean()
        self.item_biases = (item_means - self.global_mean).to_dict()
        
        # Create user-item rating matrix (sparse)
        user_indices = train_df['user_id'].map(self.user_id_to_idx).values
        item_indices = train_df['movie_id'].map(self.item_id_to_idx).values
        ratings = train_df['rating'].values
        
        # Subtract biases from ratings for matrix factorization
        ratings_normalized = ratings - self.global_mean
        for i, (uid, iid) in enumerate(zip(train_df['user_id'], train_df['movie_id'])):
            ratings_normalized[i] -= self.user_biases.get(uid, 0)
            ratings_normalized[i] -= self.item_biases.get(iid, 0)
        
        rating_matrix = csr_matrix(
            (ratings_normalized, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        print(f"Matrix sparsity: {1 - rating_matrix.nnz / (n_users * n_items):.4f}")
        print(f"Running SVD decomposition...")
        
        # Perform SVD (returns U, singular_values, Vt)
        # rating_matrix ≈ U @ diag(s) @ Vt
        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, s, Vt = svds(rating_matrix, k=k, random_state=self.random_state)
        
        # Store factors
        self.user_factors = U
        self.singular_values = s
        self.item_factors = Vt
        
        self.is_fitted = True
        print(f"✓ SVD training complete!")
        print(f"Global mean: {self.global_mean:.4f}")
        print(f"User biases: mean={np.mean(list(self.user_biases.values())):.4f}, "
              f"std={np.std(list(self.user_biases.values())):.4f}")
        print(f"Item biases: mean={np.mean(list(self.item_biases.values())):.4f}, "
              f"std={np.std(list(self.item_biases.values())):.4f}")
        
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
        
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Start with global mean + biases
        prediction = self.global_mean
        prediction += self.user_biases.get(user_id, 0)
        prediction += self.item_biases.get(movie_id, 0)
        
        # Add latent factor interaction if both user and item are known
        user_idx = self.user_id_to_idx.get(user_id)
        item_idx = self.item_id_to_idx.get(movie_id)
        
        if user_idx is not None and item_idx is not None:
            # Compute latent factor interaction: q_i^T @ diag(s) @ p_u
            user_vec = self.user_factors[user_idx, :]
            item_vec = self.item_factors[:, item_idx]
            prediction += np.dot(user_vec * self.singular_values, item_vec)
        
        return prediction
    
    def predict_batch(self, pairs) -> np.ndarray:
        """
        Efficient batch prediction.
        
        Args:
            pairs: List of (user_id, movie_id) tuples
        
        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        predictions = np.full(len(pairs), self.global_mean, dtype=np.float32)
        
        # Add biases
        for i, (user_id, movie_id) in enumerate(pairs):
            predictions[i] += self.user_biases.get(user_id, 0)
            predictions[i] += self.item_biases.get(movie_id, 0)
            
            # Add latent factors
            user_idx = self.user_id_to_idx.get(user_id)
            item_idx = self.item_id_to_idx.get(movie_id)
            
            if user_idx is not None and item_idx is not None:
                user_vec = self.user_factors[user_idx, :]
                item_vec = self.item_factors[:, item_idx]
                predictions[i] += np.dot(user_vec * self.singular_values, item_vec)
        
        return predictions


class SVDPlusPlusRecommender(SVDRecommender):
    """
    Enhanced SVD with implicit feedback.
    
    This is a placeholder for SVD++ which would incorporate implicit feedback
    (the fact that a user rated an item, regardless of the rating value).
    """
    
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        super().__init__(n_factors=n_factors, random_state=random_state)
        print("Note: This is currently equivalent to SVD. True SVD++ would require "
              "additional implicit feedback processing.")
