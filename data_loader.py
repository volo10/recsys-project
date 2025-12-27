"""
Data loading utilities for the MovieLens recommendation competition.

This module handles loading and preprocessing of all competition data files,
including train.csv, movies.csv, tags.csv, and ratings_submission.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


class DataLoader:
    """Loads and preprocesses competition data files."""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize DataLoader with path to data directory.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.movies_df = None
        self.tags_df = None
        self.submission_template_df = None
        
    def load_train_data(self, explicit_only: bool = True) -> pd.DataFrame:
        """
        Load training data from train.csv.
        
        Args:
            explicit_only: If True, filter out implicit feedback (null ratings).
                          If False, keep all interactions.
        
        Returns:
            DataFrame with columns: user_id, movie_id, rating
        """
        print("Loading training data...")
        self.train_df = pd.read_csv(
            self.data_dir / "train.csv",
            dtype={'user_id': 'int32', 'movie_id': 'int32', 'rating': 'float32'}
        )
        
        print(f"Loaded {len(self.train_df):,} interactions")
        
        if explicit_only:
            # Filter out implicit feedback (null ratings)
            self.train_df = self.train_df.dropna(subset=['rating'])
            print(f"Filtered to {len(self.train_df):,} explicit ratings")
        else:
            print(f"Kept {self.train_df['rating'].isna().sum():,} implicit feedback entries")
        
        return self.train_df
    
    def load_movies_metadata(self) -> pd.DataFrame:
        """
        Load movie metadata from movies.csv.
        
        Returns:
            DataFrame with columns: movie_id, title, genres
        """
        print("Loading movies metadata...")
        self.movies_df = pd.read_csv(
            self.data_dir / "movies.csv",
            dtype={'movie_id': 'int32'}
        )
        print(f"Loaded {len(self.movies_df):,} movies")
        return self.movies_df
    
    def load_tags(self) -> pd.DataFrame:
        """
        Load user-generated tags from tags.csv.
        
        Returns:
            DataFrame with columns: user_id, movie_id, tag
        """
        print("Loading tags...")
        self.tags_df = pd.read_csv(
            self.data_dir / "tags.csv",
            dtype={'user_id': 'int32', 'movie_id': 'int32'}
        )
        print(f"Loaded {len(self.tags_df):,} tags")
        return self.tags_df
    
    def load_submission_template(self) -> pd.DataFrame:
        """
        Load submission template from ratings_submission.csv.
        
        Returns:
            DataFrame with columns: id (userId_movieId), prediction
        """
        print("Loading submission template...")
        self.submission_template_df = pd.read_csv(
            self.data_dir / "ratings_submission.csv"
        )
        
        # Parse userId_movieId format
        split_ids = self.submission_template_df['id'].str.split('_', expand=True)
        self.submission_template_df['user_id'] = split_ids[0].astype('int32')
        self.submission_template_df['movie_id'] = split_ids[1].astype('int32')
        
        print(f"Loaded {len(self.submission_template_df):,} submission pairs")
        return self.submission_template_df
    
    def get_genre_features(self) -> Dict[int, np.ndarray]:
        """
        Extract genre features for movies as binary vectors.
        
        Returns:
            Dictionary mapping movie_id to binary genre vector
        """
        if self.movies_df is None:
            self.load_movies_metadata()
        
        # Get all unique genres
        all_genres = set()
        for genres_str in self.movies_df['genres'].dropna():
            all_genres.update(genres_str.split('|'))
        all_genres = sorted(list(all_genres))
        
        # Create binary genre vectors
        genre_features = {}
        for _, row in self.movies_df.iterrows():
            movie_id = row['movie_id']
            genres = row['genres'].split('|') if pd.notna(row['genres']) else []
            genre_vector = np.array([1 if g in genres else 0 for g in all_genres], dtype=np.float32)
            genre_features[movie_id] = genre_vector
        
        print(f"Extracted genre features with {len(all_genres)} unique genres")
        return genre_features
    
    def get_data_statistics(self) -> Dict[str, any]:
        """
        Compute basic statistics about the training data.
        
        Returns:
            Dictionary with statistics (num_users, num_movies, sparsity, etc.)
        """
        if self.train_df is None:
            self.load_train_data()
        
        stats = {
            'num_interactions': len(self.train_df),
            'num_users': self.train_df['user_id'].nunique(),
            'num_movies': self.train_df['movie_id'].nunique(),
            'min_rating': self.train_df['rating'].min(),
            'max_rating': self.train_df['rating'].max(),
            'mean_rating': self.train_df['rating'].mean(),
            'median_rating': self.train_df['rating'].median(),
        }
        
        # Compute sparsity
        total_possible = stats['num_users'] * stats['num_movies']
        stats['sparsity'] = 1 - (stats['num_interactions'] / total_possible)
        
        return stats
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets.
        
        Args:
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_df, val_df)
        """
        if self.train_df is None:
            self.load_train_data()
        
        # Shuffle and split
        train_df_shuffled = self.train_df.sample(frac=1, random_state=random_state)
        split_idx = int(len(train_df_shuffled) * (1 - test_size))
        
        train_split = train_df_shuffled.iloc[:split_idx].copy()
        val_split = train_df_shuffled.iloc[split_idx:].copy()
        
        print(f"Split: {len(train_split):,} train, {len(val_split):,} validation")
        return train_split, val_split


def main():
    """Demo usage of DataLoader."""
    loader = DataLoader()
    
    # Load all data
    train_df = loader.load_train_data(explicit_only=True)
    movies_df = loader.load_movies_metadata()
    tags_df = loader.load_tags()
    submission_df = loader.load_submission_template()
    
    # Print statistics
    print("\n=== Data Statistics ===")
    stats = loader.get_data_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    # Show sample data
    print("\n=== Sample Training Data ===")
    print(train_df.head())
    
    print("\n=== Sample Movies ===")
    print(movies_df.head())
    
    print("\n=== Sample Submission Template ===")
    print(submission_df.head())


if __name__ == "__main__":
    main()
