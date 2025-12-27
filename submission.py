"""
Submission file generation for the competition.

Handles loading submission template, generating predictions, and exporting
competition-ready CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from recommenders.base import BaseRecommender


class SubmissionGenerator:
    """Generates competition submission files."""
    
    def __init__(self, submission_template_path: str = "ratings_submission.csv"):
        """
        Initialize submission generator.
        
        Args:
            submission_template_path: Path to ratings_submission.csv
        """
        self.template_path = Path(submission_template_path)
        self.submission_df = None
    
    def load_template(self) -> pd.DataFrame:
        """
        Load submission template and parse user-movie pairs.
        
        Returns:
            DataFrame with columns: id, user_id, movie_id, prediction
        """
        print("Loading submission template...")
        self.submission_df = pd.read_csv(self.template_path)
        
        # Parse userId_movieId format
        split_ids = self.submission_df['id'].str.split('_', expand=True)
        self.submission_df['user_id'] = split_ids[0].astype('int32')
        self.submission_df['movie_id'] = split_ids[1].astype('int32')
        
        print(f"Loaded {len(self.submission_df):,} submission pairs")
        return self.submission_df
    
    def generate_predictions(
        self,
        recommender: BaseRecommender,
        clip: bool = True,
        min_rating: float = 0.5,
        max_rating: float = 5.0
    ) -> np.ndarray:
        """
        Generate predictions for all submission pairs.
        
        Args:
            recommender: Fitted recommender instance
            clip: Whether to clip predictions to valid range
            min_rating: Minimum valid rating
            max_rating: Maximum valid rating
        
        Returns:
            Array of predictions
        """
        if self.submission_df is None:
            self.load_template()
        
        if not recommender.is_fitted:
            raise RuntimeError("Recommender must be fitted before generating predictions")
        
        print(f"Generating predictions with {recommender.__class__.__name__}...")
        
        # Generate predictions
        predictions = recommender.predict_dataframe(
            self.submission_df[['user_id', 'movie_id']]
        )
        
        # Clip if requested
        if clip:
            predictions = np.clip(predictions, min_rating, max_rating)
            print(f"Clipped predictions to [{min_rating}, {max_rating}]")
        
        # Check for invalid predictions
        if np.any(np.isnan(predictions)):
            num_nan = np.isnan(predictions).sum()
            print(f"WARNING: {num_nan} NaN predictions found! Replacing with {(min_rating + max_rating) / 2}")
            predictions = np.nan_to_num(predictions, nan=(min_rating + max_rating) / 2)
        
        return predictions
    
    def create_submission_file(
        self,
        recommender: BaseRecommender,
        output_path: Optional[str] = None,
        clip: bool = True
    ) -> str:
        """
        Create complete submission CSV file.
        
        Args:
            recommender: Fitted recommender instance
            output_path: Path for output file (auto-generated if None)
            clip: Whether to clip predictions to valid range
        
        Returns:
            Path to generated submission file
        """
        if self.submission_df is None:
            self.load_template()
        
        # Generate predictions
        predictions = self.generate_predictions(recommender, clip=clip)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': self.submission_df['id'],
            'prediction': predictions
        })
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = recommender.__class__.__name__
            output_path = f"submission_{model_name}_{timestamp}.csv"
        
        # Save to CSV
        submission.to_csv(output_path, index=False)
        
        print(f"\n=== Submission Summary ===")
        print(f"Model: {recommender.__class__.__name__}")
        print(f"Predictions: {len(submission):,}")
        print(f"Mean prediction: {predictions.mean():.4f}")
        print(f"Std prediction: {predictions.std():.4f}")
        print(f"Min prediction: {predictions.min():.4f}")
        print(f"Max prediction: {predictions.max():.4f}")
        print(f"\nSaved to: {output_path}")
        
        return output_path
    
    def validate_submission(self, submission_path: str) -> bool:
        """
        Validate submission file format.
        
        Args:
            submission_path: Path to submission file
        
        Returns:
            True if valid, False otherwise
        """
        print(f"\nValidating submission file: {submission_path}")
        
        try:
            df = pd.read_csv(submission_path)
            
            # Check required columns
            if not all(col in df.columns for col in ['id', 'prediction']):
                print("ERROR: Missing required columns (id, prediction)")
                return False
            
            # Check row count
            expected_rows = 100000  # From competition description
            if len(df) != expected_rows:
                print(f"WARNING: Expected {expected_rows} rows, found {len(df)}")
            
            # Check for missing predictions
            if df['prediction'].isna().any():
                print(f"ERROR: Found {df['prediction'].isna().sum()} missing predictions")
                return False
            
            # Check prediction range
            min_pred = df['prediction'].min()
            max_pred = df['prediction'].max()
            if min_pred < 0.5 or max_pred > 5.0:
                print(f"WARNING: Predictions outside valid range [0.5, 5.0]: [{min_pred:.2f}, {max_pred:.2f}]")
            
            print("✓ Submission file is valid!")
            return True
            
        except Exception as e:
            print(f"ERROR validating submission: {e}")
            return False


def main():
    """Demo submission generation."""
    from data_loader import DataLoader
    from recommenders.base import BaselineRecommender
    
    # Load data and train baseline
    loader = DataLoader()
    train_df = loader.load_train_data(explicit_only=True)
    
    baseline = BaselineRecommender()
    baseline.fit(train_df)
    
    # Generate submission
    generator = SubmissionGenerator()
    submission_path = generator.create_submission_file(baseline)
    
    # Validate
    generator.validate_submission(submission_path)


if __name__ == "__main__":
    main()
