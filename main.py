"""
Main orchestration script for training and evaluating recommendation systems.

Simple interface to train, evaluate, and generate submissions.

Usage:
    1. Choose your recommender in RECOMMENDER_REGISTRY
    2. Set the operations you want: EVALUATE, SUBMIT, CROSS_VALIDATE
    3. Run: python main.py
"""

from pathlib import Path
from typing import Type

from data_loader import DataLoader
from recommenders.base import BaseRecommender, BaselineRecommender
from recommenders.svd_recommender import SVDRecommender, SVDPlusPlusRecommender
from evaluation import evaluate_recommender, print_evaluation_results, cross_validate
from submission import SubmissionGenerator


# ============ CONFIGURATION ============

# Base data directory
BASE_DATA_DIR = "recsys-runi-2026/"

# Registry of available recommenders
RECOMMENDER_REGISTRY = {
    'baseline': BaselineRecommender,
    'svd': SVDRecommender,
    'svd++': SVDPlusPlusRecommender,
    # Add more recommenders here as you implement them:
    # 'user_mean': UserMeanRecommender,
    # 'item_mean': ItemMeanRecommender,
}

# Choose which recommender to use
ACTIVE_RECOMMENDER = 'svd'

# Choose what operations to run
EVALUATE = True          # Evaluate on validation split
SUBMIT = False           # Generate submission file
CROSS_VALIDATE = False   # Run k-fold cross-validation

# Parameters
TEST_SIZE = 0.2          # Validation split size
N_FOLDS = 5              # Number of folds for cross-validation
OUTPUT_PATH = None       # Submission output path (None = auto-generate)

# Recommender hyperparameters (if needed)
RECOMMENDER_KWARGS = {
    'n_factors': 50,      # Number of latent factors (higher = more complex)
    'random_state': 42
}

# =======================================


def get_recommender_class(name: str) -> Type[BaseRecommender]:
    """
    Get recommender class by name from registry.
    
    Args:
        name: Name of the recommender
    
    Returns:
        Recommender class
    
    Raises:
        ValueError: If recommender name not found
    """
    if name not in RECOMMENDER_REGISTRY:
        available = ', '.join(RECOMMENDER_REGISTRY.keys())
        raise ValueError(f"Unknown recommender '{name}'. Available: {available}")
    
    return RECOMMENDER_REGISTRY[name]


def train_recommender(
    recommender_name: str,
    data_dir: str = ".",
    **recommender_kwargs
) -> BaseRecommender:
    """
    Train a recommender on full training data.
    
    Args:
        recommender_name: Name of recommender from registry
        data_dir: Path to data directory
        **recommender_kwargs: Hyperparameters for recommender
    
    Returns:
        Fitted recommender instance
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {recommender_name}")
    print(f"{'='*60}\n")
    
    # Load data
    loader = DataLoader(data_dir)
    train_df = loader.load_train_data(explicit_only=True)
    
    # Print data statistics
    stats = loader.get_data_statistics()
    print("\n--- Training Data Statistics ---")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    # Initialize and train recommender
    print(f"\n--- Training {recommender_name} ---")
    RecommenderClass = get_recommender_class(recommender_name)
    recommender = RecommenderClass(**recommender_kwargs)
    recommender.fit(train_df)
    
    print(f"\n✓ Training complete!")
    return recommender


def evaluate_on_validation(
    recommender: BaseRecommender,
    data_dir: str = ".",
    test_size: float = 0.2
):
    """
    Evaluate recommender on validation split.
    
    Args:
        recommender: Fitted recommender instance
        data_dir: Path to data directory
        test_size: Fraction of data for validation
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {recommender.__class__.__name__}")
    print(f"{'='*60}\n")
    
    # Load and split data
    loader = DataLoader(data_dir)
    train_df = loader.load_train_data(explicit_only=True)
    train_split, val_split = loader.get_train_test_split(test_size=test_size)
    
    # Retrain on train split for fair evaluation
    print("Re-training on training split...")
    recommender.fit(train_split)
    
    # Evaluate
    metrics = evaluate_recommender(recommender, val_split, train_split)
    print_evaluation_results(metrics)
    
    return metrics


def generate_submission(
    recommender: BaseRecommender,
    data_dir: str = ".",
    output_path: str = None
):
    """
    Generate competition submission file.
    
    Args:
        recommender: Fitted recommender instance
        data_dir: Path to data directory
        output_path: Output file path (auto-generated if None)
    """
    print(f"\n{'='*60}")
    print(f"SUBMISSION: {recommender.__class__.__name__}")
    print(f"{'='*60}\n")
    
    generator = SubmissionGenerator(
        submission_template_path=Path(data_dir) / "ratings_submission.csv"
    )
    
    submission_path = generator.create_submission_file(
        recommender,
        output_path=output_path,
        clip=True
    )
    
    # Validate submission
    generator.validate_submission(submission_path)
    
    return submission_path


def run_cross_validation(
    recommender_name: str,
    data_dir: str = ".",
    n_folds: int = 5,
    **recommender_kwargs
):
    """
    Run k-fold cross-validation.
    
    Args:
        recommender_name: Name of recommender from registry
        data_dir: Path to data directory
        n_folds: Number of folds
        **recommender_kwargs: Hyperparameters for recommender
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION: {recommender_name}")
    print(f"{'='*60}\n")
    
    # Load data
    loader = DataLoader(data_dir)
    train_df = loader.load_train_data(explicit_only=True)
    
    # Run cross-validation
    RecommenderClass = get_recommender_class(recommender_name)
    fold_metrics = cross_validate(
        RecommenderClass,
        train_df,
        n_folds=n_folds,
        **recommender_kwargs
    )
    
    return fold_metrics


def main():
    """Main entry point."""
    print(f"\n{'='*60}")
    print(f"MovieLens Recommendation System")
    print(f"{'='*60}")
    print(f"\nActive Recommender: {ACTIVE_RECOMMENDER}")
    print(f"Data Directory: {BASE_DATA_DIR}")
    print(f"Operations: Evaluate={EVALUATE}, Submit={SUBMIT}, CV={CROSS_VALIDATE}")
    print(f"{'='*60}\n")
    
    # Train recommender on full data
    recommender = train_recommender(
        ACTIVE_RECOMMENDER, 
        BASE_DATA_DIR,
        **RECOMMENDER_KWARGS
    )
    
    # Run requested operations
    if CROSS_VALIDATE:
        run_cross_validation(
            ACTIVE_RECOMMENDER,
            BASE_DATA_DIR,
            n_folds=N_FOLDS,
            **RECOMMENDER_KWARGS
        )
    
    if EVALUATE:
        evaluate_on_validation(recommender, BASE_DATA_DIR, TEST_SIZE)
    
    if SUBMIT:
        generate_submission(recommender, BASE_DATA_DIR, OUTPUT_PATH)
    
    # If no operations specified, just show training completed
    if not (EVALUATE or SUBMIT or CROSS_VALIDATE):
        print("\n✓ Training complete!")
        print(f"\nTo run operations, edit main.py and set:")
        print(f"  EVALUATE = True          # Evaluate on validation split")
        print(f"  SUBMIT = True            # Generate submission file")
        print(f"  CROSS_VALIDATE = True    # Run k-fold cross-validation")


if __name__ == "__main__":
    main()
