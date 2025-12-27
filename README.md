# MovieLens Recommendation System Competition

Modular framework for building and testing recommendation algorithms for the MovieLens competition.

## Project Structure

```
.
├── data_loader.py          # Data loading and preprocessing
├── recommenders/           # Recommendation algorithms
│   ├── __init__.py
│   └── base.py            # BaseRecommender interface + baseline
├── evaluation.py          # Weighted RMSE and evaluation metrics
├── submission.py          # Competition submission generation
├── main.py               # Main CLI orchestrator
└── README.md            # This file
```

## Quick Start

### 1. Train and Evaluate

```bash
# Train baseline and evaluate on validation split
python main.py --recommender baseline --evaluate

# Run 5-fold cross-validation
python main.py --recommender baseline --cross-validate

# Generate submission file
python main.py --recommender baseline --submit
```

### 2. Adding Your Own Recommender

Create a new file in `recommenders/` (e.g., `recommenders/my_algorithm.py`):

```python
from recommenders.base import BaseRecommender
import pandas as pd
import numpy as np

class MyRecommender(BaseRecommender):
    def __init__(self, hyperparameter1=0.1, hyperparameter2=10):
        super().__init__()
        self.param1 = hyperparameter1
        self.param2 = hyperparameter2
        # Initialize your model here
    
    def fit(self, train_df: pd.DataFrame) -> 'MyRecommender':
        """Train your model on train_df."""
        # train_df has columns: user_id, movie_id, rating
        
        # Your training code here
        
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a single user-movie pair."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Your prediction code here
        prediction = 3.0  # placeholder
        
        return prediction
```

Register in `main.py`:

```python
from recommenders.my_algorithm import MyRecommender

RECOMMENDER_REGISTRY = {
    'baseline': BaselineRecommender,
    'my_algo': MyRecommender,  # Add your recommender here
}
```

Then use it:

```bash
python main.py --recommender my_algo --evaluate
```

## Dataset Overview

- **train.csv**: 18M+ ratings (explicit + implicit feedback)
  - Explicit: ratings from 0.5 to 5.0
  - Implicit: null ratings (watched but not rated)
  
- **movies.csv**: 2K movies with titles and genres
- **tags.csv**: 960K+ user-generated tags
- **ratings_submission.csv**: 100K user-movie pairs to predict

## Evaluation Metric

**Weighted RMSE**: Movies with many ratings are down-weighted

```
W-RMSE = sqrt(Σ(w_i * (ŷ_i - y_i)²) / Σ(w_i))
where w_i = 1 / sqrt(rating_count_for_movie_i)
```

## Usage Examples

```bash
# Basic training
python main.py --recommender baseline

# Evaluate with custom validation size
python main.py --recommender baseline --evaluate --test-size 0.3

# Generate submission with custom output path
python main.py --recommender baseline --submit --output my_submission.csv

# Cross-validate with 10 folds
python main.py --recommender baseline --cross-validate --n-folds 10

# Specify data directory
python main.py --recommender baseline --data-dir /path/to/data --submit
```

## Modules

### data_loader.py
- `DataLoader`: Load and preprocess all CSV files
- Separate explicit/implicit feedback
- Generate train/test splits
- Extract genre features
- Compute data statistics

### recommenders/base.py
- `BaseRecommender`: Abstract interface for all algorithms
- `BaselineRecommender`: Simple global mean baseline

### evaluation.py
- `compute_weighted_rmse()`: Competition metric
- `evaluate_recommender()`: Full evaluation suite
- `cross_validate()`: K-fold cross-validation

### submission.py
- `SubmissionGenerator`: Create competition CSV files
- Automatic prediction clipping
- Validation checks

### main.py
- CLI orchestrator
- Recommender registry
- Train/evaluate/submit workflows

## Tips

1. **Start simple**: Test with the baseline first
2. **Validate locally**: Use `--evaluate` before submitting
3. **Handle missing data**: Some user-movie pairs may not be in training data
4. **Clip predictions**: Ensure outputs are in [0.5, 5.0] range
5. **Memory management**: Training data is large (50MB+), use efficient data structures

## Next Steps

Implement your recommendation algorithm:
- Collaborative Filtering (user-based, item-based)
- Matrix Factorization (SVD, ALS)
- Neural Networks (NCF, autoencoders)
- Hybrid models (combine collaborative + content-based)
- Leverage tags and genres for cold-start problems
