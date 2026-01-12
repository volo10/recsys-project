#!/usr/bin/env python3
"""
Wide & Deep Recommender V17 - SIMPLIFIED ARCHITECTURE
Target: Test RMSE < 0.80

Key Changes - Reduce Complexity:
1. STRATIFIED SPLIT: Same as V12 (stratified by user), but larger val (25%)
2. SIMPLER TARGET ENCODING: Standard smoothing, no adaptive tricks
3. SIMPLER FEATURES: Remove complex continuous stats, keep genres/year/buckets
4. SHALLOWER MODEL: [256, 128] deep tower (removed 3rd layer)
5. MODERATE REGULARIZATION: weight_decay=5e-3, dropout=0.35
6. LARGER ENSEMBLE: 5 models averaged
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import gc
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
@dataclass
class ConfigV17:
    # Embeddings
    embedding_dim: int = 32
    embedding_dropout: float = 0.25  # V17: Moderate dropout
    
    # Deep tower - V17: Shallower [256, 128]
    deep_layers: Tuple[int, ...] = (256, 128)  # V17: Removed 3rd layer
    deep_dropout: float = 0.35  # V17: Moderate dropout
    
    # Wide tower
    wide_hidden_dim: int = 32
    wide_dropout: float = 0.35  # V17: Moderate dropout
    
    # Biases
    bias_dropout: float = 0.1
    
    # Optimizer - V17: Moderate weight decay
    lr: float = 5e-4
    weight_decay: float = 5e-3  # V17: Moderate weight decay (0.005)
    
    # Scheduler
    warmup_epochs: int = 2
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1e-5
    
    # Training
    n_epochs: int = 25
    batch_size: int = 2048
    patience: int = 6
    grad_clip: float = 0.9
    
    # Features - V17: Standard smoothing (no adaptive)
    smoothing_strength: float = 80.0  # Standard Bayesian smoothing
    max_tags: int = 80
    
    # Bucketing thresholds
    user_activity_thresholds: Tuple[int, ...] = (10, 40, 100)
    item_popularity_thresholds: Tuple[int, ...] = (30, 150, 400)
    
    # Ensemble
    seeds: Tuple[int, ...] = (42, 123, 456, 789, 2025)
    n_seeds_to_use: int = 5
    
    # V17: Larger validation set for better estimate
    val_ratio: float = 0.25  # 25% validation
    final_val_ratio: float = 0.05  # For retrain phase


CONFIG = ConfigV17()


# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("✓ Using CPU")

print(f"PyTorch: {torch.__version__}")

# === DATA PATHS ===
DATA_DIR = "../project/recsys-runi-2026"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "recsys-runi-2026"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "."
print(f"Data directory: {DATA_DIR}")


def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === PREPROCESSING ===
def normalize_tag(tag: str) -> str:
    if pd.isna(tag):
        return ""
    tag = str(tag).lower()
    tag = re.sub(r'[^a-z0-9\s]', '', tag)
    tag = re.sub(r'\s+', ' ', tag)
    return tag.strip()


def extract_movie_year(title: str) -> Tuple[str, Optional[int]]:
    if pd.isna(title):
        return "", None
    match = re.search(r'\((\d{4})(?:-\d{4})?\)\s*$', title)
    if match:
        return title.strip(), int(match.group(1))
    return title.strip(), None


# === V17: USER COLD-START SPLIT (No Timestamps Available) ===
def user_coldstart_split(df: pd.DataFrame, val_user_ratio: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    V17: Split by USERS - hold out val_user_ratio% of users entirely.
    This prevents leakage by ensuring validation users are completely unseen during training.
    Simulates cold-start scenario which better matches test distribution.
    """
    print(f"\n[USER COLD-START SPLIT] Holding out {val_user_ratio*100:.0f}% of users...")
    
    np.random.seed(random_state)
    
    # Get all unique users
    all_users = df['user_id'].unique()
    n_users = len(all_users)
    n_val_users = int(n_users * val_user_ratio)
    
    # Shuffle and split users
    np.random.shuffle(all_users)
    val_users = set(all_users[:n_val_users])
    train_users = set(all_users[n_val_users:])
    
    # Split dataframe by users
    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['user_id'].isin(val_users)].reset_index(drop=True)
    
    print(f"  Train users: {len(train_users):,}, Train ratings: {len(train_df):,}")
    print(f"  Val users: {len(val_users):,}, Val ratings: {len(val_df):,}")
    print(f"  Avg ratings per train user: {len(train_df)/len(train_users):.1f}")
    print(f"  Avg ratings per val user: {len(val_df)/len(val_users):.1f}")
    
    return train_df, val_df


def stratified_rating_split(df: pd.DataFrame, val_ratio: float = 0.05, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split for retrain phase - take val_ratio of each user's ratings.
    """
    np.random.seed(random_state)
    train_indices, val_indices = [], []
    
    for user_id, group in df.groupby('user_id'):
        indices = group.index.tolist()
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        train_indices.extend(indices[n_val:])
        val_indices.extend(indices[:n_val])
    
    return df.loc[train_indices].reset_index(drop=True), df.loc[val_indices].reset_index(drop=True)


# === FEATURE STORE V17 (Simplified) ===
class FeatureStoreV17:
    """
    V17: Simplified feature store - no adaptive smoothing, no complex stats.
    Target encoding computed ONLY on training data to prevent leakage.
    """
    def __init__(self):
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.n_users = 0
        self.n_items = 0
        self.genre_list = []
        self.genre_features = {}
        self.movie_years = {}
        self.movie_year_bucket = {}
        
        # V17: Standard target encoding (computed on train only)
        self.user_mean_rating = {}
        self.item_mean_rating = {}
        self.user_rating_count = {}
        self.item_rating_count = {}
        self.user_activity_bucket = {}
        self.item_popularity_bucket = {}
        
        self.global_mean = 3.5
    
    def build_basic(self, train_df, submission_df, movies_df):
        """Build ID mappings and movie metadata."""
        print("\n" + "="*60)
        print("BUILDING FEATURE STORE V17 (Temporal Generalization)")
        print("="*60)
        
        # ID mappings (include all users/items from train + submission)
        print("[1/3] Building ID mappings...")
        all_users = set(train_df['user_id'].unique()) | set(submission_df['user_id'].unique())
        all_items = set(train_df['movie_id'].unique()) | set(submission_df['movie_id'].unique()) | set(movies_df['movie_id'].unique())
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(all_users))}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(sorted(all_items))}
        self.n_users = len(self.user_id_to_idx)
        self.n_items = len(self.item_id_to_idx)
        print(f"  Users: {self.n_users:,}, Items: {self.n_items:,}")
        
        # Movie metadata
        print("[2/3] Extracting movie metadata...")
        for _, row in movies_df.iterrows():
            mid = row['movie_id']
            _, year = extract_movie_year(row['title'])
            self.movie_years[mid] = year
            if year:
                if year < 1970: self.movie_year_bucket[mid] = 0
                elif year >= 2010: self.movie_year_bucket[mid] = 5
                else: self.movie_year_bucket[mid] = min(5, (year - 1970) // 10 + 1)
            else:
                self.movie_year_bucket[mid] = 3
        
        # Genre features
        print("[3/3] Building genre features...")
        all_genres = set()
        for g in movies_df['genres'].dropna():
            if g != '(no genres listed)':
                all_genres.update(g.split('|'))
        self.genre_list = sorted(list(all_genres))
        
        for _, row in movies_df.iterrows():
            mid = row['movie_id']
            genres = row['genres'].split('|') if pd.notna(row['genres']) and row['genres'] != '(no genres listed)' else []
            self.genre_features[mid] = np.array([1.0 if g in genres else 0.0 for g in self.genre_list], dtype=np.float32)
        print(f"  Genres: {len(self.genre_list)}")
        print("="*60)
    
    def build_target_encoding(self, train_df: pd.DataFrame):
        """
        V17: Standard Bayesian target encoding - computed ONLY on training data.
        No adaptive smoothing - fixed smoothing strength.
        """
        print(f"\nBuilding TARGET ENCODING (V17 - train data only)...")
        print(f"  Smoothing strength: {CONFIG.smoothing_strength}")
        
        explicit = train_df[train_df['rating'].notna()]
        self.global_mean = explicit['rating'].mean()
        
        # User stats with standard smoothing
        user_stats = explicit.groupby('user_id')['rating'].agg(['mean', 'count'])
        for uid, row in user_stats.iterrows():
            n, avg = row['count'], row['mean']
            # Standard Bayesian smoothing: (n * avg + m * global_mean) / (n + m)
            smoothed = (n * avg + CONFIG.smoothing_strength * self.global_mean) / (n + CONFIG.smoothing_strength)
            self.user_mean_rating[uid] = smoothed
            self.user_rating_count[uid] = n
            
            thresholds = CONFIG.user_activity_thresholds
            if n < thresholds[0]: self.user_activity_bucket[uid] = 0
            elif n < thresholds[1]: self.user_activity_bucket[uid] = 1
            elif n < thresholds[2]: self.user_activity_bucket[uid] = 2
            else: self.user_activity_bucket[uid] = 3
        
        # Item stats with standard smoothing
        item_stats = explicit.groupby('movie_id')['rating'].agg(['mean', 'count'])
        for mid, row in item_stats.iterrows():
            n, avg = row['count'], row['mean']
            smoothed = (n * avg + CONFIG.smoothing_strength * self.global_mean) / (n + CONFIG.smoothing_strength)
            self.item_mean_rating[mid] = smoothed
            self.item_rating_count[mid] = n
            
            thresholds = CONFIG.item_popularity_thresholds
            if n < thresholds[0]: self.item_popularity_bucket[mid] = 0
            elif n < thresholds[1]: self.item_popularity_bucket[mid] = 1
            elif n < thresholds[2]: self.item_popularity_bucket[mid] = 2
            else: self.item_popularity_bucket[mid] = 3
        
        print(f"  Global mean: {self.global_mean:.4f}")
        print(f"  Users with ratings: {len(self.user_mean_rating):,}")
        print(f"  Items with ratings: {len(self.item_mean_rating):,}")


FEATURES = FeatureStoreV17()


# === WIDE & DEEP V17 (Simplified, High Regularization) ===
class WideDeepModelV17(nn.Module):
    """
    V17: Simplified architecture for better generalization.
    - Shallower deep tower: [256, 128]
    - Consistent high dropout: 0.4
    - Removed complex continuous features
    """
    
    def __init__(self, n_users, n_items, n_genres, global_mean=3.5):
        super().__init__()
        
        self.global_mean = nn.Parameter(torch.tensor([global_mean]), requires_grad=False)
        
        # === BIASES ===
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.bias_dropout = nn.Dropout(CONFIG.bias_dropout)
        
        # === EMBEDDINGS ===
        self.user_emb = nn.Embedding(n_users, CONFIG.embedding_dim)
        self.item_emb = nn.Embedding(n_items, CONFIG.embedding_dim)
        self.emb_dropout = nn.Dropout(CONFIG.embedding_dropout)
        
        # === WIDE PART ===
        # Features: genres(19) + year_bucket(6) + user_activity(4) + item_pop(4) = 33
        wide_input_dim = n_genres + 6 + 4 + 4
        self.wide_hidden = nn.Linear(wide_input_dim, CONFIG.wide_hidden_dim)
        self.wide_bn = nn.BatchNorm1d(CONFIG.wide_hidden_dim)
        self.wide_dropout = nn.Dropout(CONFIG.wide_dropout)
        self.wide_output = nn.Linear(CONFIG.wide_hidden_dim, 1)
        
        # === DEEP PART (Simplified) ===
        # Input: user_emb(32) + item_emb(32) + genres(19) + year_normalized(1) = 84
        deep_input_dim = CONFIG.embedding_dim * 2 + n_genres + 1
        
        self.deep_layers = nn.ModuleList()
        self.deep_bns = nn.ModuleList()
        self.deep_dropouts = nn.ModuleList()
        
        prev_dim = deep_input_dim
        for hidden_dim in CONFIG.deep_layers:
            self.deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.deep_bns.append(nn.BatchNorm1d(hidden_dim))
            self.deep_dropouts.append(nn.Dropout(CONFIG.deep_dropout))
            prev_dim = hidden_dim
        
        self.deep_output = nn.Linear(CONFIG.deep_layers[-1], 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        
        nn.init.xavier_uniform_(self.wide_hidden.weight)
        nn.init.zeros_(self.wide_hidden.bias)
        nn.init.xavier_uniform_(self.wide_output.weight)
        nn.init.zeros_(self.wide_output.bias)
        
        for layer in self.deep_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.deep_output.weight)
        nn.init.zeros_(self.deep_output.bias)
    
    def forward(self, user_idx, item_idx, genre, wide_features, year_normalized):
        # === BIASES ===
        u_bias = self.user_bias(user_idx).squeeze(-1)
        i_bias = self.item_bias(item_idx).squeeze(-1)
        if self.training:
            u_bias = self.bias_dropout(u_bias.unsqueeze(-1)).squeeze(-1) / (1 - CONFIG.bias_dropout)
            i_bias = self.bias_dropout(i_bias.unsqueeze(-1)).squeeze(-1) / (1 - CONFIG.bias_dropout)
        
        # === WIDE ===
        wide_h = self.wide_dropout(F.relu(self.wide_bn(self.wide_hidden(wide_features))))
        wide_out = self.wide_output(wide_h).squeeze(-1)
        
        # === EMBEDDINGS ===
        u_emb = self.emb_dropout(self.user_emb(user_idx))
        i_emb = self.emb_dropout(self.item_emb(item_idx))
        
        # === DEEP ===
        deep_in = torch.cat([u_emb, i_emb, genre, year_normalized], dim=1)
        
        x = deep_in
        for layer, bn, dropout in zip(self.deep_layers, self.deep_bns, self.deep_dropouts):
            x = dropout(F.relu(bn(layer(x))))
        
        deep_out = self.deep_output(x).squeeze(-1)
        
        # === FINAL ===
        return self.global_mean + u_bias + i_bias + wide_out + deep_out


# === FEATURE PREPARATION V17 (Simplified) ===
def prepare_features_v17(user_ids, movie_ids):
    """Simplified feature preparation - no complex statistics."""
    n = len(user_ids)
    n_genres = len(FEATURES.genre_list)
    
    # Genre features
    genre = np.zeros((n, n_genres), dtype=np.float32)
    for i, mid in enumerate(movie_ids):
        if mid in FEATURES.genre_features:
            genre[i] = FEATURES.genre_features[mid]
    
    # Wide features: genres + year_bucket + user_activity + item_pop
    wide_dim = n_genres + 6 + 4 + 4
    wide_features = np.zeros((n, wide_dim), dtype=np.float32)
    
    for i, (uid, mid) in enumerate(zip(user_ids, movie_ids)):
        offset = 0
        
        # Genres
        if mid in FEATURES.genre_features:
            wide_features[i, :n_genres] = FEATURES.genre_features[mid]
        offset += n_genres
        
        # Year bucket (one-hot, 6 buckets)
        year_bucket = FEATURES.movie_year_bucket.get(mid, 3)
        wide_features[i, offset + year_bucket] = 1.0
        offset += 6
        
        # User activity bucket (one-hot, 4 buckets)
        activity = FEATURES.user_activity_bucket.get(uid, 1)
        wide_features[i, offset + activity] = 1.0
        offset += 4
        
        # Item popularity bucket (one-hot, 4 buckets)
        pop = FEATURES.item_popularity_bucket.get(mid, 1)
        wide_features[i, offset + pop] = 1.0
    
    # Year normalized (single continuous feature for deep tower)
    year_normalized = np.zeros((n, 1), dtype=np.float32)
    for i, mid in enumerate(movie_ids):
        year = FEATURES.movie_years.get(mid)
        if year:
            year_normalized[i, 0] = (year - 1990) / 30.0  # Normalize to roughly [-1, 1]
    
    return (torch.from_numpy(genre),
            torch.from_numpy(wide_features),
            torch.from_numpy(year_normalized))


# === TRAINING V17 ===
def train_model_v17(train_df, val_df, seed=42):
    """
    V17 Training with high regularization and temporal validation.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WIDE & DEEP V17 (seed={seed})")
    print("="*60)
    print(f"Config: emb={CONFIG.embedding_dim}, layers={CONFIG.deep_layers}")
    print(f"        dropout={CONFIG.deep_dropout}, weight_decay={CONFIG.weight_decay}")
    print(f"        lr={CONFIG.lr}, patience={CONFIG.patience}")
    
    set_all_seeds(seed)
    
    global_mean = float(train_df['rating'].mean())
    n_users, n_items = FEATURES.n_users, FEATURES.n_items
    n_genres = len(FEATURES.genre_list)
    
    # Prepare features
    print("\nPreparing features...")
    train_user_ids = train_df['user_id'].values
    train_movie_ids = train_df['movie_id'].values
    train_user_idx = np.array([FEATURES.user_id_to_idx.get(u, 0) for u in train_user_ids], dtype=np.int64)
    train_item_idx = np.array([FEATURES.item_id_to_idx.get(m, 0) for m in train_movie_ids], dtype=np.int64)
    train_ratings = train_df['rating'].values.astype(np.float32)
    train_genre, train_wide, train_year = prepare_features_v17(train_user_ids, train_movie_ids)
    
    val_user_ids = val_df['user_id'].values
    val_movie_ids = val_df['movie_id'].values
    val_user_idx = np.array([FEATURES.user_id_to_idx.get(u, 0) for u in val_user_ids], dtype=np.int64)
    val_item_idx = np.array([FEATURES.item_id_to_idx.get(m, 0) for m in val_movie_ids], dtype=np.int64)
    val_ratings = val_df['rating'].values.astype(np.float32)
    val_genre, val_wide, val_year = prepare_features_v17(val_user_ids, val_movie_ids)
    
    print(f"Train: {len(train_ratings):,}, Val: {len(val_ratings):,}")
    
    # Model
    model = WideDeepModelV17(n_users, n_items, n_genres, global_mean=global_mean).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer with high weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG.scheduler_factor, 
        patience=CONFIG.scheduler_patience, min_lr=CONFIG.min_lr
    )
    
    # Move to device
    train_user_t = torch.from_numpy(train_user_idx).to(DEVICE)
    train_item_t = torch.from_numpy(train_item_idx).to(DEVICE)
    train_rating_t = torch.from_numpy(train_ratings).to(DEVICE)
    train_genre = train_genre.to(DEVICE)
    train_wide = train_wide.to(DEVICE)
    train_year = train_year.to(DEVICE)
    
    val_user_t = torch.from_numpy(val_user_idx).to(DEVICE)
    val_item_t = torch.from_numpy(val_item_idx).to(DEVICE)
    val_rating_t = torch.from_numpy(val_ratings).to(DEVICE)
    val_genre = val_genre.to(DEVICE)
    val_wide = val_wide.to(DEVICE)
    val_year = val_year.to(DEVICE)
    
    # Training state
    best_val_rmse = float('inf')
    patience_cnt = 0
    best_state = None
    best_epoch = 0
    
    n_train = len(train_ratings)
    n_batches = (n_train + CONFIG.batch_size - 1) // CONFIG.batch_size
    
    print("\nTraining...")
    for epoch in range(CONFIG.n_epochs):
        model.train()
        
        # Learning rate warmup
        if epoch < CONFIG.warmup_epochs:
            warmup_factor = (epoch + 1) / CONFIG.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = CONFIG.lr * warmup_factor
        
        perm = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        
        for b in range(n_batches):
            s, e = b * CONFIG.batch_size, min((b + 1) * CONFIG.batch_size, n_train)
            idx = perm[s:e]
            
            pred = model(train_user_t[idx], train_item_t[idx],
                        train_genre[idx], train_wide[idx], train_year[idx])
            loss = F.mse_loss(pred, train_rating_t[idx])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * (e - s)
        
        train_rmse = np.sqrt(epoch_loss / n_train)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(val_user_t, val_item_t, val_genre, val_wide, val_year)
            val_rmse = np.sqrt(F.mse_loss(val_pred, val_rating_t).item())
        
        # Step scheduler after warmup
        if epoch >= CONFIG.warmup_epochs:
            scheduler.step(val_rmse)
        
        gap = train_rmse - val_rmse
        status = "OK" if gap > -0.05 else "WARNING" if gap > -0.1 else "OVERFIT!"
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch {epoch+1:2d}: Train={train_rmse:.4f}, Val={val_rmse:.4f}, Gap={gap:+.4f} [{status}], LR={current_lr:.6f}")
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    print(f"\n✓ Best Epoch: {best_epoch}, Val RMSE: {best_val_rmse:.4f}")
    gc.collect()
    return model, best_val_rmse, best_epoch


# === PREDICTION ===
def predict_batch_v17(model, user_ids, movie_ids, batch_size=8192):
    model.eval()
    all_preds = []
    
    for start in range(0, len(user_ids), batch_size):
        end = min(start + batch_size, len(user_ids))
        batch_users, batch_movies = user_ids[start:end], movie_ids[start:end]
        
        user_idx = np.array([FEATURES.user_id_to_idx.get(u, 0) for u in batch_users], dtype=np.int64)
        item_idx = np.array([FEATURES.item_id_to_idx.get(m, 0) for m in batch_movies], dtype=np.int64)
        genre, wide, year = prepare_features_v17(batch_users, batch_movies)
        
        with torch.no_grad():
            preds = model(torch.from_numpy(user_idx).to(DEVICE),
                         torch.from_numpy(item_idx).to(DEVICE),
                         genre.to(DEVICE), wide.to(DEVICE), year.to(DEVICE))
        all_preds.append(preds.cpu().numpy())
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    return np.concatenate(all_preds)


# === MAIN ===
def main():
    print("\n" + "="*70)
    print("WIDE & DEEP V17 - SIMPLIFIED ARCHITECTURE")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    movies_df = pd.read_csv(f"{DATA_DIR}/movies.csv")
    submission_df = pd.read_csv(f"{DATA_DIR}/ratings_submission.csv")
    
    split_ids = submission_df['id'].str.split('_', expand=True)
    submission_df['user_id'] = split_ids[0].astype('int32')
    submission_df['movie_id'] = split_ids[1].astype('int32')
    
    train_explicit = train_df[train_df['rating'].notna()].copy()
    print(f"Explicit ratings: {len(train_explicit):,}")
    
    # Build basic features (ID mappings, genres, years)
    FEATURES.build_basic(train_df, submission_df, movies_df)
    gc.collect()
    
    # === V17: STRATIFIED USER SPLIT (25% val) ===
    print("\n" + "="*60)
    print("V17: STRATIFIED USER SPLIT (25% validation)")
    print("  Larger validation for better test RMSE estimate")
    print("="*60)
    
    train_split, val_split = stratified_rating_split(train_explicit, val_ratio=CONFIG.val_ratio)
    print(f"  Train: {len(train_split):,} ratings")
    print(f"  Val: {len(val_split):,} ratings")
    
    # Build target encoding on full data (users appear in both train/val)
    FEATURES.build_target_encoding(train_explicit)
    
    # === ENSEMBLE TRAINING ===
    print("\n" + "="*60)
    n_seeds = min(CONFIG.n_seeds_to_use, len(CONFIG.seeds))
    seeds_to_use = CONFIG.seeds[:n_seeds]
    print(f"TRAINING ENSEMBLE ({n_seeds} models)")
    print("="*60)
    
    models = []
    val_rmses = []
    best_epochs = []
    
    for seed in seeds_to_use:
        model, val_rmse, best_epoch = train_model_v17(train_split, val_split, seed=seed)
        models.append(model)
        val_rmses.append(val_rmse)
        best_epochs.append(best_epoch)
        gc.collect()
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    print(f"\n--- Per-Model Summary ---")
    for i, (seed, vr, be) in enumerate(zip(seeds_to_use, val_rmses, best_epochs)):
        print(f"  Model {i+1} (seed={seed}): Val RMSE={vr:.4f}, Best Epoch={be}")
    print(f"  Average Val RMSE: {np.mean(val_rmses):.4f}")
    print(f"  Std Val RMSE: {np.std(val_rmses):.4f}")
    
    # === ENSEMBLE EVALUATION ===
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION (Temporal Validation)")
    print("="*60)
    
    val_preds_list = []
    for model in models:
        preds = predict_batch_v17(model, val_split['user_id'].values, val_split['movie_id'].values)
        val_preds_list.append(preds)
    
    val_preds_array = np.array(val_preds_list)
    ensemble_val_preds = val_preds_array.mean(axis=0)
    val_targets = val_split['rating'].values
    ensemble_rmse = np.sqrt(np.mean((ensemble_val_preds - val_targets) ** 2))
    
    pred_std = val_preds_array.std(axis=0)
    
    print(f"\n  ensemble_val_rmse (temporal): {ensemble_rmse:.4f}")
    print(f"  prediction_std_mean: {pred_std.mean():.4f}")
    print(f"  prediction_std_95th_percentile: {np.percentile(pred_std, 95):.4f}")
    print(f"  val_prediction_mean: {ensemble_val_preds.mean():.4f}")
    
    # === RETRAIN ON FULL DATA ===
    print("\n" + "="*60)
    print("RETRAINING ENSEMBLE ON FULL DATA")
    print("="*60)
    
    # For final training, use stratified split with smaller val ratio
    final_train, final_val = stratified_rating_split(train_explicit, val_ratio=CONFIG.final_val_ratio)
    
    # Rebuild target encoding on final training data
    FEATURES.build_target_encoding(final_train)
    
    final_models = []
    for seed in seeds_to_use:
        model, _, _ = train_model_v17(final_train, final_val, seed=seed)
        final_models.append(model)
        gc.collect()
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    # === GENERATE SUBMISSION ===
    print("\n" + "="*60)
    print("GENERATING ENSEMBLE SUBMISSION")
    print("="*60)
    
    # Get ensemble predictions
    sub_preds_list = []
    for model in final_models:
        preds = predict_batch_v17(model, submission_df['user_id'].values, submission_df['movie_id'].values)
        sub_preds_list.append(preds)
    
    ensemble_sub_preds = np.mean(sub_preds_list, axis=0)
    
    # Clip to valid range
    ensemble_sub_preds = np.clip(ensemble_sub_preds, 0.5, 5.0)
    
    submission = pd.DataFrame({
        'id': submission_df['id'],
        'prediction': ensemble_sub_preds
    })
    
    output_file = 'submission_wide_deep_v17.csv'
    submission.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"Predictions: {len(submission):,}")
    print(f"Rating range: [{ensemble_sub_preds.min():.2f}, {ensemble_sub_preds.max():.2f}]")
    print(f"Rating mean (submission): {ensemble_sub_preds.mean():.4f}")
    
    print("\n" + "="*70)
    print(f"V17 EXPECTED TEST RMSE (25% val): ~{ensemble_rmse:.4f}")
    print("  Shallower model [256,128], moderate regularization")
    print("="*70)


if __name__ == "__main__":
    main()

