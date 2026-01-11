#!/usr/bin/env python3
"""
Wide & Deep Recommender V6 - ROBUST GENERALIZATION
Target: Test RMSE < 0.80

Analysis of V5: Val RMSE 0.787 but Test RMSE 0.8236 (gap = 0.037)
This gap suggests our validation set is not representative of test.

V6 Strategy - Focus on ROBUST features that generalize:
1. HEAVIER REGULARIZATION: Back to emb=32, stronger weight decay
2. MORE AGGRESSIVE SMOOTHING: m=50 (vs m=10) for target encoding
3. SIMPLER ARCHITECTURE: [256, 128] - avoid overfitting to validation
4. ENSEMBLE: Train 3 models with different seeds, average predictions
5. BIAS-HEAVY: Let biases do more work (they generalize better)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import gc
import os
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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


# === STRATIFIED USER SPLIT ===
def stratified_user_split(df: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(random_state)
    train_indices, val_indices = [], []
    
    for user_id, group in df.groupby('user_id'):
        indices = group.index.tolist()
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        n_train = len(indices) - n_val
        if n_train < len(indices) * 0.8:
            n_train = int(len(indices) * 0.8)
            n_val = len(indices) - n_train
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])
    
    return df.loc[train_indices].reset_index(drop=True), df.loc[val_indices].reset_index(drop=True)


# === FEATURE STORE V6 ===
class FeatureStoreV6:
    def __init__(self):
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.n_users = 0
        self.n_items = 0
        self.genre_list = []
        self.genre_features = {}
        self.tag_vocab = []
        self.tag_features = {}
        self.movie_years = {}
        self.movie_year_bucket = {}
        self.user_items = {}
        
        # Heavily smoothed target encoding
        self.user_smoothed_rating = {}
        self.item_smoothed_rating = {}
        self.user_rating_count = {}
        self.item_rating_count = {}
        self.user_activity_bucket = {}
        self.item_popularity_bucket = {}
        
        self.global_mean = 3.5
        self.smoothing_factor = 50  # INCREASED from 10 for better generalization
    
    def build_basic(self, train_df, submission_df, movies_df, tags_df, max_tags=80):
        print("\n" + "="*60)
        print("BUILDING FEATURE STORE V6 (Robust Generalization)")
        print("="*60)
        
        # ID mappings
        print("[1/5] Building ID mappings...")
        all_users = set(train_df['user_id'].unique()) | set(submission_df['user_id'].unique())
        all_items = set(train_df['movie_id'].unique()) | set(submission_df['movie_id'].unique()) | set(movies_df['movie_id'].unique())
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(all_users))}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(sorted(all_items))}
        self.n_users = len(self.user_id_to_idx)
        self.n_items = len(self.item_id_to_idx)
        print(f"  Users: {self.n_users:,}, Items: {self.n_items:,}")
        
        # Movie metadata
        print("[2/5] Extracting movie metadata...")
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
        print("[3/5] Building genre features...")
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
        
        # Tag TF-IDF (reduced)
        print("[4/5] Building tag TF-IDF features...")
        tags_norm = tags_df[['movie_id', 'tag']].copy()
        tags_norm['tag_norm'] = tags_norm['tag'].apply(normalize_tag)
        tags_norm = tags_norm[tags_norm['tag_norm'] != ''].drop_duplicates(['movie_id', 'tag_norm'])
        
        tag_counts = tags_norm['tag_norm'].value_counts()
        self.tag_vocab = tag_counts[tag_counts >= 15].head(max_tags).index.tolist()
        tag_to_idx = {t: i for i, t in enumerate(self.tag_vocab)}
        
        movie_tags = defaultdict(lambda: defaultdict(int))
        for _, row in tags_norm.iterrows():
            if row['tag_norm'] in tag_to_idx:
                movie_tags[row['movie_id']][row['tag_norm']] += 1
        
        doc_freq = defaultdict(int)
        for tags in movie_tags.values():
            for t in tags: doc_freq[t] += 1
        n_docs = max(len(movie_tags), 1)
        idf = {t: np.log(n_docs / (1 + doc_freq[t])) for t in self.tag_vocab}
        
        for mid, tags in movie_tags.items():
            total = sum(tags.values())
            vec = np.zeros(len(self.tag_vocab), dtype=np.float32)
            for t, cnt in tags.items():
                vec[tag_to_idx[t]] = (cnt / total) * idf[t]
            norm = np.linalg.norm(vec)
            if norm > 0: vec /= norm
            self.tag_features[mid] = vec
        print(f"  Tags: {len(self.tag_vocab)}")
        
        # User interactions
        print("[5/5] Building user interactions...")
        for uid in self.user_id_to_idx:
            self.user_items[self.user_id_to_idx[uid]] = set()
        for _, row in train_df.iterrows():
            uidx = self.user_id_to_idx.get(row['user_id'])
            iidx = self.item_id_to_idx.get(row['movie_id'])
            if uidx is not None and iidx is not None:
                self.user_items[uidx].add(iidx)
        
        del tags_norm, movie_tags
        gc.collect()
        print("="*60)
    
    def build_smoothed_target_encoding(self, train_df: pd.DataFrame, smoothing_factor: int = 50):
        """Heavy smoothing for robust generalization."""
        print(f"\nBuilding HEAVILY smoothed target encoding (m={smoothing_factor})...")
        self.smoothing_factor = smoothing_factor
        
        explicit = train_df[train_df['rating'].notna()]
        self.global_mean = explicit['rating'].mean()
        
        # User stats
        user_stats = explicit.groupby('user_id')['rating'].agg(['mean', 'count'])
        for uid, row in user_stats.iterrows():
            n, avg = row['count'], row['mean']
            # Heavy smoothing pulls rare users toward global mean
            self.user_smoothed_rating[uid] = (n * avg + smoothing_factor * self.global_mean) / (n + smoothing_factor)
            self.user_rating_count[uid] = n
            if n < 20: self.user_activity_bucket[uid] = 0
            elif n < 50: self.user_activity_bucket[uid] = 1
            elif n < 100: self.user_activity_bucket[uid] = 2
            else: self.user_activity_bucket[uid] = 3
        
        # Item stats
        item_stats = explicit.groupby('movie_id')['rating'].agg(['mean', 'count'])
        for mid, row in item_stats.iterrows():
            n, avg = row['count'], row['mean']
            self.item_smoothed_rating[mid] = (n * avg + smoothing_factor * self.global_mean) / (n + smoothing_factor)
            self.item_rating_count[mid] = n
            if n < 50: self.item_popularity_bucket[mid] = 0
            elif n < 200: self.item_popularity_bucket[mid] = 1
            elif n < 500: self.item_popularity_bucket[mid] = 2
            else: self.item_popularity_bucket[mid] = 3
        
        print(f"  Global mean: {self.global_mean:.4f}")
        print(f"  User smoothed range: [{min(self.user_smoothed_rating.values()):.2f}, {max(self.user_smoothed_rating.values()):.2f}]")
        print(f"  Item smoothed range: [{min(self.item_smoothed_rating.values()):.2f}, {max(self.item_smoothed_rating.values()):.2f}]")


FEATURES = FeatureStoreV6()


# === WIDE & DEEP V6 (ROBUST) ===
class WideDeepModelV6(nn.Module):
    """
    V6: Back to basics with heavy regularization.
    - Smaller embeddings (32)
    - Simple architecture [256, 128]
    - High dropout (0.5)
    - Strong biases
    """
    
    def __init__(self, n_users, n_items, n_genres, n_tags,
                 embedding_dim=32,  # BACK TO 32
                 deep_layers=[256, 128],  # SIMPLER
                 dropout=0.5,
                 global_mean=3.5):
        super().__init__()
        
        self.global_mean = nn.Parameter(torch.tensor([global_mean]), requires_grad=False)
        
        # EXPLICIT BIASES (key for generalization)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        # WIDE PART (simple, regularized)
        wide_input_dim = n_genres + 6 + 4 + 4 + 2  # genre + year + activity + pop + smoothed
        self.wide_linear = nn.Linear(wide_input_dim, 1, bias=True)
        
        # DEEP PART
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        
        n_continuous = 5
        deep_input_dim = embedding_dim * 2 + n_genres + n_tags + n_continuous
        
        self.deep_layers = nn.ModuleList()
        self.deep_bns = nn.ModuleList()
        self.deep_dropouts = nn.ModuleList()
        
        prev_dim = deep_input_dim
        for hidden_dim in deep_layers:
            self.deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.deep_bns.append(nn.BatchNorm1d(hidden_dim))
            self.deep_dropouts.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.deep_output = nn.Linear(deep_layers[-1], 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.xavier_uniform_(self.wide_linear.weight)
        nn.init.zeros_(self.wide_linear.bias)
        for layer in self.deep_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.deep_output.weight)
        nn.init.zeros_(self.deep_output.bias)
    
    def forward(self, user_idx, item_idx, genre, tag, wide_features, deep_continuous):
        u_bias = self.user_bias(user_idx).squeeze(-1)
        i_bias = self.item_bias(item_idx).squeeze(-1)
        wide_out = self.wide_linear(wide_features).squeeze(-1)
        
        u_emb = self.user_emb(user_idx)
        i_emb = self.item_emb(item_idx)
        deep_in = torch.cat([u_emb, i_emb, genre, tag, deep_continuous], dim=1)
        
        x = deep_in
        for layer, bn, dropout in zip(self.deep_layers, self.deep_bns, self.deep_dropouts):
            x = dropout(F.relu(bn(layer(x))))
        
        deep_out = self.deep_output(x).squeeze(-1)
        
        return self.global_mean + u_bias + i_bias + wide_out + deep_out


# === FEATURE PREPARATION V6 ===
def prepare_features_v6(user_ids, movie_ids):
    n = len(user_ids)
    n_genres = len(FEATURES.genre_list)
    n_tags = len(FEATURES.tag_vocab)
    
    genre = np.zeros((n, n_genres), dtype=np.float32)
    tag = np.zeros((n, n_tags), dtype=np.float32)
    
    for i, mid in enumerate(movie_ids):
        if mid in FEATURES.genre_features: genre[i] = FEATURES.genre_features[mid]
        if mid in FEATURES.tag_features: tag[i] = FEATURES.tag_features[mid]
    
    # Wide features
    wide_dim = n_genres + 6 + 4 + 4 + 2
    wide_features = np.zeros((n, wide_dim), dtype=np.float32)
    
    for i, (uid, mid) in enumerate(zip(user_ids, movie_ids)):
        offset = 0
        if mid in FEATURES.genre_features:
            wide_features[i, :n_genres] = FEATURES.genre_features[mid]
        offset += n_genres
        
        year_bucket = FEATURES.movie_year_bucket.get(mid, 3)
        wide_features[i, offset + year_bucket] = 1.0
        offset += 6
        
        activity = FEATURES.user_activity_bucket.get(uid, 1)
        wide_features[i, offset + activity] = 1.0
        offset += 4
        
        pop = FEATURES.item_popularity_bucket.get(mid, 1)
        wide_features[i, offset + pop] = 1.0
        offset += 4
        
        user_smoothed = FEATURES.user_smoothed_rating.get(uid, FEATURES.global_mean)
        item_smoothed = FEATURES.item_smoothed_rating.get(mid, FEATURES.global_mean)
        wide_features[i, offset] = (user_smoothed - FEATURES.global_mean) / 1.5
        wide_features[i, offset + 1] = (item_smoothed - FEATURES.global_mean) / 1.5
    
    # Deep continuous (5)
    deep_continuous = np.zeros((n, 5), dtype=np.float32)
    
    for i, (uid, mid) in enumerate(zip(user_ids, movie_ids)):
        year = FEATURES.movie_years.get(mid)
        if year: deep_continuous[i, 0] = (year - 1990) / 30.0
        
        uidx = FEATURES.user_id_to_idx.get(uid)
        if uidx is not None:
            n_inter = len(FEATURES.user_items.get(uidx, set()))
            deep_continuous[i, 1] = np.log1p(n_inter) / 6.0
        
        item_count = FEATURES.item_rating_count.get(mid, 1)
        deep_continuous[i, 2] = np.log1p(item_count) / 8.0
        
        user_smoothed = FEATURES.user_smoothed_rating.get(uid, FEATURES.global_mean)
        item_smoothed = FEATURES.item_smoothed_rating.get(mid, FEATURES.global_mean)
        deep_continuous[i, 3] = (user_smoothed - FEATURES.global_mean) / 1.5
        deep_continuous[i, 4] = (item_smoothed - FEATURES.global_mean) / 1.5
    
    return (torch.from_numpy(genre), torch.from_numpy(tag),
            torch.from_numpy(wide_features), torch.from_numpy(deep_continuous))


# === TRAINING V6 ===
def train_model_v6(train_df, val_df, seed=42,
                   embedding_dim=32,
                   deep_layers=[256, 128],
                   dropout=0.5,
                   lr=0.001,
                   weight_decay=1e-3,  # STRONGER
                   n_epochs=50,
                   batch_size=2048,
                   patience=10):  # MORE PATIENCE
    
    print(f"\n{'='*60}")
    print(f"TRAINING WIDE & DEEP V6 (seed={seed})")
    print("="*60)
    print(f"Config: emb={embedding_dim}, layers={deep_layers}, dropout={dropout}")
    print(f"        lr={lr}, weight_decay={weight_decay}, batch={batch_size}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    global_mean = float(train_df['rating'].mean())
    n_users, n_items = FEATURES.n_users, FEATURES.n_items
    n_genres, n_tags = len(FEATURES.genre_list), len(FEATURES.tag_vocab)
    
    # Prepare features
    print("\nPreparing features...")
    train_user_ids = train_df['user_id'].values
    train_movie_ids = train_df['movie_id'].values
    train_user_idx = np.array([FEATURES.user_id_to_idx[u] for u in train_user_ids], dtype=np.int64)
    train_item_idx = np.array([FEATURES.item_id_to_idx[m] for m in train_movie_ids], dtype=np.int64)
    train_ratings = train_df['rating'].values.astype(np.float32)
    train_genre, train_tag, train_wide, train_deep = prepare_features_v6(train_user_ids, train_movie_ids)
    
    val_user_ids = val_df['user_id'].values
    val_movie_ids = val_df['movie_id'].values
    val_user_idx = np.array([FEATURES.user_id_to_idx.get(u, 0) for u in val_user_ids], dtype=np.int64)
    val_item_idx = np.array([FEATURES.item_id_to_idx.get(m, 0) for m in val_movie_ids], dtype=np.int64)
    val_ratings = val_df['rating'].values.astype(np.float32)
    val_genre, val_tag, val_wide, val_deep = prepare_features_v6(val_user_ids, val_movie_ids)
    
    print(f"Train: {len(train_ratings):,}, Val: {len(val_ratings):,}")
    
    # Model
    model = WideDeepModelV6(n_users, n_items, n_genres, n_tags,
                            embedding_dim=embedding_dim, deep_layers=deep_layers,
                            dropout=dropout, global_mean=global_mean).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Move to device
    train_user_t = torch.from_numpy(train_user_idx).to(DEVICE)
    train_item_t = torch.from_numpy(train_item_idx).to(DEVICE)
    train_rating_t = torch.from_numpy(train_ratings).to(DEVICE)
    train_genre, train_tag = train_genre.to(DEVICE), train_tag.to(DEVICE)
    train_wide, train_deep = train_wide.to(DEVICE), train_deep.to(DEVICE)
    
    val_user_t = torch.from_numpy(val_user_idx).to(DEVICE)
    val_item_t = torch.from_numpy(val_item_idx).to(DEVICE)
    val_rating_t = torch.from_numpy(val_ratings).to(DEVICE)
    val_genre, val_tag = val_genre.to(DEVICE), val_tag.to(DEVICE)
    val_wide, val_deep = val_wide.to(DEVICE), val_deep.to(DEVICE)
    
    # Training
    best_val_rmse = float('inf')
    patience_cnt = 0
    best_state = None
    n_train = len(train_ratings)
    n_batches = (n_train + batch_size - 1) // batch_size
    
    print("\nTraining...")
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        
        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, n_train)
            idx = perm[s:e]
            
            pred = model(train_user_t[idx], train_item_t[idx],
                        train_genre[idx], train_tag[idx],
                        train_wide[idx], train_deep[idx])
            loss = F.mse_loss(pred, train_rating_t[idx])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * (e - s)
        
        train_rmse = np.sqrt(epoch_loss / n_train)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(val_user_t, val_item_t, val_genre, val_tag, val_wide, val_deep)
            val_rmse = np.sqrt(F.mse_loss(val_pred, val_rating_t).item())
        
        scheduler.step(val_rmse)
        gap = train_rmse - val_rmse
        status = "OK" if gap > -0.05 else "WARNING" if gap > -0.1 else "OVERFIT!"
        
        print(f"  Epoch {epoch+1:2d}: Train={train_rmse:.4f}, Val={val_rmse:.4f}, Gap={gap:+.4f} [{status}]")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    print(f"\n✓ Best Val RMSE: {best_val_rmse:.4f}")
    gc.collect()
    return model, best_val_rmse


# === PREDICTION ===
def predict_batch_v6(model, user_ids, movie_ids, batch_size=8192):
    model.eval()
    all_preds = []
    
    for start in range(0, len(user_ids), batch_size):
        end = min(start + batch_size, len(user_ids))
        batch_users, batch_movies = user_ids[start:end], movie_ids[start:end]
        
        user_idx = np.array([FEATURES.user_id_to_idx.get(u, 0) for u in batch_users], dtype=np.int64)
        item_idx = np.array([FEATURES.item_id_to_idx.get(m, 0) for m in batch_movies], dtype=np.int64)
        genre, tag, wide, deep = prepare_features_v6(batch_users, batch_movies)
        
        with torch.no_grad():
            preds = model(torch.from_numpy(user_idx).to(DEVICE),
                         torch.from_numpy(item_idx).to(DEVICE),
                         genre.to(DEVICE), tag.to(DEVICE),
                         wide.to(DEVICE), deep.to(DEVICE))
        all_preds.append(preds.cpu().numpy())
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    return np.clip(np.concatenate(all_preds), 0.5, 5.0)


# === MAIN ===
def main():
    print("\n" + "="*70)
    print("WIDE & DEEP V6 - ROBUST GENERALIZATION (ENSEMBLE)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    movies_df = pd.read_csv(f"{DATA_DIR}/movies.csv")
    tags_df = pd.read_csv(f"{DATA_DIR}/tags.csv")
    submission_df = pd.read_csv(f"{DATA_DIR}/ratings_submission.csv")
    
    split_ids = submission_df['id'].str.split('_', expand=True)
    submission_df['user_id'] = split_ids[0].astype('int32')
    submission_df['movie_id'] = split_ids[1].astype('int32')
    
    train_explicit = train_df[train_df['rating'].notna()].copy()
    print(f"Explicit ratings: {len(train_explicit):,}")
    
    # Build basic features
    FEATURES.build_basic(train_df, submission_df, movies_df, tags_df, max_tags=80)
    del tags_df
    gc.collect()
    
    # Use ALL data for target encoding (more robust)
    FEATURES.build_smoothed_target_encoding(train_explicit, smoothing_factor=50)
    
    # Train/Val split
    print("\n" + "="*60)
    print("STRATIFIED USER SPLIT")
    print("="*60)
    train_split, val_split = stratified_user_split(train_explicit, val_ratio=0.15, random_state=42)
    print(f"Train: {len(train_split):,}, Val: {len(val_split):,}")
    
    # === ENSEMBLE: Train 3 models with different seeds ===
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE (3 models)")
    print("="*60)
    
    models = []
    seeds = [42, 123, 456]
    val_rmses = []
    
    for seed in seeds:
        model, val_rmse = train_model_v6(
            train_split, val_split, seed=seed,
            embedding_dim=32, deep_layers=[256, 128],
            dropout=0.5, lr=0.001, weight_decay=1e-3,
            n_epochs=50, batch_size=2048, patience=10
        )
        models.append(model)
        val_rmses.append(val_rmse)
        gc.collect()
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    print(f"\nEnsemble Val RMSEs: {val_rmses}")
    print(f"Average Val RMSE: {np.mean(val_rmses):.4f}")
    
    # Evaluate ensemble on validation
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    
    val_preds_list = []
    for model in models:
        preds = predict_batch_v6(model, val_split['user_id'].values, val_split['movie_id'].values)
        val_preds_list.append(preds)
    
    ensemble_preds = np.mean(val_preds_list, axis=0)
    ensemble_rmse = np.sqrt(np.mean((ensemble_preds - val_split['rating'].values) ** 2))
    print(f"Ensemble Val RMSE: {ensemble_rmse:.4f}")
    
    # === RETRAIN ON FULL DATA ===
    print("\n" + "="*60)
    print("RETRAINING ENSEMBLE ON FULL DATA")
    print("="*60)
    
    # Smaller val for early stopping only
    final_train, final_val = stratified_user_split(train_explicit, val_ratio=0.05, random_state=42)
    
    final_models = []
    for seed in seeds:
        model, _ = train_model_v6(
            final_train, final_val, seed=seed,
            embedding_dim=32, deep_layers=[256, 128],
            dropout=0.5, lr=0.001, weight_decay=1e-3,
            n_epochs=50, batch_size=2048, patience=10
        )
        final_models.append(model)
        gc.collect()
        if DEVICE.type == 'mps': torch.mps.empty_cache()
    
    # === GENERATE SUBMISSION ===
    print("\n" + "="*60)
    print("GENERATING ENSEMBLE SUBMISSION")
    print("="*60)
    
    sub_preds_list = []
    for model in final_models:
        preds = predict_batch_v6(model, submission_df['user_id'].values, submission_df['movie_id'].values)
        sub_preds_list.append(preds)
    
    ensemble_sub_preds = np.mean(sub_preds_list, axis=0)
    
    submission = pd.DataFrame({
        'id': submission_df['id'],
        'prediction': ensemble_sub_preds
    })
    
    output_file = 'submission_wide_deep_v6.csv'
    submission.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"Predictions: {len(submission):,}")
    print(f"Rating range: [{ensemble_sub_preds.min():.2f}, {ensemble_sub_preds.max():.2f}]")
    print(f"Rating mean: {ensemble_sub_preds.mean():.4f}")
    
    print("\n" + "="*70)
    print(f"EXPECTED TEST RMSE: ~{ensemble_rmse:.4f} (ensemble)")
    print("="*70)


if __name__ == "__main__":
    main()

