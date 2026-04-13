"""
Contrastive Dataset Builder for Cross-Content Recommendations

Builds positive and hard negative pairs for contrastive learning:
- Positive pairs: (book_idx, movie_idx) sharing at least 1 genre
- Hard negatives: Semantically similar but genre-incompatible movies

Usage:
    python src/data/contrastive_dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import json
import numpy as np
import pandas as pd
import faiss
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from tqdm import tqdm

from project_config import (
    EMBEDDINGS_DIR,
    PROCESSED_DATA_DIR,
    GOODREADS_RAW_CSV,
    IMDB_FILTERED_CSV,
    GENRE_MAPPING_JSON
)
from utils.helpers import (
    setup_logger,
    timer,
    load_datasets,
    load_genre_mapping,
    get_mapped_movie_genres
)


class ContrastivePairBuilder:
    """Builds positive and hard negative pairs for contrastive learning"""

    def __init__(self):
        self.logger = setup_logger("contrastive_dataset")
        self.book_df = None
        self.movie_df = None
        self.book_embeddings = None  # (9923, 768)
        self.movie_embeddings = None  # (7662, 768)
        self.book_ids = None
        self.movie_ids = None
        self.genre_mapping = None
        self.temp_movie_index = None  # Temporary FAISS for hard negative mining

    def load_data(self) -> None:
        """Load all required data into memory"""
        self.logger.info("\n" + "="*70)
        self.logger.info("LOADING DATA")
        self.logger.info("="*70)

        with timer("Loading datasets", self.logger):
            self.book_df, self.movie_df = load_datasets(self.logger)
            self.logger.info(f"  Loaded {len(self.book_df)} books")
            self.logger.info(f"  Loaded {len(self.movie_df)} movies")

        with timer("Loading embeddings", self.logger):
            # Load embeddings
            book_emb_path = EMBEDDINGS_DIR / "book_embeddings.npy"
            movie_emb_path = EMBEDDINGS_DIR / "movie_embeddings.npy"

            self.book_embeddings = np.load(book_emb_path)
            self.movie_embeddings = np.load(movie_emb_path)

            # Load ID mappings
            with open(EMBEDDINGS_DIR / "book_ids.json") as f:
                self.book_ids = json.load(f)
            with open(EMBEDDINGS_DIR / "movie_ids.json") as f:
                self.movie_ids = json.load(f)

            self.logger.info(f"  Book embeddings: {self.book_embeddings.shape}")
            self.logger.info(f"  Movie embeddings: {self.movie_embeddings.shape}")

            # Memory usage
            total_mb = (self.book_embeddings.nbytes + self.movie_embeddings.nbytes) / (1024**2)
            self.logger.info(f"  Total memory: {total_mb:.1f} MB")

        with timer("Loading genre mapping", self.logger):
            self.genre_mapping = load_genre_mapping(self.logger)
            self.logger.info(f"  Loaded {len(self.genre_mapping)} genre mappings")

    def build_temp_faiss_index(self) -> None:
        """Build temporary 768-dim FAISS index for hard negative mining"""
        self.logger.info("\n" + "="*70)
        self.logger.info("BUILDING TEMPORARY FAISS INDEX")
        self.logger.info("="*70)

        with timer("Building FAISS index", self.logger):
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.temp_movie_index = faiss.IndexFlatIP(768)
            self.temp_movie_index.add(self.movie_embeddings.astype('float32'))
            self.logger.info(f"  ✓ Index built: {self.temp_movie_index.ntotal} movies")

    def find_positive_pairs(self, max_per_book: int = 5) -> List[Tuple[int, int]]:
        """
        Find (book_idx, movie_idx) pairs sharing ≥1 genre.

        Args:
            max_per_book: Maximum movies per book (default 5)

        Returns:
            List of (book_idx, movie_idx) tuples
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("MINING POSITIVE PAIRS")
        self.logger.info("="*70)

        positive_pairs = []
        books_with_pairs = 0
        books_without_pairs = 0
        books_without_mapping = 0

        for book_idx, book_row in tqdm(self.book_df.iterrows(),
                                       total=len(self.book_df),
                                       desc="Processing books"):
            # Skip if book not in embeddings
            if book_idx not in self.book_ids:
                continue

            # Get mapped movie genres
            try:
                book_movie_genres = get_mapped_movie_genres(book_row, self.genre_mapping)
            except Exception as e:
                books_without_mapping += 1
                continue

            if not book_movie_genres:
                books_without_mapping += 1
                continue

            # Find genre-compatible movies
            compatible_movies = []
            for movie_idx, movie_row in self.movie_df.iterrows():
                # Skip if movie not in embeddings
                if movie_idx not in self.movie_ids:
                    continue

                # Parse movie genres
                movie_genres_str = str(movie_row.get('genres', ''))
                if movie_genres_str in ['', 'nan', '\\N']:
                    continue

                movie_genres = [g.strip() for g in movie_genres_str.split(',')]

                # Check genre overlap
                if any(g in movie_genres for g in book_movie_genres):
                    # Get embedding indices
                    book_emb_idx = self.book_ids.index(book_idx)
                    movie_emb_idx = self.movie_ids.index(movie_idx)

                    # Store with rating for stratification
                    rating = movie_row.get('averageRating', 0)
                    compatible_movies.append((movie_emb_idx, rating))

            if not compatible_movies:
                books_without_pairs += 1
                continue

            # Sample up to max_per_book movies (prefer higher-rated)
            compatible_movies.sort(key=lambda x: x[1], reverse=True)
            selected_movies = compatible_movies[:max_per_book]

            book_emb_idx = self.book_ids.index(book_idx)
            for movie_emb_idx, _ in selected_movies:
                positive_pairs.append((book_emb_idx, movie_emb_idx))

            books_with_pairs += 1

        self.logger.info(f"\n✓ Positive pair mining complete:")
        self.logger.info(f"  Total pairs: {len(positive_pairs)}")
        self.logger.info(f"  Books with pairs: {books_with_pairs}")
        self.logger.info(f"  Books without pairs: {books_without_pairs}")
        self.logger.info(f"  Books without mapping: {books_without_mapping}")
        self.logger.info(f"  Avg pairs per book: {len(positive_pairs) / books_with_pairs:.2f}")

        return positive_pairs

    def find_hard_negatives(self, positive_pairs: List[Tuple[int, int]], k: int = 50) -> Dict[str, int]:
        """
        For each positive pair, find 1 hard negative movie.

        Args:
            positive_pairs: List of (book_idx, movie_idx) positives
            k: Number of candidates to retrieve (default 50)

        Returns:
            Dict mapping "bookidx_movieidx" → movie_idx_neg
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("MINING HARD NEGATIVES")
        self.logger.info("="*70)

        hard_negatives = {}
        success_count = 0
        fallback_count = 0

        # Group pairs by book for efficiency
        pairs_by_book = defaultdict(list)
        for book_idx, movie_idx in positive_pairs:
            pairs_by_book[book_idx].append(movie_idx)

        for book_idx, movie_indices in tqdm(pairs_by_book.items(), desc="Mining hard negatives"):
            # Get book embedding and search
            book_embedding = self.book_embeddings[book_idx:book_idx+1].astype('float32')
            scores, candidates = self.temp_movie_index.search(book_embedding, k)
            candidate_indices = candidates[0].tolist()

            # Get book's genres for filtering
            book_df_idx = self.book_ids[book_idx]
            book_row = self.book_df.loc[book_df_idx]
            book_movie_genres = get_mapped_movie_genres(book_row, self.genre_mapping)

            if not book_movie_genres:
                # Fallback to random negatives
                for movie_idx in movie_indices:
                    random_neg = np.random.choice([i for i in range(len(self.movie_ids))
                                                   if i not in movie_indices])
                    key = f"{book_idx}_{movie_idx}"
                    hard_negatives[key] = int(random_neg)
                    fallback_count += 1
                continue

            # For each positive movie, find a hard negative
            for movie_idx in movie_indices:
                hard_neg_idx = None

                # Search candidates for genre-incompatible movie
                for cand_idx in candidate_indices:
                    # Skip if it's the positive movie
                    if cand_idx == movie_idx:
                        continue

                    # Skip if it's another positive for this book
                    if cand_idx in movie_indices:
                        continue

                    # Get candidate movie's genres
                    movie_df_idx = self.movie_ids[cand_idx]
                    movie_row = self.movie_df.loc[movie_df_idx]
                    movie_genres_str = str(movie_row.get('genres', ''))

                    if movie_genres_str in ['', 'nan', '\\N']:
                        continue

                    movie_genres = [g.strip() for g in movie_genres_str.split(',')]

                    # Check if genre-incompatible
                    if not any(g in movie_genres for g in book_movie_genres):
                        hard_neg_idx = cand_idx
                        success_count += 1
                        break

                # Fallback to random if no hard negative found
                if hard_neg_idx is None:
                    available = [i for i in range(len(self.movie_ids))
                                if i not in movie_indices and i != movie_idx]
                    if available:
                        hard_neg_idx = np.random.choice(available)
                        fallback_count += 1
                    else:
                        # Very rare edge case
                        hard_neg_idx = (movie_idx + 1) % len(self.movie_ids)
                        fallback_count += 1

                key = f"{book_idx}_{movie_idx}"
                hard_negatives[key] = int(hard_neg_idx)

        self.logger.info(f"\n✓ Hard negative mining complete:")
        self.logger.info(f"  Total hard negatives: {len(hard_negatives)}")
        self.logger.info(f"  Successful mining: {success_count} ({success_count/len(hard_negatives)*100:.1f}%)")
        self.logger.info(f"  Fallback to random: {fallback_count} ({fallback_count/len(hard_negatives)*100:.1f}%)")

        return hard_negatives

    def save_pairs(self, pairs_data: Dict) -> None:
        """Save pairs to data/processed/contrastive_pairs.json"""
        output_path = PROCESSED_DATA_DIR / "contrastive_pairs.json"

        with timer("Saving pairs", self.logger):
            with open(output_path, 'w') as f:
                json.dump(pairs_data, f, indent=2)

        file_size_mb = output_path.stat().st_size / (1024**2)
        self.logger.info(f"  ✓ Saved to {output_path}")
        self.logger.info(f"  File size: {file_size_mb:.2f} MB")

    def print_stats(self, pairs_data: Dict) -> None:
        """Print dataset statistics"""
        self.logger.info("\n" + "="*70)
        self.logger.info("DATASET STATISTICS")
        self.logger.info("="*70)

        stats = pairs_data['stats']

        self.logger.info(f"\n📊 Pair Counts:")
        self.logger.info(f"  Total positive pairs: {stats['n_positive']:,}")
        self.logger.info(f"  Total negative pairs: {stats['n_negative']:,}")
        self.logger.info(f"  Positive/Negative ratio: {stats['pos_neg_ratio']:.2f}")

        self.logger.info(f"\n📚 Coverage:")
        self.logger.info(f"  Unique books: {stats['unique_books']:,} ({stats['book_coverage_pct']:.1f}% of {len(self.book_ids)})")
        self.logger.info(f"  Unique movies (positive): {stats['unique_movies_pos']:,} ({stats['movie_pos_coverage_pct']:.1f}% of {len(self.movie_ids)})")
        self.logger.info(f"  Unique movies (negative): {stats['unique_movies_neg']:,} ({stats['movie_neg_coverage_pct']:.1f}% of {len(self.movie_ids)})")

        self.logger.info(f"\n🎯 Quality Metrics:")
        self.logger.info(f"  Avg hard negative similarity: {stats['avg_hard_negative_similarity']:.4f}")
        self.logger.info(f"  Hard negative mining success rate: {stats['hard_negative_success_rate']:.1f}%")

    def compute_statistics(self, positive_pairs: List[Tuple[int, int]],
                          hard_negatives: Dict[str, int]) -> Dict:
        """Compute dataset statistics"""
        # Basic counts
        n_positive = len(positive_pairs)
        n_negative = len(hard_negatives)
        pos_neg_ratio = n_positive / n_negative if n_negative > 0 else 0

        # Coverage
        unique_books = len(set(book_idx for book_idx, _ in positive_pairs))
        unique_movies_pos = len(set(movie_idx for _, movie_idx in positive_pairs))
        unique_movies_neg = len(set(hard_negatives.values()))

        book_coverage_pct = (unique_books / len(self.book_ids)) * 100
        movie_pos_coverage_pct = (unique_movies_pos / len(self.movie_ids)) * 100
        movie_neg_coverage_pct = (unique_movies_neg / len(self.movie_ids)) * 100

        # Hard negative quality - sample for efficiency
        sample_size = min(1000, len(hard_negatives))
        sample_keys = np.random.choice(list(hard_negatives.keys()), sample_size, replace=False)

        similarities = []
        for key in sample_keys:
            book_idx, movie_pos_idx = map(int, key.split('_'))
            movie_neg_idx = hard_negatives[key]

            book_emb = self.book_embeddings[book_idx]
            movie_neg_emb = self.movie_embeddings[movie_neg_idx]

            # Cosine similarity (embeddings are L2-normalized)
            similarity = float(np.dot(book_emb, movie_neg_emb))
            similarities.append(similarity)

        avg_hard_negative_similarity = float(np.mean(similarities))

        # Count successful hard negatives (not fallback)
        # Approximate by checking if similarity > threshold
        hard_negative_success_rate = sum(1 for s in similarities if s > 0.3) / len(similarities) * 100

        return {
            'n_positive': n_positive,
            'n_negative': n_negative,
            'pos_neg_ratio': pos_neg_ratio,
            'unique_books': unique_books,
            'unique_movies_pos': unique_movies_pos,
            'unique_movies_neg': unique_movies_neg,
            'book_coverage_pct': book_coverage_pct,
            'movie_pos_coverage_pct': movie_pos_coverage_pct,
            'movie_neg_coverage_pct': movie_neg_coverage_pct,
            'avg_hard_negative_similarity': avg_hard_negative_similarity,
            'hard_negative_success_rate': hard_negative_success_rate
        }

    def run(self) -> None:
        """Main pipeline"""
        self.logger.info("\n" + "="*70)
        self.logger.info("CONTRASTIVE DATASET BUILDER")
        self.logger.info("="*70)

        # Step 1: Load data
        self.load_data()

        # Step 2: Build temporary FAISS index
        self.build_temp_faiss_index()

        # Step 3: Find positive pairs
        positive_pairs = self.find_positive_pairs(max_per_book=5)

        # Step 4: Find hard negatives
        hard_negatives = self.find_hard_negatives(positive_pairs, k=50)

        # Step 5: Compute statistics
        stats = self.compute_statistics(positive_pairs, hard_negatives)

        # Step 6: Prepare output data
        pairs_data = {
            'positive_pairs': positive_pairs,
            'hard_negatives': hard_negatives,
            'stats': stats,
            'metadata': {
                'n_books': len(self.book_ids),
                'n_movies': len(self.movie_ids),
                'embedding_dim': self.book_embeddings.shape[1],
                'max_per_book': 5,
                'faiss_k': 50
            }
        }

        # Step 7: Save
        self.save_pairs(pairs_data)

        # Step 8: Print statistics
        self.print_stats(pairs_data)

        self.logger.info("\n" + "="*70)
        self.logger.info("✓ CONTRASTIVE DATASET COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"\nOutput: {PROCESSED_DATA_DIR / 'contrastive_pairs.json'}")
        self.logger.info(f"Ready for training!")


if __name__ == "__main__":
    builder = ContrastivePairBuilder()
    builder.run()
