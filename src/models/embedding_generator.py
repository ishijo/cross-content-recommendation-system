"""
Generate SBERT embeddings for all books and movies.
Uses all-mpnet-base-v2 model with genre-aware input formatting.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ast

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from project_config import (
    EMBEDDINGS_DIR,
    GOODREADS_EMBEDDINGS_NPY,
    IMDB_EMBEDDINGS_NPY,
    SENTENCE_TRANSFORMER_MODEL,
    EMBEDDING_DIM,
    BATCH_SIZE
)
from utils.helpers import (
    setup_logger,
    timer,
    load_datasets,
    load_genre_mapping,
    get_mapped_movie_genres
)

# Constants
MAX_TOKENS = 512  # SBERT limit


class EmbeddingGenerator:
    """Generate and manage SBERT embeddings for books and movies"""

    def __init__(self):
        self.logger = setup_logger("embedding_generator")
        self.model = None
        self.genre_mapping = None

    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is not None:
            self.logger.info("Model already loaded")
            return

        with timer(f"Loading {SENTENCE_TRANSFORMER_MODEL}", self.logger):
            self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        self.logger.info(f"  Model dimension: {self.model.get_sentence_embedding_dimension()}")
        self.logger.info(f"  Max sequence length: {self.model.max_seq_length}")

    def build_input_string(
        self,
        description: str,
        genres: str,
        is_book: bool = True
    ) -> str:
        """
        Build input string with genre and description signals.

        Args:
            description: Text description
            genres: Genres (string representation of list for books, comma-separated for movies)
            is_book: Whether this is a book (affects genre parsing)

        Returns:
            Formatted string: "[GENRES] genre1, genre2 [DESC] description text..."
        """
        # Parse genres
        if is_book:
            try:
                genre_list = ast.literal_eval(genres) if isinstance(genres, str) else []
            except:
                genre_list = []
        else:
            # Movies have comma-separated genres
            genre_list = [g.strip() for g in str(genres).split(',') if g.strip()]

        genres_str = ', '.join(genre_list) if genre_list else 'unknown'

        # Build formatted string
        input_str = f"[GENRES] {genres_str} [DESC] {description}"

        # Truncate if too long (rough estimate: 1 token ≈ 4 chars)
        max_chars = MAX_TOKENS * 4
        if len(input_str) > max_chars:
            input_str = input_str[:max_chars]

        return input_str

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts in batches and normalize.

        Args:
            texts: List of input strings
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings

        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize  # L2 normalize for cosine similarity
        )

        return embeddings

    def process_books(self, goodreads_df: pd.DataFrame) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
        """
        Process books: drop nulls, build inputs, encode.

        Args:
            goodreads_df: Goodreads dataframe

        Returns:
            Tuple of (embeddings, book_ids, filtered_df)
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PROCESSING BOOKS")
        self.logger.info("="*70)

        # Drop books with empty descriptions
        original_count = len(goodreads_df)
        goodreads_df = goodreads_df[goodreads_df['Description'].str.strip() != ''].copy()
        dropped = original_count - len(goodreads_df)

        if dropped > 0:
            self.logger.warning(f"Dropped {dropped} books with null/empty descriptions ({dropped/original_count*100:.1f}%)")

        self.logger.info(f"Processing {len(goodreads_df):,} books")

        # Calculate description stats
        desc_lengths = goodreads_df['Description'].str.len()
        avg_length = desc_lengths.mean()
        self.logger.info(f"  Average description length: {avg_length:.0f} chars")
        self.logger.info(f"  Range: {desc_lengths.min():.0f} - {desc_lengths.max():.0f} chars")

        # Build input strings
        with timer("Building input strings for books", self.logger):
            input_strings = []
            for _, row in tqdm(goodreads_df.iterrows(), total=len(goodreads_df), desc="Building inputs"):
                input_str = self.build_input_string(
                    row['Description'],
                    row['Genres'],
                    is_book=True
                )
                input_strings.append(input_str)

        # Encode
        with timer(f"Encoding {len(input_strings):,} books", self.logger):
            embeddings = self.encode_batch(input_strings)

        self.logger.info(f"  Embeddings shape: {embeddings.shape}")
        self.logger.info(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f} (should be ~1.0)")

        # Get book IDs (using index)
        book_ids = goodreads_df.index.tolist()

        return embeddings, book_ids, goodreads_df

    def process_movies(self, imdb_df: pd.DataFrame) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
        """
        Process movies: drop nulls, build inputs, encode.

        Args:
            imdb_df: IMDB dataframe

        Returns:
            Tuple of (embeddings, movie_ids, filtered_df)
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PROCESSING MOVIES")
        self.logger.info("="*70)

        # Drop movies with empty descriptions
        original_count = len(imdb_df)
        imdb_df = imdb_df[imdb_df['Description'].str.strip() != ''].copy()
        dropped = original_count - len(imdb_df)

        if dropped > 0:
            self.logger.warning(f"Dropped {dropped} movies with null/empty descriptions ({dropped/original_count*100:.1f}%)")

        self.logger.info(f"Processing {len(imdb_df):,} movies")

        # Calculate description stats
        desc_lengths = imdb_df['Description'].str.len()
        avg_length = desc_lengths.mean()
        self.logger.info(f"  Average description length: {avg_length:.0f} chars")
        self.logger.info(f"  Range: {desc_lengths.min():.0f} - {desc_lengths.max():.0f} chars")

        # Build input strings
        with timer("Building input strings for movies", self.logger):
            input_strings = []
            for _, row in tqdm(imdb_df.iterrows(), total=len(imdb_df), desc="Building inputs"):
                input_str = self.build_input_string(
                    row['Description'],
                    row['genres'],
                    is_book=False
                )
                input_strings.append(input_str)

        # Encode
        with timer(f"Encoding {len(input_strings):,} movies", self.logger):
            embeddings = self.encode_batch(input_strings)

        self.logger.info(f"  Embeddings shape: {embeddings.shape}")
        self.logger.info(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f} (should be ~1.0)")

        # Get movie IDs (using index)
        movie_ids = imdb_df.index.tolist()

        return embeddings, movie_ids, imdb_df

    def compute_similarity_stats(
        self,
        book_embeddings: np.ndarray,
        movie_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute within-domain and cross-domain similarity statistics.

        Args:
            book_embeddings: Book embeddings array
            movie_embeddings: Movie embeddings array

        Returns:
            Dictionary with similarity statistics
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPUTING SIMILARITY STATISTICS")
        self.logger.info("="*70)

        stats = {}

        # Sample for efficiency (computing full pairwise is expensive)
        sample_size = min(500, len(book_embeddings), len(movie_embeddings))
        book_sample = book_embeddings[np.random.choice(len(book_embeddings), sample_size, replace=False)]
        movie_sample = movie_embeddings[np.random.choice(len(movie_embeddings), sample_size, replace=False)]

        # Within-book similarity
        with timer("Computing within-book similarity", self.logger):
            book_similarities = np.dot(book_sample, book_sample.T)
            # Exclude diagonal (self-similarity = 1.0)
            mask = ~np.eye(sample_size, dtype=bool)
            book_sim_mean = book_similarities[mask].mean()
            book_sim_std = book_similarities[mask].std()

        stats['within_book_mean'] = float(book_sim_mean)
        stats['within_book_std'] = float(book_sim_std)
        self.logger.info(f"  Within-book similarity: {book_sim_mean:.4f} ± {book_sim_std:.4f}")

        # Within-movie similarity
        with timer("Computing within-movie similarity", self.logger):
            movie_similarities = np.dot(movie_sample, movie_sample.T)
            mask = ~np.eye(sample_size, dtype=bool)
            movie_sim_mean = movie_similarities[mask].mean()
            movie_sim_std = movie_similarities[mask].std()

        stats['within_movie_mean'] = float(movie_sim_mean)
        stats['within_movie_std'] = float(movie_sim_std)
        self.logger.info(f"  Within-movie similarity: {movie_sim_mean:.4f} ± {movie_sim_std:.4f}")

        # Cross-domain similarity
        with timer("Computing cross-domain similarity", self.logger):
            cross_similarities = np.dot(book_sample, movie_sample.T)
            cross_sim_mean = cross_similarities.mean()
            cross_sim_std = cross_similarities.std()

        stats['cross_domain_mean'] = float(cross_sim_mean)
        stats['cross_domain_std'] = float(cross_sim_std)
        self.logger.info(f"  Cross-domain similarity: {cross_sim_mean:.4f} ± {cross_sim_std:.4f}")

        # Compute the gap
        within_domain_mean = (book_sim_mean + movie_sim_mean) / 2
        gap = within_domain_mean - cross_sim_mean
        gap_pct = (gap / within_domain_mean) * 100

        stats['similarity_gap'] = float(gap)
        stats['similarity_gap_percent'] = float(gap_pct)

        self.logger.info(f"\n  🎯 SIMILARITY GAP: {gap:.4f} ({gap_pct:.1f}%)")
        self.logger.info(f"     Within-domain avg: {within_domain_mean:.4f}")
        self.logger.info(f"     Cross-domain: {cross_sim_mean:.4f}")
        self.logger.info(f"     → This gap motivates Sprint 3 (contrastive learning)")

        return stats

    def save_embeddings(
        self,
        book_embeddings: np.ndarray,
        book_ids: List[int],
        movie_embeddings: np.ndarray,
        movie_ids: List[int],
        book_df: pd.DataFrame,
        movie_df: pd.DataFrame,
        similarity_stats: Dict[str, float]
    ):
        """
        Save all embeddings and metadata.

        Args:
            book_embeddings: Book embeddings array
            book_ids: List of book IDs
            movie_embeddings: Movie embeddings array
            movie_ids: List of movie IDs
            book_df: Book dataframe (for metadata)
            movie_df: Movie dataframe (for metadata)
            similarity_stats: Similarity statistics
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("SAVING EMBEDDINGS")
        self.logger.info("="*70)

        # Save embeddings
        book_emb_path = EMBEDDINGS_DIR / "book_embeddings.npy"
        np.save(book_emb_path, book_embeddings)
        self.logger.info(f"  ✓ Saved book embeddings: {book_emb_path}")

        movie_emb_path = EMBEDDINGS_DIR / "movie_embeddings.npy"
        np.save(movie_emb_path, movie_embeddings)
        self.logger.info(f"  ✓ Saved movie embeddings: {movie_emb_path}")

        # Save IDs
        book_ids_path = EMBEDDINGS_DIR / "book_ids.json"
        with open(book_ids_path, 'w') as f:
            json.dump(book_ids, f)
        self.logger.info(f"  ✓ Saved book IDs: {book_ids_path}")

        movie_ids_path = EMBEDDINGS_DIR / "movie_ids.json"
        with open(movie_ids_path, 'w') as f:
            json.dump(movie_ids, f)
        self.logger.info(f"  ✓ Saved movie IDs: {movie_ids_path}")

        # Save metadata
        metadata = {
            'model_name': SENTENCE_TRANSFORMER_MODEL,
            'embedding_dim': EMBEDDING_DIM,
            'n_books': len(book_ids),
            'n_movies': len(movie_ids),
            'date_generated': datetime.now().isoformat(),
            'avg_book_description_length': float(book_df['Description'].str.len().mean()),
            'avg_movie_description_length': float(movie_df['Description'].str.len().mean()),
            'similarity_stats': similarity_stats
        }

        # Check for description length asymmetry
        length_ratio = metadata['avg_book_description_length'] / metadata['avg_movie_description_length']
        if length_ratio > 2.0 or length_ratio < 0.5:
            self.logger.warning(f"\n  ⚠️  Description length asymmetry detected!")
            self.logger.warning(f"     Books avg: {metadata['avg_book_description_length']:.0f} chars")
            self.logger.warning(f"     Movies avg: {metadata['avg_movie_description_length']:.0f} chars")
            self.logger.warning(f"     Ratio: {length_ratio:.2f}x")
            self.logger.warning(f"     → This is a known limitation to address in Sprint 3")

        metadata_path = EMBEDDINGS_DIR / "embedding_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"  ✓ Saved metadata: {metadata_path}")

    def check_existing_embeddings(self) -> bool:
        """
        Check if embeddings already exist.

        Returns:
            True if embeddings exist and should be skipped
        """
        book_exists = (EMBEDDINGS_DIR / "book_embeddings.npy").exists()
        movie_exists = (EMBEDDINGS_DIR / "movie_embeddings.npy").exists()
        metadata_exists = (EMBEDDINGS_DIR / "embedding_metadata.json").exists()

        if book_exists and movie_exists and metadata_exists:
            self.logger.info("✓ Embeddings already exist")
            self.logger.info("  To regenerate, delete files in data/embeddings/")

            # Load and show metadata
            with open(EMBEDDINGS_DIR / "embedding_metadata.json", 'r') as f:
                metadata = json.load(f)

            self.logger.info(f"\n  Generated: {metadata['date_generated']}")
            self.logger.info(f"  Model: {metadata['model_name']}")
            self.logger.info(f"  Books: {metadata['n_books']:,}")
            self.logger.info(f"  Movies: {metadata['n_movies']:,}")

            if 'similarity_stats' in metadata:
                stats = metadata['similarity_stats']
                self.logger.info(f"\n  Similarity Stats:")
                self.logger.info(f"    Within-book: {stats['within_book_mean']:.4f}")
                self.logger.info(f"    Within-movie: {stats['within_movie_mean']:.4f}")
                self.logger.info(f"    Cross-domain: {stats['cross_domain_mean']:.4f}")
                self.logger.info(f"    Gap: {stats['similarity_gap']:.4f} ({stats['similarity_gap_percent']:.1f}%)")

            return True

        return False

    def run(self):
        """Main execution pipeline"""
        self.logger.info("\n" + "="*70)
        self.logger.info("EMBEDDING GENERATION PIPELINE")
        self.logger.info("="*70)

        # Check if embeddings already exist
        if self.check_existing_embeddings():
            self.logger.info("\n✓ Using existing embeddings")
            return

        # Load model
        self.load_model()

        # Load data
        with timer("Loading datasets", self.logger):
            goodreads_df, imdb_df = load_datasets(self.logger)

        # Process books
        book_embeddings, book_ids, book_df = self.process_books(goodreads_df)

        # Process movies
        movie_embeddings, movie_ids, movie_df = self.process_movies(imdb_df)

        # Compute similarity statistics
        similarity_stats = self.compute_similarity_stats(book_embeddings, movie_embeddings)

        # Save everything
        self.save_embeddings(
            book_embeddings,
            book_ids,
            movie_embeddings,
            movie_ids,
            book_df,
            movie_df,
            similarity_stats
        )

        self.logger.info("\n" + "="*70)
        self.logger.info("✅ EMBEDDING GENERATION COMPLETE")
        self.logger.info("="*70)


def main():
    """Entry point"""
    generator = EmbeddingGenerator()
    generator.run()


if __name__ == "__main__":
    main()
