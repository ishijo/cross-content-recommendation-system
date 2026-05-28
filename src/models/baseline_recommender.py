"""
Baseline recommendation system using embeddings + genre filtering.
This is the baseline before contrastive learning (Sprint 3).
"""
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from project_config import (
    EMBEDDINGS_DIR,
    GOODREADS_RAW_CSV,
    IMDB_FILTERED_CSV,
    SENTENCE_TRANSFORMER_MODEL
)
from utils.helpers import (
    setup_logger,
    timer,
    load_genre_mapping,
    get_mapped_movie_genres,
    calculate_genre_overlap
)
from models.faiss_indexer import FAISSIndexer


class BaselineRecommender:
    """Baseline recommender using content similarity + genre filtering"""

    def __init__(self):
        self.logger = setup_logger("baseline_recommender")
        self.model = None
        self.indexer = None
        self.book_df = None
        self.movie_df = None
        self.book_embeddings = None
        self.book_ids = None
        self.genre_mapping = None

    def load_data(self):
        """Load all required data"""
        self.logger.info("\n" + "="*70)
        self.logger.info("LOADING DATA")
        self.logger.info("="*70)

        # Load model
        with timer("Loading sentence transformer model", self.logger):
            self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        # Load FAISS indexer
        with timer("Loading FAISS index", self.logger):
            self.indexer = FAISSIndexer()
            self.indexer.load_all()

        # Load book data
        with timer("Loading book data", self.logger):
            self.book_df = pd.read_csv(GOODREADS_RAW_CSV, index_col=0)
            self.logger.info(f"  Loaded {len(self.book_df):,} books")

        # Load movie data
        with timer("Loading movie data", self.logger):
            self.movie_df = pd.read_csv(IMDB_FILTERED_CSV, index_col=0)
            self.logger.info(f"  Loaded {len(self.movie_df):,} movies")

        # Load book embeddings
        with timer("Loading book embeddings", self.logger):
            emb_path = EMBEDDINGS_DIR / "book_embeddings.npy"
            self.book_embeddings = np.load(emb_path)
            self.logger.info(f"  Loaded embeddings: {self.book_embeddings.shape}")

        # Load book IDs
        with open(EMBEDDINGS_DIR / "book_ids.json", 'r') as f:
            self.book_ids = json.load(f)

        # Load genre mapping
        self.genre_mapping = load_genre_mapping(self.logger)

        self.logger.info("\n✓ All data loaded successfully")

    def get_book_embedding(self, book_title: str) -> Optional[np.ndarray]:
        """
        Get embedding for a book by title.
        If not found, encodes the title on the fly.

        Args:
            book_title: Book title to look up

        Returns:
            Embedding vector (1D array) or None if not found
        """
        # Try exact match first
        matches = self.book_df[self.book_df['Book'].str.lower() == book_title.lower()]

        if len(matches) > 0:
            book_id = matches.index[0]

            # Find embedding index
            if book_id in self.book_ids:
                emb_idx = self.book_ids.index(book_id)
                return self.book_embeddings[emb_idx]

        # Try partial match
        matches = self.book_df[self.book_df['Book'].str.lower().str.contains(book_title.lower())]

        if len(matches) > 0:
            book_id = matches.index[0]
            if book_id in self.book_ids:
                emb_idx = self.book_ids.index(book_id)
                self.logger.info(f"  Found partial match: {matches.iloc[0]['Book']}")
                return self.book_embeddings[emb_idx]

        # If not found, encode on the fly
        self.logger.warning(f"  Book not found: '{book_title}'. Encoding title on the fly.")
        embedding = self.model.encode(
            [book_title],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        return embedding

    def get_book_info(self, book_title: str) -> Optional[pd.Series]:
        """
        Get book information by title.

        Args:
            book_title: Book title to look up

        Returns:
            Pandas Series with book info or None
        """
        matches = self.book_df[self.book_df['Book'].str.lower() == book_title.lower()]

        if len(matches) == 0:
            # Try partial match
            matches = self.book_df[self.book_df['Book'].str.lower().str.contains(book_title.lower())]

        if len(matches) > 0:
            return matches.iloc[0]

        return None

    def recommend_movies(
        self,
        book_title: str,
        k: int = 10,
        genre_filter: bool = True,
        genre_boost: float = 0.2
    ) -> pd.DataFrame:
        """
        Recommend movies based on a book.

        Args:
            book_title: Title of the book
            k: Number of recommendations to return
            genre_filter: Whether to apply genre-based re-ranking
            genre_boost: Weight for genre overlap bonus (0.0 to 1.0)

        Returns:
            Dataframe with columns: [rank, movie_id, movie_title, year, genres, rating,
                                     similarity_score, genre_bonus, final_score, description]
        """
        # Get book embedding
        book_embedding = self.get_book_embedding(book_title)
        if book_embedding is None:
            self.logger.error(f"Could not get embedding for: {book_title}")
            return pd.DataFrame()

        # Get book info for genre filtering
        book_info = self.get_book_info(book_title)

        # Search FAISS for candidates (get more than k for genre filtering)
        n_candidates = 50 if genre_filter else k
        candidates = self.indexer.search(book_embedding, k=n_candidates)

        # Build results list
        results = []
        for movie_id, similarity in candidates:
            movie_info = self.indexer.get_movie_info(movie_id)

            # Calculate genre overlap if filtering enabled
            genre_bonus = 0.0
            if genre_filter and book_info is not None:
                # Get mapped movie genres for this book
                book_movie_genres = get_mapped_movie_genres(book_info, self.genre_mapping)

                # Get actual movie genres
                movie_genres = [g.strip() for g in str(movie_info['genres']).split(',')]

                # Calculate overlap
                genre_overlap = calculate_genre_overlap(book_movie_genres, movie_genres)
                genre_bonus = genre_overlap * genre_boost

            final_score = similarity + genre_bonus

            results.append({
                'movie_id': movie_id,
                'movie_title': movie_info['title'],
                'genres': movie_info['genres'],
                'year': movie_info['year'],
                'rating': movie_info['rating'],
                'similarity_score': similarity,
                'genre_bonus': genre_bonus,
                'final_score': final_score,
                'description': movie_info['description']
            })

        # Sort by final score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False)

        # Take top k
        results_df = results_df.head(k).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)

        # Reorder columns (keep movie_id for evaluation)
        results_df = results_df[[
            'rank', 'movie_id', 'movie_title', 'year', 'genres', 'rating',
            'similarity_score', 'genre_bonus', 'final_score', 'description'
        ]]

        return results_df

    def print_recommendations(self, book_title: str, recommendations: pd.DataFrame):
        """
        Pretty print recommendations.

        Args:
            book_title: Book title
            recommendations: Recommendations dataframe
        """
        print("\n" + "="*70)
        print(f"RECOMMENDATIONS FOR: {book_title}")
        print("="*70)

        if len(recommendations) == 0:
            print("No recommendations found.")
            return

        for _, row in recommendations.iterrows():
            print(f"\n{row['rank']}. {row['movie_title']} ({row['year']}) - ⭐ {row['rating']:.1f}")
            print(f"   Genres: {row['genres']}")
            print(f"   Scores: Similarity={row['similarity_score']:.3f}, "
                  f"Genre Bonus={row['genre_bonus']:.3f}, Final={row['final_score']:.3f}")
            print(f"   {row['description']}")

    def test_diverse_books(self):
        """
        Test recommendations with 5 diverse books.
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("TESTING WITH DIVERSE BOOKS")
        self.logger.info("="*70)

        # 5 diverse test books
        test_books = [
            "Harry Potter and the Sorcerer's Stone",  # Fantasy
            "Gone Girl",                               # Thriller
            "Pride and Prejudice",                     # Romance
            "The Martian",                             # Sci-Fi
            "To Kill a Mockingbird"                    # Literary Fiction
        ]

        for book_title in test_books:
            recommendations = self.recommend_movies(book_title, k=5, genre_filter=True)
            self.print_recommendations(book_title, recommendations)

    def run(self):
        """Main execution pipeline"""
        self.logger.info("\n" + "="*70)
        self.logger.info("BASELINE RECOMMENDER")
        self.logger.info("="*70)

        # Load data
        self.load_data()

        # Test with diverse books
        self.test_diverse_books()

        self.logger.info("\n" + "="*70)
        self.logger.info("✅ BASELINE RECOMMENDER TESTING COMPLETE")
        self.logger.info("="*70)


def main():
    """Entry point"""
    recommender = BaselineRecommender()
    recommender.run()


if __name__ == "__main__":
    main()
