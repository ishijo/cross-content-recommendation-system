"""
Contrastive Recommender - Extends baseline with projection head.

Adds contrastive learning projection to map 768-dim embeddings
to 128-dim aligned space for improved cross-domain recommendations.

Supports bidirectional recommendations:
  - Book  → Movies/Shows  (recommend_movies)
  - Movie/Show → Books    (recommend_books_from_movie)

Usage:
    python src/models/contrastive_recommender.py
"""

import ast
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import json
import numpy as np
import pandas as pd
import torch
import faiss
from typing import Optional, Tuple
from tqdm import tqdm

from project_config import (
    EMBEDDINGS_DIR,
    PROJECTION_HEAD_DIR,
    BOOK_FAISS_INDEX,
    BOOK_FAISS_PROJECTED_INDEX,
    BOOK_EMBEDDINGS_PROJECTED_NPY,
)
from utils.helpers import (
    setup_logger,
    timer,
    get_mapped_movie_genres,
    calculate_genre_overlap,
)
from models.baseline_recommender import BaselineRecommender
from models.projection_head import ProjectionHead


class ContrastiveRecommender(BaselineRecommender):
    """
    Extends BaselineRecommender with projection head for contrastive recommendations.

    Maintains backward compatibility — can toggle between baseline and contrastive modes.
    Supports both directions:
        book → movies/shows   via recommend_movies()
        movie/show → books    via recommend_books_from_movie()
    """

    def __init__(self):
        super().__init__()
        self.projection_head = None
        self.projected_movie_index = None      # 128-dim FAISS IndexFlatIP (movies)
        self.projected_movie_embeddings = None  # (n_movies, 128)
        self.movie_embeddings = None            # (n_movies, 768) — raw
        self.movie_ids = None                   # list: emb_idx → movie_id

        # Sprint 4A — book-side indices
        self.book_baseline_index = None         # 768-dim FAISS IndexFlatIP (books)
        self.projected_book_index = None        # 128-dim FAISS IndexFlatIP (books)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    #  Projection head                                                     #
    # ------------------------------------------------------------------ #

    def load_projection_head(self, model_path: Path = None):
        """
        Load trained projection head from checkpoint.

        Args:
            model_path: Path to checkpoint (default: models/projection_head/best_model.pt)
        """
        if model_path is None:
            model_path = PROJECTION_HEAD_DIR / "best_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Projection head checkpoint not found: {model_path}"
            )

        self.logger.info("\n" + "=" * 70)
        self.logger.info("LOADING PROJECTION HEAD")
        self.logger.info("=" * 70)

        with timer("Loading checkpoint", self.logger):
            checkpoint = torch.load(model_path, map_location=self.device)

        self.projection_head = ProjectionHead(
            input_dim=768,
            hidden_dims=[512, 256],
            output_dim=128,
            dropout=0.2,
        ).to(self.device)

        self.projection_head.load_state_dict(checkpoint["model_state_dict"])
        self.projection_head.eval()

        self.logger.info(f"✓ Loaded from: {model_path}")
        self.logger.info(f"  Trained for {checkpoint['epoch']} epochs")
        self.logger.info(f"  Best val loss: {checkpoint['val_loss']:.4f}")
        self.logger.info(f"  Device: {self.device}")

    def project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project a 768-dim embedding to 128-dim via the trained head.

        Args:
            embedding: (768,) numpy array

        Returns:
            (128,) numpy array, L2-normalised
        """
        if self.projection_head is None:
            raise RuntimeError(
                "Projection head not loaded. Call load_projection_head() first."
            )
        with torch.no_grad():
            tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)
            projected = self.projection_head(tensor)
            return projected.squeeze(0).cpu().numpy()

    # Backward-compat alias
    def project_book(self, book_embedding: np.ndarray) -> np.ndarray:
        """Alias for project_embedding — kept for backward compatibility."""
        return self.project_embedding(book_embedding)

    # ------------------------------------------------------------------ #
    #  Movie embeddings / projected movie index                            #
    # ------------------------------------------------------------------ #

    def load_movie_embeddings(self):
        """Load raw movie embeddings (768-dim) and movie IDs from disk."""
        if self.movie_embeddings is None:
            with timer("Loading movie embeddings", self.logger):
                movie_emb_path = EMBEDDINGS_DIR / "movie_embeddings.npy"
                self.movie_embeddings = np.load(movie_emb_path)
                self.logger.info(f"  Shape: {self.movie_embeddings.shape}")

        if self.movie_ids is None:
            with open(EMBEDDINGS_DIR / "movie_ids.json") as f:
                self.movie_ids = json.load(f)

    def build_projected_movie_index(self, batch_size: int = 256):
        """
        Project all movies to 128-dim and build FAISS index.

        Saves movie_faiss_projected.index to EMBEDDINGS_DIR.

        Args:
            batch_size: Batch size for GPU/CPU projection
        """
        if self.projection_head is None:
            raise RuntimeError(
                "Projection head not loaded. Call load_projection_head() first."
            )
        self.load_movie_embeddings()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("BUILDING PROJECTED MOVIE INDEX")
        self.logger.info("=" * 70)

        n_movies = len(self.movie_embeddings)
        projected_embeddings = []

        with timer("Projecting movie embeddings", self.logger):
            with torch.no_grad():
                for i in tqdm(range(0, n_movies, batch_size), desc="Projecting movies"):
                    batch = self.movie_embeddings[i : i + batch_size]
                    tensor = torch.from_numpy(batch).float().to(self.device)
                    projected = self.projection_head(tensor)
                    projected_embeddings.append(projected.cpu().numpy())

        self.projected_movie_embeddings = np.vstack(projected_embeddings)
        self.logger.info(f"  Projected shape: {self.projected_movie_embeddings.shape}")

        with timer("Building FAISS index (128-dim, movies)", self.logger):
            self.projected_movie_index = faiss.IndexFlatIP(128)
            self.projected_movie_index.add(
                self.projected_movie_embeddings.astype("float32")
            )
            self.logger.info(
                f"  ✓ Index built: {self.projected_movie_index.ntotal} movies"
            )

        index_path = EMBEDDINGS_DIR / "movie_faiss_projected.index"
        with timer("Saving projected movie index", self.logger):
            faiss.write_index(self.projected_movie_index, str(index_path))
            self.logger.info(f"  ✓ Saved: {index_path}")

    def load_projected_index(self):
        """Load pre-built projected movie FAISS index (builds if missing)."""
        index_path = EMBEDDINGS_DIR / "movie_faiss_projected.index"

        if not index_path.exists():
            self.logger.warning(f"Projected movie index not found: {index_path}")
            self.logger.info("Building index now...")
            self.build_projected_movie_index()
            return

        with timer("Loading projected movie index", self.logger):
            self.projected_movie_index = faiss.read_index(str(index_path))
            self.logger.info(
                f"  ✓ Loaded: {self.projected_movie_index.ntotal} movies (128-dim)"
            )

        self.load_movie_embeddings()

    # ------------------------------------------------------------------ #
    #  Sprint 4A — book FAISS indices                                      #
    # ------------------------------------------------------------------ #

    def build_book_baseline_index(self):
        """
        Build a 768-dim FAISS IndexFlatIP over all book embeddings.

        Saves to BOOK_FAISS_INDEX. Requires load_data() to have been called.
        """
        if self.book_embeddings is None:
            raise RuntimeError(
                "Book embeddings not loaded. Call load_data() first."
            )

        self.logger.info("\n" + "=" * 70)
        self.logger.info("BUILDING BASELINE BOOK INDEX (768-dim)")
        self.logger.info("=" * 70)
        self.logger.info(f"  Input: {self.book_embeddings.shape}")

        with timer("Building book FAISS index (768-dim)", self.logger):
            self.book_baseline_index = faiss.IndexFlatIP(768)
            self.book_baseline_index.add(self.book_embeddings.astype("float32"))
            self.logger.info(
                f"  ✓ Index built: {self.book_baseline_index.ntotal} books"
            )

        with timer("Saving baseline book index", self.logger):
            faiss.write_index(self.book_baseline_index, str(BOOK_FAISS_INDEX))
            self.logger.info(f"  ✓ Saved: {BOOK_FAISS_INDEX}")

    def build_projected_book_index(self, batch_size: int = 256):
        """
        Project all book embeddings to 128-dim and build FAISS index.

        Saves:
        - BOOK_FAISS_PROJECTED_INDEX  (128-dim FAISS index)
        - BOOK_EMBEDDINGS_PROJECTED_NPY  (128-dim embedding array)

        This mirrors build_projected_movie_index() for the book side.

        Args:
            batch_size: Batch size for projection
        """
        if self.projection_head is None:
            raise RuntimeError(
                "Projection head not loaded. Call load_projection_head() first."
            )
        if self.book_embeddings is None:
            raise RuntimeError(
                "Book embeddings not loaded. Call load_data() first."
            )

        self.logger.info("\n" + "=" * 70)
        self.logger.info("BUILDING PROJECTED BOOK INDEX (128-dim)")
        self.logger.info("=" * 70)
        self.logger.info(f"  Input: {self.book_embeddings.shape}")

        n_books = len(self.book_embeddings)
        projected_embeddings = []

        with timer("Projecting book embeddings", self.logger):
            with torch.no_grad():
                for i in tqdm(range(0, n_books, batch_size), desc="Projecting books"):
                    batch = self.book_embeddings[i : i + batch_size]
                    tensor = torch.from_numpy(batch).float().to(self.device)
                    projected = self.projection_head(tensor)
                    projected_embeddings.append(projected.cpu().numpy())

        projected_arr = np.vstack(projected_embeddings)
        self.logger.info(f"  Projected shape: {projected_arr.shape}")

        with timer("Saving projected book embeddings", self.logger):
            np.save(BOOK_EMBEDDINGS_PROJECTED_NPY, projected_arr)
            self.logger.info(f"  ✓ Saved: {BOOK_EMBEDDINGS_PROJECTED_NPY}")

        with timer("Building FAISS index (128-dim, books)", self.logger):
            self.projected_book_index = faiss.IndexFlatIP(128)
            self.projected_book_index.add(projected_arr.astype("float32"))
            self.logger.info(
                f"  ✓ Index built: {self.projected_book_index.ntotal} books"
            )

        with timer("Saving projected book index", self.logger):
            faiss.write_index(self.projected_book_index, str(BOOK_FAISS_PROJECTED_INDEX))
            self.logger.info(f"  ✓ Saved: {BOOK_FAISS_PROJECTED_INDEX}")

    def load_book_baseline_index(self):
        """Load or build the 768-dim baseline book FAISS index."""
        if not BOOK_FAISS_INDEX.exists():
            self.logger.warning(f"Baseline book index not found: {BOOK_FAISS_INDEX}")
            self.logger.info("Building now...")
            self.build_book_baseline_index()
            return

        with timer("Loading baseline book index", self.logger):
            self.book_baseline_index = faiss.read_index(str(BOOK_FAISS_INDEX))
            self.logger.info(
                f"  ✓ Loaded: {self.book_baseline_index.ntotal} books (768-dim)"
            )

    def load_projected_book_index(self):
        """Load or build the 128-dim projected book FAISS index."""
        if not BOOK_FAISS_PROJECTED_INDEX.exists():
            self.logger.warning(
                f"Projected book index not found: {BOOK_FAISS_PROJECTED_INDEX}"
            )
            self.logger.info("Building now...")
            self.build_projected_book_index()
            return

        with timer("Loading projected book index", self.logger):
            self.projected_book_index = faiss.read_index(
                str(BOOK_FAISS_PROJECTED_INDEX)
            )
            self.logger.info(
                f"  ✓ Loaded: {self.projected_book_index.ntotal} books (128-dim)"
            )

    # ------------------------------------------------------------------ #
    #  Movie lookup helper                                                  #
    # ------------------------------------------------------------------ #

    def _find_movie_by_title(self, movie_title: str) -> Tuple[pd.Series, int]:
        """
        Find a movie/show in the catalog by title (case-insensitive, partial match).

        Args:
            movie_title: Title to search for

        Returns:
            (movie_row, emb_idx) — the matched row and its embedding array index

        Raises:
            ValueError: If title not found or has no embedding
        """
        if self.movie_ids is None:
            self.load_movie_embeddings()

        movie_df = self.movie_df

        # Exact match
        mask = movie_df["primaryTitle"].str.lower() == movie_title.lower()
        matches = movie_df[mask]

        if len(matches) == 0:
            # Partial match
            mask = movie_df["primaryTitle"].str.lower().str.contains(
                movie_title.lower(), na=False, regex=False
            )
            matches = movie_df[mask]

        if len(matches) == 0:
            raise ValueError(
                f"Movie/show '{movie_title}' not found in catalog. "
                "Note: dataset covers titles up to 2022."
            )

        movie_row = matches.iloc[0]
        movie_id = matches.index[0]

        if movie_id not in self.movie_ids:
            raise ValueError(
                f"Movie '{movie_row['primaryTitle']}' has no pre-computed embedding."
            )

        emb_idx = self.movie_ids.index(movie_id)
        return movie_row, emb_idx

    # ------------------------------------------------------------------ #
    #  recommend_movies — book → movies/shows                              #
    # ------------------------------------------------------------------ #

    def recommend_movies(
        self,
        book_title: str,
        k: int = 10,
        genre_filter: bool = True,
        genre_boost: float = 0.2,
        use_projection: bool = False,
    ) -> pd.DataFrame:
        """
        Recommend movies/shows for a given book.

        Args:
            book_title: Title of the book
            k: Number of recommendations
            genre_filter: Whether to apply genre-based re-ranking
            genre_boost: Genre overlap weight (0.0–1.0)
            use_projection: If True use 128-dim contrastive space; else use baseline

        Returns:
            DataFrame: [rank, movie_id, movie_title, year, genres, rating,
                        similarity_score, genre_bonus, final_score, description]
        """
        if not use_projection:
            return super().recommend_movies(book_title, k, genre_filter, genre_boost)

        if self.projection_head is None:
            self.logger.warning("Projection head not loaded — falling back to baseline.")
            return super().recommend_movies(book_title, k, genre_filter, genre_boost)

        if self.projected_movie_index is None:
            self.logger.warning("Projected movie index not loaded — loading...")
            try:
                self.load_projected_index()
            except Exception as e:
                self.logger.error(f"Failed: {e}")
                return super().recommend_movies(book_title, k, genre_filter, genre_boost)

        book_embedding = self.get_book_embedding(book_title)
        if book_embedding is None:
            self.logger.error(f"No embedding for: {book_title}")
            return pd.DataFrame()

        projected_book = self.project_embedding(book_embedding)
        book_info = self.get_book_info(book_title)

        n_candidates = 50 if genre_filter else k
        scores, indices = self.projected_movie_index.search(
            projected_book.reshape(1, -1).astype("float32"), n_candidates
        )

        results = []
        for idx, score in zip(indices[0], scores[0]):
            movie_id = self.movie_ids[idx]
            movie_info = self.indexer.get_movie_info(movie_id)

            genre_bonus = 0.0
            if genre_filter and book_info is not None:
                book_movie_genres = get_mapped_movie_genres(book_info, self.genre_mapping)
                movie_genres = [g.strip() for g in str(movie_info["genres"]).split(",")]
                genre_overlap = calculate_genre_overlap(book_movie_genres, movie_genres)
                genre_bonus = genre_overlap * genre_boost

            results.append(
                {
                    "movie_id": movie_id,
                    "movie_title": movie_info["title"],
                    "genres": movie_info["genres"],
                    "year": movie_info["year"],
                    "rating": movie_info["rating"],
                    "similarity_score": float(score),
                    "genre_bonus": genre_bonus,
                    "final_score": float(score) + genre_bonus,
                    "description": movie_info["description"],
                }
            )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("final_score", ascending=False)
        results_df = results_df.head(k).reset_index(drop=True)
        results_df["rank"] = range(1, len(results_df) + 1)
        results_df = results_df[
            [
                "rank", "movie_id", "movie_title", "year", "genres", "rating",
                "similarity_score", "genre_bonus", "final_score", "description",
            ]
        ]
        return results_df

    # ------------------------------------------------------------------ #
    #  recommend_books_from_movie — movie/show → books  (Sprint 4A)        #
    # ------------------------------------------------------------------ #

    def recommend_books_from_movie(
        self,
        movie_title: str,
        k: int = 10,
        use_projection: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend books for a given movie or TV show.

        Steps:
        1. Look up movie by title (case-insensitive, partial match OK).
        2. Retrieve its 768-dim embedding.
        3. If use_projection=True: project to 128-dim and search book_faiss_projected.index.
           Else: search book_faiss.index (768-dim baseline).
        4. Re-rank top-50 candidates:
               final_score = cosine_similarity + 0.15 * genre_overlap_ratio
           where genre_overlap_ratio = fraction of the movie's genres that
           appear in the book's mapped movie-genre set.
        5. Return top k.

        Args:
            movie_title: Title of the movie or TV show
            k: Number of book recommendations
            use_projection: Use contrastive 128-dim space if True

        Returns:
            DataFrame: [rank, book_id, book_title, author, genres,
                        similarity_score, genre_bonus, final_score, description]

        Raises:
            ValueError: If movie/show not found in catalog
        """
        # Ensure movie embeddings are available
        if self.movie_embeddings is None:
            self.load_movie_embeddings()

        # Locate movie
        movie_row, movie_emb_idx = self._find_movie_by_title(movie_title)
        found_title = movie_row["primaryTitle"]
        self.logger.info(f"  Found movie: '{found_title}'")

        movie_embedding = self.movie_embeddings[movie_emb_idx]  # 768-dim

        # Choose search space
        if use_projection and self.projection_head is not None:
            if self.projected_book_index is None:
                self.load_projected_book_index()
            query_embedding = self.project_embedding(movie_embedding)  # 128-dim
            search_index = self.projected_book_index
        else:
            if use_projection and self.projection_head is None:
                self.logger.warning(
                    "Projection head not loaded — falling back to baseline book index."
                )
            if self.book_baseline_index is None:
                self.load_book_baseline_index()
            query_embedding = movie_embedding  # 768-dim
            search_index = self.book_baseline_index

        # Search top 50 candidates
        scores, indices = search_index.search(
            query_embedding.reshape(1, -1).astype("float32"), 50
        )

        # Movie genres for overlap calculation
        raw_genres = str(movie_row.get("genres", ""))
        movie_genres = [g.strip() for g in raw_genres.split(",") if g.strip()]

        # Build results
        results = []
        for emb_idx, score in zip(indices[0], scores[0]):
            if emb_idx < 0 or emb_idx >= len(self.book_ids):
                continue

            book_id = self.book_ids[emb_idx]
            if book_id not in self.book_df.index:
                continue

            book_row = self.book_df.loc[book_id]

            # Genre overlap ratio: fraction of movie genres in book's mapped genres
            book_mapped_genres = get_mapped_movie_genres(book_row, self.genre_mapping)
            if movie_genres and book_mapped_genres:
                bm_set = set(book_mapped_genres)
                overlap_count = sum(1 for g in movie_genres if g in bm_set)
                genre_overlap_ratio = overlap_count / len(movie_genres)
            else:
                genre_overlap_ratio = 0.0

            genre_bonus = genre_overlap_ratio * 0.15
            final_score = float(score) + genre_bonus

            # Parse genres for display
            try:
                genres_display = ", ".join(
                    ast.literal_eval(str(book_row.get("Genres", "[]")))
                )
            except Exception:
                genres_display = str(book_row.get("Genres", ""))

            results.append(
                {
                    "book_id": book_id,
                    "book_title": str(book_row.get("Book", "Unknown")),
                    "author": str(book_row.get("Author", "Unknown")),
                    "genres": genres_display,
                    "similarity_score": float(score),
                    "genre_bonus": genre_bonus,
                    "final_score": final_score,
                    "description": str(book_row.get("Description", ""))[:300],
                }
            )

        if not results:
            self.logger.warning(f"No book results found for: {movie_title}")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("final_score", ascending=False)
        results_df = results_df.head(k).reset_index(drop=True)
        results_df["rank"] = range(1, len(results_df) + 1)
        results_df = results_df[
            [
                "rank", "book_id", "book_title", "author", "genres",
                "similarity_score", "genre_bonus", "final_score", "description",
            ]
        ]
        return results_df

    # ------------------------------------------------------------------ #
    #  Comparison helpers                                                   #
    # ------------------------------------------------------------------ #

    def compare_baseline_vs_contrastive(
        self, book_title: str, k: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Side-by-side comparison of baseline vs contrastive recommendations (book→movie).

        Args:
            book_title: Book title
            k: Number of recommendations

        Returns:
            (baseline_df, contrastive_df)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"COMPARISON: {book_title}")
        self.logger.info("=" * 70)

        baseline = self.recommend_movies(book_title, k, use_projection=False)
        contrastive = self.recommend_movies(book_title, k, use_projection=True)

        print("\n" + "-" * 70)
        print("BASELINE (book → movies/shows):")
        print("-" * 70)
        for _, row in baseline.iterrows():
            print(
                f"{row['rank']}. {row['movie_title']} ({row['year']}) "
                f"— Score: {row['final_score']:.3f}"
            )

        print("\n" + "-" * 70)
        print("CONTRASTIVE (book → movies/shows):")
        print("-" * 70)
        for _, row in contrastive.iterrows():
            print(
                f"{row['rank']}. {row['movie_title']} ({row['year']}) "
                f"— Score: {row['final_score']:.3f}"
            )
        print("-" * 70)

        return baseline, contrastive

    def test_comparison(self):
        """Test both directions with canonical examples."""
        # Book → movies/shows
        for book_title in [
            "Harry Potter and the Sorcerer's Stone",
            "Gone Girl",
            "The Martian",
        ]:
            self.compare_baseline_vs_contrastive(book_title, k=5)

        # Movie/show → books
        print("\n" + "=" * 70)
        print("REVERSE DIRECTION: movie/show → books")
        print("=" * 70)
        for movie_title in ["Inception", "Breaking Bad", "The Crown"]:
            print(f"\n--- {movie_title} → Books ---")
            try:
                recs = self.recommend_books_from_movie(movie_title, k=3, use_projection=True)
                for _, row in recs.iterrows():
                    print(
                        f"{row['rank']}. {row['book_title']} by {row['author']} "
                        f"— Score: {row['final_score']:.3f}"
                    )
            except ValueError as e:
                print(f"  ⚠ {e}")

    # ------------------------------------------------------------------ #
    #  Main execution                                                       #
    # ------------------------------------------------------------------ #

    def run(self):
        """Full initialisation + test pipeline for both directions."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("CONTRASTIVE RECOMMENDER — SPRINT 4A")
        self.logger.info("=" * 70)

        # Load baseline data (books, movies, embeddings, genre mapping)
        self.load_data()

        # Try to load projection head
        try:
            self.load_projection_head()
            has_projection = True
        except FileNotFoundError as e:
            self.logger.error(str(e))
            self.logger.warning("Contrastive mode unavailable — using baseline only.")
            has_projection = False

        # ---- Movie-side indices ----
        if has_projection:
            self.load_projected_index()  # 128-dim movie index

        # ---- Book-side indices (Sprint 4A) ----
        self.load_book_baseline_index()  # always build/load 768-dim book index

        if has_projection:
            self.load_projected_book_index()  # 128-dim book index

        # ---- Canonical tests ----
        self.logger.info("\n" + "=" * 70)
        self.logger.info("TEST: 'Dune' → movies/shows")
        self.logger.info("=" * 70)
        dune_recs = self.recommend_movies("Dune", k=3, use_projection=has_projection)
        if len(dune_recs):
            for _, row in dune_recs.iterrows():
                print(
                    f"  {row['rank']}. {row['movie_title']} ({row['year']}) "
                    f"— {row['final_score']:.3f}"
                )

        self.logger.info("\n" + "=" * 70)
        self.logger.info("TEST: 'Inception' → books")
        self.logger.info("=" * 70)
        try:
            inception_recs = self.recommend_books_from_movie(
                "Inception", k=3, use_projection=has_projection
            )
            if len(inception_recs):
                for _, row in inception_recs.iterrows():
                    print(
                        f"  {row['rank']}. {row['book_title']} by {row['author']} "
                        f"— {row['final_score']:.3f}"
                    )
        except ValueError as e:
            self.logger.error(str(e))

        # Full comparison test
        self.test_comparison()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("✅ CONTRASTIVE RECOMMENDER TESTING COMPLETE")
        self.logger.info("=" * 70)


def main():
    """Entry point"""
    recommender = ContrastiveRecommender()
    recommender.run()


if __name__ == "__main__":
    main()
