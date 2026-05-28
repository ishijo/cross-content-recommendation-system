"""
Build FAISS index for fast movie similarity search.
Uses IndexFlatIP (inner product) since embeddings are L2-normalized.
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from project_config import (
    EMBEDDINGS_DIR,
    IMDB_FILTERED_CSV,
    SENTENCE_TRANSFORMER_MODEL
)
from utils.helpers import setup_logger, timer
from sentence_transformers import SentenceTransformer


class FAISSIndexer:
    """Build and query FAISS index for movie embeddings"""

    def __init__(self):
        self.logger = setup_logger("faiss_indexer")
        self.index = None
        self.movie_ids = None
        self.movie_df = None
        self.model = None

    def load_movie_embeddings(self) -> np.ndarray:
        """
        Load movie embeddings from disk.

        Returns:
            Movie embeddings array (n_movies x embedding_dim)
        """
        emb_path = EMBEDDINGS_DIR / "movie_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Movie embeddings not found at {emb_path}. "
                "Run embedding_generator.py first."
            )

        with timer("Loading movie embeddings", self.logger):
            embeddings = np.load(emb_path)

        self.logger.info(f"  Loaded embeddings shape: {embeddings.shape}")
        self.logger.info(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")

        return embeddings

    def load_movie_ids(self) -> List[int]:
        """
        Load movie IDs from disk.

        Returns:
            List of movie IDs matching embedding order
        """
        ids_path = EMBEDDINGS_DIR / "movie_ids.json"
        if not ids_path.exists():
            raise FileNotFoundError(f"Movie IDs not found at {ids_path}")

        with open(ids_path, 'r') as f:
            movie_ids = json.load(f)

        self.logger.info(f"  Loaded {len(movie_ids):,} movie IDs")

        return movie_ids

    def load_movie_dataframe(self) -> pd.DataFrame:
        """
        Load movie dataframe for lookups.

        Returns:
            IMDB dataframe
        """
        movie_df = pd.read_csv(IMDB_FILTERED_CSV, index_col=0)
        self.logger.info(f"  Loaded movie dataframe: {len(movie_df):,} rows")

        return movie_df

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build FAISS IndexFlatIP for inner product search.

        Args:
            embeddings: Movie embeddings (must be L2-normalized)

        Returns:
            FAISS index
        """
        with timer(f"Building FAISS index for {len(embeddings):,} movies", self.logger):
            # Get embedding dimension
            embedding_dim = embeddings.shape[1]

            # Create index (Inner Product = cosine similarity for normalized vectors)
            index = faiss.IndexFlatIP(embedding_dim)

            # Add embeddings
            index.add(embeddings.astype('float32'))

        self.logger.info(f"  Index built: {index.ntotal} vectors indexed")

        return index

    def save_index(self, index: faiss.IndexFlatIP):
        """
        Save FAISS index to disk.

        Args:
            index: FAISS index to save
        """
        index_path = EMBEDDINGS_DIR / "movie_faiss.index"

        with timer(f"Saving index to {index_path}", self.logger):
            faiss.write_index(index, str(index_path))

        self.logger.info(f"  ✓ Index saved")

    def load_index(self) -> faiss.IndexFlatIP:
        """
        Load FAISS index from disk.

        Returns:
            FAISS index
        """
        index_path = EMBEDDINGS_DIR / "movie_faiss.index"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Build index first."
            )

        with timer("Loading FAISS index", self.logger):
            index = faiss.read_index(str(index_path))

        self.logger.info(f"  Loaded index: {index.ntotal} vectors")

        return index

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for top-k most similar movies.

        Args:
            query_vector: Query embedding (1D array or 2D array with shape (1, dim))
            k: Number of results to return

        Returns:
            List of (movie_id, similarity_score) tuples, sorted by score descending
        """
        if self.index is None or self.movie_ids is None:
            raise ValueError("Index not loaded. Call load_all() first.")

        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_vector.astype('float32'), k)

        # Convert to list of tuples
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.movie_ids):  # Valid index
                movie_id = self.movie_ids[idx]
                results.append((movie_id, float(score)))

        return results

    def get_movie_info(self, movie_id: int) -> dict:
        """
        Get movie information by ID.

        Args:
            movie_id: Movie ID (index in dataframe)

        Returns:
            Dictionary with movie info
        """
        if self.movie_df is None:
            raise ValueError("Movie dataframe not loaded")

        if movie_id not in self.movie_df.index:
            return {'title': 'Unknown', 'genres': 'Unknown', 'rating': 0.0}

        row = self.movie_df.loc[movie_id]

        return {
            'title': row['primaryTitle'],
            'genres': row['genres'],
            'rating': row.get('averageRating', 0.0),
            'year': row.get('startYear', 'N/A'),
            'description': row.get('Description', '')[:100] + '...'
        }

    def test_search(self):
        """
        Test the search with a sample query.
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("TEST SEARCH")
        self.logger.info("="*70)

        # Load model for encoding test query
        self.logger.info("Loading model for test query...")
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        # Test query
        test_query = "a dark psychological thriller with an unreliable narrator"
        self.logger.info(f"\nTest query: \"{test_query}\"")

        # Encode query
        query_vector = model.encode(
            [test_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # Search
        results = self.search(query_vector, k=5)

        # Display results
        self.logger.info(f"\nTop 5 results:")
        for rank, (movie_id, score) in enumerate(results, 1):
            movie_info = self.get_movie_info(movie_id)
            self.logger.info(
                f"\n  {rank}. {movie_info['title']} ({movie_info['year']})"
            )
            self.logger.info(f"     Genres: {movie_info['genres']}")
            self.logger.info(f"     Rating: {movie_info['rating']}")
            self.logger.info(f"     Similarity: {score:.4f}")
            self.logger.info(f"     Description: {movie_info['description']}")

    def load_all(self):
        """Load index, IDs, and dataframe"""
        self.index = self.load_index()
        self.movie_ids = self.load_movie_ids()
        self.movie_df = self.load_movie_dataframe()

    def run(self):
        """Main execution pipeline"""
        self.logger.info("\n" + "="*70)
        self.logger.info("FAISS INDEX BUILDER")
        self.logger.info("="*70)

        # Check if index already exists
        index_path = EMBEDDINGS_DIR / "movie_faiss.index"
        if index_path.exists():
            self.logger.info("\n✓ FAISS index already exists")
            self.logger.info("  To rebuild, delete: data/embeddings/movie_faiss.index")

            # Load and test
            self.load_all()
            self.test_search()

            return

        # Load movie embeddings
        embeddings = self.load_movie_embeddings()

        # Load movie IDs
        self.movie_ids = self.load_movie_ids()

        # Build index
        self.index = self.build_index(embeddings)

        # Save index
        self.save_index(self.index)

        # Load movie dataframe for testing
        self.movie_df = self.load_movie_dataframe()

        # Test search
        self.test_search()

        self.logger.info("\n" + "="*70)
        self.logger.info("✅ FAISS INDEX COMPLETE")
        self.logger.info("="*70)


def main():
    """Entry point"""
    indexer = FAISSIndexer()
    indexer.run()


if __name__ == "__main__":
    main()
