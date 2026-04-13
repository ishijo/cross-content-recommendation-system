"""
Debug script to diagnose evaluation failure.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import ast

sys.path.append('src')
from project_config import GOODREADS_RAW_CSV, IMDB_FILTERED_CSV, EMBEDDINGS_DIR
from utils.helpers import setup_logger, load_genre_mapping, get_mapped_movie_genres
from models.baseline_recommender import BaselineRecommender

logger = setup_logger("debug_evaluation")

def test_single_book_evaluation():
    """Test evaluation pipeline with a single book"""

    logger.info("\n" + "="*70)
    logger.info("DEBUGGING EVALUATION FAILURE")
    logger.info("="*70)

    # Load data
    logger.info("\n1. Loading data...")
    recommender = BaselineRecommender()
    recommender.load_data()

    # Load genre mapping
    genre_mapping = load_genre_mapping(logger)

    # Test with a single known book
    test_book_title = "Harry Potter and the Sorcerer's Stone"

    logger.info(f"\n2. Testing with: {test_book_title}")

    # Find book in dataframe
    book_matches = recommender.book_df[
        recommender.book_df['Book'].str.lower().str.contains("harry potter")
    ]

    if len(book_matches) == 0:
        logger.error("  ❌ Harry Potter not found in book dataframe!")
        return

    book_row = book_matches.iloc[0]
    logger.info(f"  ✓ Found book: {book_row['Book']}")
    logger.info(f"  Book genres: {book_row['Genres']}")

    # Get mapped movie genres
    book_movie_genres = get_mapped_movie_genres(book_row, genre_mapping)
    logger.info(f"  Mapped movie genres: {book_movie_genres}")

    if not book_movie_genres:
        logger.error("  ❌ No mapped movie genres! This is the problem.")
        logger.error("  Genre mapping might be broken or empty.")
        return

    # Get recommendations
    logger.info("\n3. Getting recommendations...")
    try:
        recommendations = recommender.recommend_movies(test_book_title, k=10, genre_filter=True)
        logger.info(f"  ✓ Got {len(recommendations)} recommendations")

        if len(recommendations) > 0:
            logger.info("\n  Top 3 recommendations:")
            for _, row in recommendations.head(3).iterrows():
                logger.info(f"    {row['rank']}. {row['movie_title']} - {row['genres']}")
                logger.info(f"       Score: {row['final_score']:.3f}")
    except Exception as e:
        logger.error(f"  ❌ Recommendation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Find relevant movies (movies that share genres)
    logger.info("\n4. Finding relevant movies...")
    relevant_count = 0

    for movie_id, movie_row in recommender.movie_df.iterrows():
        movie_genres = [g.strip() for g in str(movie_row['genres']).split(',')]
        if any(g in movie_genres for g in book_movie_genres):
            relevant_count += 1

    logger.info(f"  ✓ Found {relevant_count} relevant movies (share at least 1 genre)")
    logger.info(f"  Relevant genres: {book_movie_genres}")

    if relevant_count == 0:
        logger.error("  ❌ ZERO relevant movies found!")
        logger.error("  This explains why evaluation fails.")
        logger.error("  Problem: Genre mapping or genre matching logic is broken.")

        # Debug: Show some movie genres
        logger.info("\n  Sample IMDB movie genres:")
        for i, (_, row) in enumerate(recommender.movie_df.head(10).iterrows()):
            logger.info(f"    {row['primaryTitle']}: {row['genres']}")
    else:
        logger.info(f"  ✓ Evaluation should work. Coverage: {relevant_count/len(recommender.movie_df)*100:.1f}%")

    # Check if recommended movies are in relevant set
    logger.info("\n5. Checking if recommendations are relevant...")
    if len(recommendations) > 0:
        recommended_ids = recommendations['movie_id'].tolist()

        relevant_in_recs = 0
        for movie_id in recommended_ids:
            if movie_id in recommender.movie_df.index:
                movie_row = recommender.movie_df.loc[movie_id]
                movie_genres = [g.strip() for g in str(movie_row['genres']).split(',')]
                if any(g in movie_genres for g in book_movie_genres):
                    relevant_in_recs += 1

        logger.info(f"  Relevant in top-10: {relevant_in_recs}/10")
        logger.info(f"  Precision: {relevant_in_recs/10:.2f}")

        if relevant_in_recs == 0:
            logger.warning("  ⚠️  None of the recommendations are relevant!")
            logger.warning("  This suggests genre filtering isn't working properly.")

    logger.info("\n" + "="*70)
    logger.info("DEBUG COMPLETE")
    logger.info("="*70)


def check_embedding_quality():
    """Check if embeddings look reasonable"""

    logger.info("\n" + "="*70)
    logger.info("CHECKING EMBEDDING QUALITY")
    logger.info("="*70)

    import numpy as np

    # Load embeddings
    book_emb = np.load(EMBEDDINGS_DIR / "book_embeddings.npy")
    movie_emb = np.load(EMBEDDINGS_DIR / "movie_embeddings.npy")

    logger.info(f"\nBook embeddings shape: {book_emb.shape}")
    logger.info(f"Movie embeddings shape: {movie_emb.shape}")

    # Check norms (should be ~1.0 if normalized)
    book_norms = np.linalg.norm(book_emb, axis=1)
    movie_norms = np.linalg.norm(movie_emb, axis=1)

    logger.info(f"\nBook embedding norms:")
    logger.info(f"  Mean: {book_norms.mean():.4f} (should be ~1.0)")
    logger.info(f"  Std: {book_norms.std():.4f}")
    logger.info(f"  Range: {book_norms.min():.4f} - {book_norms.max():.4f}")

    logger.info(f"\nMovie embedding norms:")
    logger.info(f"  Mean: {movie_norms.mean():.4f} (should be ~1.0)")
    logger.info(f"  Std: {movie_norms.std():.4f}")
    logger.info(f"  Range: {movie_norms.min():.4f} - {movie_norms.max():.4f}")

    # Check for NaNs
    book_nans = np.isnan(book_emb).sum()
    movie_nans = np.isnan(movie_emb).sum()

    if book_nans > 0:
        logger.error(f"  ❌ {book_nans} NaN values in book embeddings!")
    if movie_nans > 0:
        logger.error(f"  ❌ {movie_nans} NaN values in movie embeddings!")

    # Check a few similarities manually
    logger.info("\n5. Manual similarity check (first 3 books vs first 3 movies):")
    for i in range(min(3, len(book_emb))):
        for j in range(min(3, len(movie_emb))):
            sim = np.dot(book_emb[i], movie_emb[j])
            logger.info(f"  Book {i} vs Movie {j}: {sim:.4f}")


def check_genre_mapping():
    """Check if genre mapping is working"""

    logger.info("\n" + "="*70)
    logger.info("CHECKING GENRE MAPPING")
    logger.info("="*70)

    genre_mapping = load_genre_mapping(logger)

    logger.info(f"\nTotal genre mappings: {len(genre_mapping)}")

    # Check how many are empty
    empty_mappings = sum(1 for v in genre_mapping.values() if len(v) == 0)
    logger.info(f"Empty mappings (no movies): {empty_mappings} ({empty_mappings/len(genre_mapping)*100:.1f}%)")

    # Show some examples
    logger.info("\nSample mappings:")
    for i, (book_genre, movie_genres) in enumerate(list(genre_mapping.items())[:10]):
        logger.info(f"  {book_genre} → {movie_genres}")

    # Check what movie genres actually exist in IMDB
    logger.info("\nChecking IMDB genres...")
    imdb_df = pd.read_csv(IMDB_FILTERED_CSV, index_col=0)

    all_imdb_genres = set()
    for genres_str in imdb_df['genres']:
        if pd.notna(genres_str):
            genres = [g.strip() for g in str(genres_str).split(',')]
            all_imdb_genres.update(genres)

    logger.info(f"Unique IMDB genres: {sorted(list(all_imdb_genres))}")

    # Check if mapped genres actually exist in IMDB
    logger.info("\nValidating mappings against actual IMDB genres...")
    invalid_count = 0
    for book_genre, movie_genres in genre_mapping.items():
        for mg in movie_genres:
            if mg not in all_imdb_genres:
                invalid_count += 1
                if invalid_count <= 5:  # Show first 5
                    logger.warning(f"  ⚠️  Mapped genre '{mg}' doesn't exist in IMDB!")

    if invalid_count > 0:
        logger.error(f"  ❌ {invalid_count} invalid genre mappings!")
        logger.error("  This will cause zero matches in evaluation.")


def main():
    """Run all debug checks"""
    check_genre_mapping()
    check_embedding_quality()
    test_single_book_evaluation()

    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*70)
    logger.info("\nCheck the output above to identify the root cause.")


if __name__ == "__main__":
    main()
