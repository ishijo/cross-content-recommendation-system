"""
Quick fix for evaluation failure.
This script patches the evaluation to be more verbose and handle edge cases.
"""
import json
import sys
from pathlib import Path
from typing import List, Set, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ast

# Add parent directory to path
sys.path.append('src')
from project_config import PLOTS_DIR, PROCESSED_DATA_DIR
from utils.helpers import setup_logger, timer, load_genre_mapping

logger = setup_logger("fixed_evaluation")


def get_mapped_movie_genres_safe(book_row: pd.Series, genre_mapping: Dict[str, List[str]]) -> List[str]:
    """
    FIXED version - safer genre mapping with detailed logging.
    """
    try:
        # Parse genres string to list
        genres_str = book_row.get('Genres', '[]')
        if isinstance(genres_str, str):
            book_genres = ast.literal_eval(genres_str)
        else:
            book_genres = []
    except Exception as e:
        logger.warning(f"  Failed to parse genres for {book_row.get('Book', 'unknown')}: {e}")
        return []

    if not book_genres:
        logger.warning(f"  No genres found for {book_row.get('Book', 'unknown')}")
        return []

    # Collect all mapped movie genres
    movie_genres = set()
    for book_genre in book_genres:
        mapped = genre_mapping.get(book_genre, [])
        if mapped:
            movie_genres.update(mapped)

    result = sorted(list(movie_genres))

    if not result:
        logger.warning(f"  No movie genres mapped for {book_row.get('Book', 'unknown')} (book genres: {book_genres})")

    return result


def find_relevant_movies_safe(
    book_row: pd.Series,
    movie_df: pd.DataFrame,
    genre_mapping: Dict[str, List[str]],
    logger
) -> Set[int]:
    """
    FIXED version - find relevant movies with detailed logging.
    """
    # Get mapped movie genres for this book
    book_movie_genres = get_mapped_movie_genres_safe(book_row, genre_mapping)

    if not book_movie_genres:
        logger.warning(f"  No mapped genres for book, skipping: {book_row.get('Book', 'unknown')}")
        return set()

    # Find movies that share at least one genre
    relevant_ids = set()

    for movie_id, movie_row in movie_df.iterrows():
        try:
            genres_str = movie_row.get('genres', '')
            if pd.isna(genres_str) or str(genres_str).strip() == '':
                continue

            movie_genres = [g.strip() for g in str(genres_str).split(',')]

            # Check for overlap
            if any(g in movie_genres for g in book_movie_genres):
                relevant_ids.add(movie_id)

        except Exception as e:
            continue

    if len(relevant_ids) == 0:
        logger.warning(f"  ZERO relevant movies for: {book_row.get('Book', 'unknown')}")
        logger.warning(f"  Book mapped genres: {book_movie_genres}")

    return relevant_ids


def precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Calculate Precision@K"""
    if k == 0 or len(recommended_ids) == 0:
        return 0.0
    top_k = recommended_ids[:k]
    relevant_count = sum(1 for item_id in top_k if item_id in relevant_ids)
    return relevant_count / k


def ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Calculate NDCG@K"""
    if k == 0 or len(recommended_ids) == 0:
        return 0.0

    top_k = recommended_ids[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(top_k, 1):
        relevance = 1.0 if item_id in relevant_ids else 0.0
        dcg += relevance / np.log2(i + 1)

    # Calculate ideal DCG
    n_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_relevant + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_baseline_fixed(
    recommender,
    books_sample_size: int = 200,
    k: int = 10
) -> Dict:
    """
    FIXED version of evaluation with verbose error logging.
    """
    logger.info("\n" + "="*70)
    logger.info(f"FIXED EVALUATION (n={books_sample_size}, k={k})")
    logger.info("="*70)

    # Load genre mapping
    genre_mapping = load_genre_mapping(logger)

    # Sample books
    sample_books = recommender.book_df.sample(n=min(books_sample_size, len(recommender.book_df)))
    logger.info(f"\nSampled {len(sample_books)} books for evaluation")

    # Storage for metrics
    precision_scores = []
    ndcg_scores = []
    all_recommended_ids = []

    # Counters for debugging
    success_count = 0
    fail_no_recs = 0
    fail_no_genres = 0
    fail_no_relevant = 0
    fail_exception = 0

    # Evaluate each book
    for idx, (book_id, book_row) in enumerate(tqdm(sample_books.iterrows(), total=len(sample_books), desc="Evaluating")):
        try:
            # Get recommendations
            book_title = book_row['Book']

            try:
                recommendations = recommender.recommend_movies(book_title, k=k, genre_filter=True)
            except Exception as e:
                logger.error(f"  Failed to get recommendations for '{book_title}': {e}")
                fail_exception += 1
                continue

            if len(recommendations) == 0:
                logger.warning(f"  No recommendations for: {book_title}")
                fail_no_recs += 1
                continue

            # Get recommended movie IDs
            recommended_ids = recommendations['movie_id'].tolist()
            all_recommended_ids.extend(recommended_ids)

            # Find relevant movies
            relevant_ids = find_relevant_movies_safe(book_row, recommender.movie_df, genre_mapping, logger)

            if len(relevant_ids) == 0:
                fail_no_relevant += 1
                continue

            # Calculate metrics
            precision = precision_at_k(recommended_ids, relevant_ids, k)
            ndcg = ndcg_at_k(recommended_ids, relevant_ids, k)

            precision_scores.append(precision)
            ndcg_scores.append(ndcg)
            success_count += 1

            # Log first few successes
            if success_count <= 3:
                logger.info(f"\n  ✓ Success #{success_count}: {book_title}")
                logger.info(f"    Relevant movies: {len(relevant_ids)}")
                logger.info(f"    Precision@{k}: {precision:.3f}")
                logger.info(f"    NDCG@{k}: {ndcg:.3f}")

        except Exception as e:
            logger.error(f"  Unexpected error for book {idx}: {e}")
            import traceback
            traceback.print_exc()
            fail_exception += 1
            continue

    # Print debug summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION DEBUG SUMMARY")
    logger.info("="*70)
    logger.info(f"\n  Total sampled: {len(sample_books)}")
    logger.info(f"  ✓ Successful: {success_count}")
    logger.info(f"  ✗ Failed (no recommendations): {fail_no_recs}")
    logger.info(f"  ✗ Failed (no mapped genres): {fail_no_genres}")
    logger.info(f"  ✗ Failed (no relevant movies): {fail_no_relevant}")
    logger.info(f"  ✗ Failed (exceptions): {fail_exception}")

    if success_count == 0:
        logger.error("\n  ❌ ZERO SUCCESSFUL EVALUATIONS!")
        logger.error("  Common causes:")
        logger.error("    1. Genre mapping is empty or broken")
        logger.error("    2. Book genres aren't being parsed")
        logger.error("    3. Genre matching logic has a bug")

        return {
            'precision_at_k': 0.0,
            'ndcg_at_k': 0.0,
            'coverage': 0.0,
            'n_evaluated': 0,
            'k': k,
            'error': 'All evaluations failed'
        }

    # Calculate averages
    results = {
        'precision_at_k': float(np.mean(precision_scores)),
        'ndcg_at_k': float(np.mean(ndcg_scores)),
        'coverage': float(len(set(all_recommended_ids)) / len(recommender.movie_df)),
        'n_evaluated': success_count,
        'k': k,
        'n_failed_no_recs': fail_no_recs,
        'n_failed_no_genres': fail_no_genres,
        'n_failed_no_relevant': fail_no_relevant,
        'n_failed_exception': fail_exception
    }

    # Print results
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"\n  Books successfully evaluated: {results['n_evaluated']}")
    logger.info(f"  Precision@{k}: {results['precision_at_k']:.4f}")
    logger.info(f"  NDCG@{k}: {results['ndcg_at_k']:.4f}")
    logger.info(f"  Coverage: {results['coverage']:.4f} ({results['coverage']*100:.1f}% of catalog)")

    return results


def main():
    """Run fixed evaluation"""
    logger.info("\n" + "="*70)
    logger.info("RUNNING FIXED EVALUATION")
    logger.info("="*70)

    # Import here to avoid circular dependency
    from models.baseline_recommender import BaselineRecommender

    # Load recommender
    logger.info("\nInitializing baseline recommender...")
    recommender = BaselineRecommender()
    recommender.load_data()

    # Evaluate with fixed version
    results = evaluate_baseline_fixed(
        recommender,
        books_sample_size=200,
        k=10
    )

    # Save results
    results_path = PROCESSED_DATA_DIR / "baseline_metrics_fixed.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  ✓ Saved results: {results_path}")

    logger.info("\n" + "="*70)
    logger.info("✅ FIXED EVALUATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
