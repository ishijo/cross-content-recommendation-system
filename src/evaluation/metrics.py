"""
Evaluation metrics for the recommendation system.
Implements Precision@K, Recall@K, NDCG@K, and Coverage.
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from project_config import PLOTS_DIR, PROCESSED_DATA_DIR, BIDIRECTIONAL_EVAL_RESULTS
from utils.helpers import setup_logger, timer, load_genre_mapping, get_mapped_movie_genres


def precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Calculate Precision@K.
    Precision = |relevant ∩ recommended| / k

    Args:
        recommended_ids: List of recommended item IDs (in ranked order)
        relevant_ids: Set of relevant item IDs
        k: Number of top recommendations to consider

    Returns:
        Precision score (0.0 to 1.0)
    """
    if k == 0 or len(recommended_ids) == 0:
        return 0.0

    # Consider only top k
    top_k = recommended_ids[:k]

    # Count relevant items in top k
    relevant_count = sum(1 for item_id in top_k if item_id in relevant_ids)

    return relevant_count / k


def recall_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Calculate Recall@K.
    Recall = |relevant ∩ recommended| / |relevant|

    Args:
        recommended_ids: List of recommended item IDs (in ranked order)
        relevant_ids: Set of relevant item IDs
        k: Number of top recommendations to consider

    Returns:
        Recall score (0.0 to 1.0)
    """
    if len(relevant_ids) == 0:
        return 0.0

    # Consider only top k
    top_k = recommended_ids[:k]

    # Count relevant items in top k
    relevant_count = sum(1 for item_id in top_k if item_id in relevant_ids)

    return relevant_count / len(relevant_ids)


def ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K.
    Uses binary relevance (1 if relevant, 0 if not).

    Args:
        recommended_ids: List of recommended item IDs (in ranked order)
        relevant_ids: Set of relevant item IDs
        k: Number of top recommendations to consider

    Returns:
        NDCG score (0.0 to 1.0)
    """
    if k == 0 or len(recommended_ids) == 0:
        return 0.0

    # Consider only top k
    top_k = recommended_ids[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(top_k, 1):
        relevance = 1.0 if item_id in relevant_ids else 0.0
        dcg += relevance / np.log2(i + 1)

    # Calculate ideal DCG (all relevant items at top)
    n_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_relevant + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def coverage(all_recommendations: List[int], total_catalog_size: int) -> float:
    """
    Calculate catalog coverage.
    Coverage = |unique recommended items| / |total catalog|

    Args:
        all_recommendations: List of all recommended item IDs (from all queries)
        total_catalog_size: Total number of items in catalog

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if total_catalog_size == 0:
        return 0.0

    unique_recommended = len(set(all_recommendations))

    return unique_recommended / total_catalog_size


def evaluate_baseline(
    recommender,
    books_sample_size: int = 200,
    k: int = 10,
    logger=None
) -> Dict:
    """
    Evaluate baseline recommender on a sample of books.

    Args:
        recommender: BaselineRecommender instance (must have load_data() called)
        books_sample_size: Number of books to sample for evaluation
        k: Number of recommendations per book
        logger: Optional logger

    Returns:
        Dictionary with evaluation results
    """
    if logger:
        logger.info("\n" + "="*70)
        logger.info(f"EVALUATING BASELINE RECOMMENDER (n={books_sample_size}, k={k})")
        logger.info("="*70)

    # Load genre mapping
    genre_mapping = load_genre_mapping(logger)

    # Sample books
    sample_books = recommender.book_df.sample(n=min(books_sample_size, len(recommender.book_df)))

    if logger:
        logger.info(f"\nSampled {len(sample_books)} books for evaluation")

    # Storage for metrics
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    all_recommended_ids = []

    # Evaluate each book
    with timer(f"Evaluating {len(sample_books)} books", logger):
        for idx, (book_id, book_row) in enumerate(tqdm(sample_books.iterrows(), total=len(sample_books), desc="Evaluating")):
            try:
                # Get recommendations
                book_title = book_row['Book']
                recommendations = recommender.recommend_movies(book_title, k=k, genre_filter=True)

                if len(recommendations) == 0:
                    continue

                # Get recommended movie IDs
                recommended_ids = recommendations['movie_id'].tolist()
                all_recommended_ids.extend(recommended_ids)

                # Get relevant movie IDs (movies that share at least one genre)
                book_movie_genres = get_mapped_movie_genres(book_row, genre_mapping)

                if not book_movie_genres:
                    continue

                # Find relevant movies in catalog
                relevant_ids = set()
                for movie_id, movie_row in recommender.movie_df.iterrows():
                    movie_genres = [g.strip() for g in str(movie_row['genres']).split(',')]
                    if any(g in movie_genres for g in book_movie_genres):
                        relevant_ids.add(movie_id)

                # Calculate metrics
                precision = precision_at_k(recommended_ids, relevant_ids, k)
                recall = recall_at_k(recommended_ids, relevant_ids, k)
                ndcg = ndcg_at_k(recommended_ids, relevant_ids, k)

                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)

            except Exception as e:
                if logger:
                    logger.warning(f"  Error evaluating book {idx}: {e}")
                continue

    # Calculate averages
    results = {
        'precision_at_k': float(np.mean(precision_scores)) if precision_scores else 0.0,
        'recall_at_k': float(np.mean(recall_scores)) if recall_scores else 0.0,
        'ndcg_at_k': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        'coverage': float(coverage(all_recommended_ids, len(recommender.movie_df))),
        'n_evaluated': len(precision_scores),
        'k': k
    }

    # Print summary
    if logger:
        logger.info("\n" + "="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"\n  Books evaluated: {results['n_evaluated']}")
        logger.info(f"  Precision@{k}: {results['precision_at_k']:.4f}")
        logger.info(f"  Recall@{k}: {results['recall_at_k']:.4f}")
        logger.info(f"  NDCG@{k}: {results['ndcg_at_k']:.4f}")
        logger.info(f"  Coverage: {results['coverage']:.4f} ({results['coverage']*100:.1f}% of catalog)")

    return results


def plot_evaluation_results(results: Dict, output_path: Path, logger=None):
    """
    Create bar chart of evaluation metrics.

    Args:
        results: Results dictionary from evaluate_baseline()
        output_path: Path to save plot
        logger: Optional logger
    """
    # Prepare data
    metrics = ['Precision@K', 'Recall@K', 'NDCG@K', 'Coverage']
    values = [
        results['precision_at_k'],
        results['recall_at_k'],
        results['ndcg_at_k'],
        results['coverage']
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')

    # Customize
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Baseline Recommender Evaluation (K={results["k"]})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if logger:
        logger.info(f"\n  ✓ Saved evaluation plot: {output_path}")

    plt.close()


def evaluate_bidirectional(
    recommender,
    sample_size: int = 200,
    k: int = 10,
    logger=None,
) -> Dict:
    """
    Evaluate the recommender in both directions on random samples.

    Samples `sample_size` books and `sample_size` movies/shows (seed=42),
    then computes Precision@K and nDCG@K for each direction.

    Relevance definitions:
    - Book → Movie/Show:  movie is relevant if any of its genres overlap
                          with the book's mapped movie-genre set.
    - Movie/Show → Book:  book is relevant if any of its mapped movie-genres
                          overlap with the movie's actual genres.

    Args:
        recommender: ContrastiveRecommender (load_data() already called)
        sample_size: Number of books and movies to sample per direction
        k: Top-K to evaluate
        logger: Optional logger

    Returns:
        Dictionary with per-direction Precision@K and nDCG@K.
        Also saves results to BIDIRECTIONAL_EVAL_RESULTS.
    """
    import ast

    if logger:
        logger.info("\n" + "="*70)
        logger.info(f"BIDIRECTIONAL EVALUATION (n={sample_size}, k={k})")
        logger.info("="*70)

    genre_mapping = load_genre_mapping(logger)

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # Direction A: Book → Movies/Shows                                     #
    # ------------------------------------------------------------------ #
    book_sample = recommender.book_df.sample(
        n=min(sample_size, len(recommender.book_df)),
        random_state=42,
    )

    prec_b2m, ndcg_b2m = [], []

    if logger:
        logger.info(f"\n[Book → Movies/Shows] Evaluating {len(book_sample)} books…")

    for book_id, book_row in tqdm(
        book_sample.iterrows(), total=len(book_sample), desc="Book→Movie"
    ):
        try:
            recs = recommender.recommend_movies(
                book_row["Book"], k=k, genre_filter=True, use_projection=True
            )
            if len(recs) == 0:
                continue

            recommended_ids = recs["movie_id"].tolist()

            book_movie_genres = get_mapped_movie_genres(book_row, genre_mapping)
            if not book_movie_genres:
                continue

            # Relevant movies: share at least one genre with book's mapped genres
            relevant_ids = set()
            for mid, mrow in recommender.movie_df.iterrows():
                mgenres = [g.strip() for g in str(mrow["genres"]).split(",")]
                if any(g in mgenres for g in book_movie_genres):
                    relevant_ids.add(mid)

            prec_b2m.append(precision_at_k(recommended_ids, relevant_ids, k))
            ndcg_b2m.append(ndcg_at_k(recommended_ids, relevant_ids, k))

        except Exception as e:
            if logger:
                logger.warning(f"  Skipping book '{book_row.get('Book', book_id)}': {e}")

    # ------------------------------------------------------------------ #
    # Direction B: Movie/Show → Books                                      #
    # ------------------------------------------------------------------ #
    movie_sample = recommender.movie_df.sample(
        n=min(sample_size, len(recommender.movie_df)),
        random_state=42,
    )

    prec_m2b, ndcg_m2b = [], []

    if logger:
        logger.info(f"\n[Movie/Show → Books] Evaluating {len(movie_sample)} movies/shows…")

    for movie_id, movie_row in tqdm(
        movie_sample.iterrows(), total=len(movie_sample), desc="Movie→Book"
    ):
        try:
            movie_title = str(movie_row["primaryTitle"])
            recs = recommender.recommend_books_from_movie(
                movie_title, k=k, use_projection=True
            )
            if len(recs) == 0:
                continue

            recommended_ids = recs["book_id"].tolist()

            movie_genres = [g.strip() for g in str(movie_row.get("genres", "")).split(",")]
            movie_genres_set = set(g for g in movie_genres if g)
            if not movie_genres_set:
                continue

            # Relevant books: any mapped movie-genre overlaps with movie's genres
            relevant_ids = set()
            for bid, brow in recommender.book_df.iterrows():
                book_mapped = set(get_mapped_movie_genres(brow, genre_mapping))
                if book_mapped & movie_genres_set:
                    relevant_ids.add(bid)

            prec_m2b.append(precision_at_k(recommended_ids, relevant_ids, k))
            ndcg_m2b.append(ndcg_at_k(recommended_ids, relevant_ids, k))

        except Exception as e:
            if logger:
                logger.warning(
                    f"  Skipping movie '{movie_row.get('primaryTitle', movie_id)}': {e}"
                )

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    results = {
        "book_to_movie": {
            f"precision_at_{k}": float(np.mean(prec_b2m)) if prec_b2m else 0.0,
            f"ndcg_at_{k}": float(np.mean(ndcg_b2m)) if ndcg_b2m else 0.0,
            "n_evaluated": len(prec_b2m),
        },
        "movie_to_book": {
            f"precision_at_{k}": float(np.mean(prec_m2b)) if prec_m2b else 0.0,
            f"ndcg_at_{k}": float(np.mean(ndcg_m2b)) if ndcg_m2b else 0.0,
            "n_evaluated": len(prec_m2b),
        },
        "k": k,
    }

    # Print comparison table
    header = f"\n{'Direction':<30}  {'Precision@'+str(k):<16}  {'nDCG@'+str(k):<12}"
    sep = "-" * len(header)
    b2m = results["book_to_movie"]
    m2b = results["movie_to_book"]
    table = (
        f"\n{sep}"
        f"{header}"
        f"\n{sep}"
        f"\n{'Book → Movies/Shows':<30}  "
        f"{b2m[f'precision_at_{k}']:<16.4f}  "
        f"{b2m[f'ndcg_at_{k}']:.4f}"
        f"\n{'Movie/Show → Books':<30}  "
        f"{m2b[f'precision_at_{k}']:<16.4f}  "
        f"{m2b[f'ndcg_at_{k}']:.4f}"
        f"\n{sep}"
    )
    print(table)
    if logger:
        logger.info(table)

    # Save results
    with open(BIDIRECTIONAL_EVAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    if logger:
        logger.info(f"\n  ✓ Saved: {BIDIRECTIONAL_EVAL_RESULTS}")

    return results


def main():
    """Entry point"""
    logger = setup_logger("metrics_evaluation")

    logger.info("\n" + "="*70)
    logger.info("EVALUATION PIPELINE")
    logger.info("="*70)

    # Import here to avoid circular dependency
    from models.baseline_recommender import BaselineRecommender
    from models.contrastive_recommender import ContrastiveRecommender

    # ---- Baseline evaluation (book → movie) ----
    logger.info("\nInitializing baseline recommender…")
    baseline = BaselineRecommender()
    baseline.load_data()

    results = evaluate_baseline(
        baseline,
        books_sample_size=200,
        k=10,
        logger=logger,
    )

    results_path = PROCESSED_DATA_DIR / "baseline_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  ✓ Saved: {results_path}")

    plot_path = PLOTS_DIR / "baseline_evaluation.png"
    plot_evaluation_results(results, plot_path, logger)

    # ---- Bidirectional evaluation ----
    logger.info("\nInitializing contrastive recommender for bidirectional eval…")
    recommender = ContrastiveRecommender()
    recommender.load_data()

    try:
        recommender.load_projection_head()
        recommender.load_projected_index()
        recommender.load_projected_book_index()
    except Exception as e:
        logger.warning(f"Contrastive model unavailable ({e}) — using baseline for both directions.")
        recommender.load_book_baseline_index()

    evaluate_bidirectional(recommender, sample_size=200, k=10, logger=logger)

    logger.info("\n" + "="*70)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
