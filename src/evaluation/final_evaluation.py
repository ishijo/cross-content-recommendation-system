"""
Final Evaluation: Baseline vs Contrastive Learning

Comprehensive comparison of baseline and contrastive recommendation systems
on 500 randomly sampled books across multiple metrics.

Metrics:
- Precision@10: Fraction of relevant recommendations
- nDCG@10: Normalized ranking quality
- Coverage: Fraction of catalog recommended
- Diversity: Mean intra-list pairwise distance

Usage:
    python src/evaluation/final_evaluation.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm

from project_config import (
    PROCESSED_DATA_DIR,
    PLOTS_DIR
)
from utils.helpers import (
    setup_logger,
    timer,
    get_mapped_movie_genres
)
from models.contrastive_recommender import ContrastiveRecommender
from evaluation.metrics import (
    precision_at_k,
    ndcg_at_k,
    coverage
)


class FinalEvaluator:
    """Compare baseline vs contrastive on comprehensive metrics"""

    def __init__(self):
        self.logger = setup_logger("final_evaluation")
        self.recommender = None
        self.sampled_books = None

    def load_recommender(self):
        """Load contrastive recommender"""
        self.logger.info("\n" + "="*70)
        self.logger.info("LOADING RECOMMENDER")
        self.logger.info("="*70)

        with timer("Loading recommender", self.logger):
            self.recommender = ContrastiveRecommender()
            self.recommender.load_data()

        with timer("Loading projection head", self.logger):
            self.recommender.load_projection_head()
            self.recommender.load_projected_index()

    def sample_books(self, n: int = 500, seed: int = 42):
        """
        Sample books for evaluation.

        Args:
            n: Number of books to sample (default 500)
            seed: Random seed for reproducibility (default 42)
        """
        np.random.seed(seed)

        self.sampled_books = self.recommender.book_df.sample(n=min(n, len(self.recommender.book_df)))

        self.logger.info(f"\n📊 Sampled {len(self.sampled_books)} books (seed={seed})")

    def _find_relevant_movies(self, book_row: pd.Series) -> set:
        """
        Find relevant movies for a book based on genre overlap.

        Args:
            book_row: Book row from dataframe

        Returns:
            Set of relevant movie IDs
        """
        relevant_ids = set()

        try:
            # Get mapped movie genres
            book_movie_genres = get_mapped_movie_genres(book_row, self.recommender.genre_mapping)

            if not book_movie_genres:
                return relevant_ids

            # Find movies with genre overlap
            for movie_id, movie_row in self.recommender.movie_df.iterrows():
                movie_genres_str = str(movie_row.get('genres', ''))
                if movie_genres_str in ['', 'nan', '\\N']:
                    continue

                movie_genres = [g.strip() for g in movie_genres_str.split(',')]

                # Check if any genre matches
                if any(g in movie_genres for g in book_movie_genres):
                    relevant_ids.add(movie_id)

        except Exception as e:
            self.logger.warning(f"Error finding relevant movies: {e}")

        return relevant_ids

    def _compute_diversity(self, recommendations: pd.DataFrame, use_projection: bool) -> float:
        """
        Compute intra-list diversity as mean pairwise distance.

        Args:
            recommendations: DataFrame with movie_id column
            use_projection: Whether using projected or original embeddings

        Returns:
            Mean pairwise cosine distance (higher = more diverse)
        """
        if len(recommendations) < 2:
            return 0.0

        movie_ids = recommendations['movie_id'].tolist()

        # Get embeddings
        embeddings = []
        for movie_id in movie_ids:
            try:
                idx = self.recommender.movie_ids.index(movie_id)
                if use_projection:
                    # Use projected embeddings
                    if self.recommender.projected_movie_embeddings is None:
                        return 0.0
                    embeddings.append(self.recommender.projected_movie_embeddings[idx])
                else:
                    # Use original embeddings
                    embeddings.append(self.recommender.movie_embeddings[idx])
            except (ValueError, AttributeError):
                continue

        if len(embeddings) < 2:
            return 0.0

        embeddings = np.array(embeddings)

        # Compute pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                # Cosine distance = 1 - cosine_similarity
                cos_sim = np.dot(embeddings[i], embeddings[j])
                distances.append(1 - cos_sim)

        return float(np.mean(distances)) if distances else 0.0

    def evaluate_system(self, use_projection: bool, k: int = 10) -> Dict:
        """
        Evaluate a recommendation system.

        Args:
            use_projection: True for contrastive, False for baseline
            k: Number of recommendations

        Returns:
            Dict with metrics
        """
        system_name = "Contrastive" if use_projection else "Baseline"

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"EVALUATING {system_name.upper()}")
        self.logger.info(f"{'='*70}")

        precision_scores = []
        ndcg_scores = []
        all_recommended_ids = []
        diversity_scores = []
        n_evaluated = 0

        for _, book_row in tqdm(self.sampled_books.iterrows(),
                               total=len(self.sampled_books),
                               desc=f"Evaluating {system_name}"):
            try:
                book_title = book_row['Book']

                # Get recommendations
                recommendations = self.recommender.recommend_movies(
                    book_title,
                    k=k,
                    use_projection=use_projection
                )

                if len(recommendations) == 0:
                    continue

                n_evaluated += 1

                # Get recommended IDs
                recommended_ids = recommendations['movie_id'].tolist()
                all_recommended_ids.extend(recommended_ids)

                # Find relevant movies (genre overlap)
                relevant_ids = self._find_relevant_movies(book_row)

                # Compute metrics
                prec = precision_at_k(recommended_ids, relevant_ids, k)
                ndcg = ndcg_at_k(recommended_ids, relevant_ids, k)

                precision_scores.append(prec)
                ndcg_scores.append(ndcg)

                # Compute diversity
                diversity = self._compute_diversity(recommendations, use_projection)
                diversity_scores.append(diversity)

            except Exception as e:
                self.logger.warning(f"Error evaluating '{book_row['Book']}': {e}")
                continue

        # Aggregate metrics
        results = {
            'precision_at_k': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'ndcg_at_k': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            'coverage': float(coverage(all_recommended_ids, len(self.recommender.movie_df))),
            'diversity': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
            'n_evaluated': n_evaluated,
            'k': k
        }

        self.logger.info(f"\n✓ {system_name} Evaluation Complete:")
        self.logger.info(f"  Books evaluated: {n_evaluated}")
        self.logger.info(f"  Precision@{k}: {results['precision_at_k']:.4f}")
        self.logger.info(f"  nDCG@{k}: {results['ndcg_at_k']:.4f}")
        self.logger.info(f"  Coverage: {results['coverage']:.4f}")
        self.logger.info(f"  Diversity: {results['diversity']:.4f}")

        return results

    def compare_systems(self, k: int = 10) -> Dict:
        """
        Compare baseline vs contrastive.

        Args:
            k: Number of recommendations (default 10)

        Returns:
            Dict with both results + comparison
        """
        # Evaluate baseline
        baseline_results = self.evaluate_system(use_projection=False, k=k)

        # Evaluate contrastive
        contrastive_results = self.evaluate_system(use_projection=True, k=k)

        # Compute improvements
        improvements = {}
        for metric in ['precision_at_k', 'ndcg_at_k', 'coverage', 'diversity']:
            baseline_val = baseline_results[metric]
            contrastive_val = contrastive_results[metric]

            if baseline_val > 0:
                improvement = ((contrastive_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 0.0

            improvements[metric] = improvement

        return {
            'baseline': baseline_results,
            'contrastive': contrastive_results,
            'improvements_percent': improvements,
            'n_books_sampled': len(self.sampled_books),
            'seed': 42
        }

    def plot_comparison(self, results: Dict, output_path: Path):
        """
        Generate 4-panel bar chart comparison.

        Args:
            results: Results dict from compare_systems()
            output_path: Path to save plot
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("GENERATING COMPARISON PLOT")
        self.logger.info("="*70)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = ['precision_at_k', 'ndcg_at_k', 'coverage', 'diversity']
        titles = ['Precision@10', 'nDCG@10', 'Coverage', 'Diversity']
        colors = ['#3498db', '#2ecc71']

        for ax, metric, title in zip(axes.flat, metrics, titles):
            baseline_val = results['baseline'][metric]
            contrastive_val = results['contrastive'][metric]

            bars = ax.bar(['Baseline', 'Contrastive'], [baseline_val, contrastive_val],
                         color=colors, edgecolor='black', linewidth=1.5)

            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.set_ylim(0, max(baseline_val, contrastive_val) * 1.2)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)

            # Add improvement percentage
            improvement = results['improvements_percent'][metric]
            if improvement != 0:
                ax.text(0.5, max(baseline_val, contrastive_val) * 1.1,
                       f'{improvement:+.1f}%',
                       ha='center', fontsize=11, fontweight='bold',
                       color='green' if improvement > 0 else 'red')

        plt.suptitle('Baseline vs Contrastive Learning - Performance Comparison',
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()

        with timer("Saving plot", self.logger):
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        self.logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

    def print_summary(self, results: Dict):
        """
        Print summary report.

        Args:
            results: Results dict from compare_systems()
        """
        print("\n" + "="*70)
        print("FINAL EVALUATION SUMMARY")
        print("="*70)

        print(f"\n📊 Evaluation Details:")
        print(f"  Books sampled: {results['n_books_sampled']}")
        print(f"  Random seed: {results['seed']}")
        print(f"  Books evaluated (baseline): {results['baseline']['n_evaluated']}")
        print(f"  Books evaluated (contrastive): {results['contrastive']['n_evaluated']}")

        baseline = results['baseline']
        contrastive = results['contrastive']
        improvements = results['improvements_percent']

        print(f"\n📈 BASELINE RESULTS:")
        print(f"  Precision@10:  {baseline['precision_at_k']:.4f}")
        print(f"  nDCG@10:       {baseline['ndcg_at_k']:.4f}")
        print(f"  Coverage:      {baseline['coverage']:.4f}")
        print(f"  Diversity:     {baseline['diversity']:.4f}")

        print(f"\n🚀 CONTRASTIVE RESULTS:")
        print(f"  Precision@10:  {contrastive['precision_at_k']:.4f} ({improvements['precision_at_k']:+.1f}%)")
        print(f"  nDCG@10:       {contrastive['ndcg_at_k']:.4f} ({improvements['ndcg_at_k']:+.1f}%)")
        print(f"  Coverage:      {contrastive['coverage']:.4f} ({improvements['coverage']:+.1f}%)")
        print(f"  Diversity:     {contrastive['diversity']:.4f} ({improvements['diversity']:+.1f}%)")

        print("\n" + "="*70)

    def run(self):
        """Main evaluation pipeline"""
        self.logger.info("\n" + "="*70)
        self.logger.info("FINAL EVALUATION: BASELINE VS CONTRASTIVE")
        self.logger.info("="*70)

        # Load recommender
        self.load_recommender()

        # Sample books
        self.sample_books(n=500, seed=42)

        # Compare systems
        with timer("Running full evaluation", self.logger):
            results = self.compare_systems(k=10)

        # Save results
        results_path = PROCESSED_DATA_DIR / "final_evaluation_results.json"
        with timer("Saving results", self.logger):
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"  ✓ Saved to: {results_path}")

        # Generate plot
        plot_path = PLOTS_DIR / "final_evaluation_comparison.png"
        self.plot_comparison(results, plot_path)

        # Print summary
        self.print_summary(results)

        self.logger.info("\n" + "="*70)
        self.logger.info("✅ FINAL EVALUATION COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"\nResults: {results_path}")
        self.logger.info(f"Plot: {plot_path}")


def main():
    """Entry point"""
    evaluator = FinalEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
