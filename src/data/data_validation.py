"""
Data Validation and EDA Report
Validates data quality and generates exploratory plots
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from project_config import (
    GOODREADS_RAW_CSV,
    IMDB_FILTERED_CSV,
    GENRE_MAPPING_JSON,
    GENRE_MAPPING_CSV,
    PLOTS_DIR
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DataValidator:
    """Validates data quality and generates EDA reports"""

    def __init__(self):
        self.plots_dir = PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load all datasets"""
        print("📚 Loading datasets...")

        self.goodreads_df = pd.read_csv(GOODREADS_RAW_CSV, index_col=0)
        print(f"  ✓ Goodreads: {len(self.goodreads_df):,} books")

        self.imdb_df = pd.read_csv(IMDB_FILTERED_CSV, index_col=0)
        print(f"  ✓ IMDB: {len(self.imdb_df):,} movies")

        if GENRE_MAPPING_JSON.exists():
            with open(GENRE_MAPPING_JSON, 'r') as f:
                self.genre_mapping = json.load(f)
            print(f"  ✓ Genre mapping: {len(self.genre_mapping)} mappings")
        else:
            print("  ⚠️  Genre mapping not found - run genre_mapper.py first")
            self.genre_mapping = None

        if GENRE_MAPPING_CSV.exists():
            self.mapping_summary = pd.read_csv(GENRE_MAPPING_CSV)
        else:
            self.mapping_summary = None

    def check_nulls(self):
        """Check for null values in critical columns"""
        print("\n🔍 Checking for null values...")

        issues = []

        # Goodreads checks
        gr_critical_cols = ['Book', 'Author', 'Description', 'Genres']
        gr_nulls = self.goodreads_df[gr_critical_cols].isnull().sum()

        print("\n  Goodreads:")
        for col, null_count in gr_nulls.items():
            pct = (null_count / len(self.goodreads_df)) * 100
            status = "✓" if null_count == 0 else "⚠️"
            print(f"    {status} {col:20s}: {null_count:5d} nulls ({pct:5.2f}%)")
            if null_count > 0:
                issues.append(f"Goodreads.{col} has {null_count} nulls ({pct:.1f}%)")

        # IMDB checks
        imdb_critical_cols = ['primaryTitle', 'Description', 'genres']
        imdb_nulls = self.imdb_df[imdb_critical_cols].isnull().sum()

        print("\n  IMDB:")
        for col, null_count in imdb_nulls.items():
            pct = (null_count / len(self.imdb_df)) * 100
            status = "✓" if null_count == 0 else "⚠️"
            print(f"    {status} {col:20s}: {null_count:5d} nulls ({pct:5.2f}%)")
            if null_count > 0:
                issues.append(f"IMDB.{col} has {null_count} nulls ({pct:.1f}%)")

        return issues

    def check_genre_coverage(self):
        """Check genre mapping coverage"""
        print("\n📊 Checking genre mapping coverage...")

        issues = []

        if self.mapping_summary is None:
            issues.append("Genre mapping summary not found")
            return issues

        # Calculate coverage
        high_conf = self.mapping_summary[self.mapping_summary['confidence'] == 'high']
        total_books = len(self.goodreads_df)
        books_covered = high_conf['num_books_with_genre'].sum()
        coverage_pct = (books_covered / total_books) * 100

        print(f"  Total books: {total_books:,}")
        print(f"  Books with high-confidence genre mapping: {books_covered:,}")
        print(f"  Coverage: {coverage_pct:.1f}%")

        if coverage_pct < 80:
            issues.append(f"Genre mapping coverage is {coverage_pct:.1f}% (target: 80%)")
            print(f"  ⚠️  Coverage below 80% threshold")
        else:
            print(f"  ✓ Coverage meets 80% threshold")

        return issues

    def analyze_description_lengths(self):
        """Analyze description length distributions"""
        print("\n📏 Analyzing description lengths...")

        # Calculate lengths
        gr_lengths = self.goodreads_df['Description'].dropna().apply(len)
        imdb_lengths = self.imdb_df['Description'].dropna().apply(len)

        print(f"\n  Goodreads descriptions:")
        print(f"    Mean: {gr_lengths.mean():.0f} chars")
        print(f"    Median: {gr_lengths.median():.0f} chars")
        print(f"    Range: {gr_lengths.min():.0f} - {gr_lengths.max():.0f} chars")

        print(f"\n  IMDB descriptions:")
        print(f"    Mean: {imdb_lengths.mean():.0f} chars")
        print(f"    Median: {imdb_lengths.median():.0f} chars")
        print(f"    Range: {imdb_lengths.min():.0f} - {imdb_lengths.max():.0f} chars")

        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(gr_lengths, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
        axes[0].set_xlabel('Description Length (characters)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Goodreads Description Lengths')
        axes[0].axvline(gr_lengths.mean(), color='red', linestyle='--', label=f'Mean: {gr_lengths.mean():.0f}')
        axes[0].legend()

        axes[1].hist(imdb_lengths, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        axes[1].set_xlabel('Description Length (characters)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('IMDB Description Lengths')
        axes[1].axvline(imdb_lengths.mean(), color='red', linestyle='--', label=f'Mean: {imdb_lengths.mean():.0f}')
        axes[1].legend()

        plt.tight_layout()
        output_path = self.plots_dir / 'description_length_distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  💾 Saved plot: {output_path}")
        plt.close()

    def analyze_top_genres(self):
        """Analyze top genres in both datasets"""
        print("\n🎭 Analyzing top genres...")

        # Goodreads genres
        gr_genre_counter = Counter()
        for genres_str in self.goodreads_df['Genres']:
            try:
                genres_list = ast.literal_eval(genres_str)
                gr_genre_counter.update(genres_list)
            except:
                pass

        # IMDB genres
        imdb_genre_counter = Counter()
        for genres_str in self.imdb_df['genres']:
            if pd.notna(genres_str):
                genres_list = [g.strip() for g in str(genres_str).split(',')]
                imdb_genre_counter.update(genres_list)

        # Get top 20 for each
        top_gr = gr_genre_counter.most_common(20)
        top_imdb = imdb_genre_counter.most_common(20)

        print(f"\n  Top 10 Goodreads genres:")
        for genre, count in top_gr[:10]:
            print(f"    {genre:30s}: {count:5d}")

        print(f"\n  Top 10 IMDB genres:")
        for genre, count in top_imdb[:10]:
            print(f"    {genre:30s}: {count:5d}")

        # Plot top genres
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Goodreads
        genres_gr, counts_gr = zip(*top_gr)
        axes[0].barh(range(len(genres_gr)), counts_gr, color='#2ecc71')
        axes[0].set_yticks(range(len(genres_gr)))
        axes[0].set_yticklabels(genres_gr)
        axes[0].set_xlabel('Count')
        axes[0].set_title('Top 20 Goodreads Genres')
        axes[0].invert_yaxis()

        # IMDB
        genres_imdb, counts_imdb = zip(*top_imdb)
        axes[1].barh(range(len(genres_imdb)), counts_imdb, color='#3498db')
        axes[1].set_yticks(range(len(genres_imdb)))
        axes[1].set_yticklabels(genres_imdb)
        axes[1].set_xlabel('Count')
        axes[1].set_title('Top 20 IMDB Genres')
        axes[1].invert_yaxis()

        plt.tight_layout()
        output_path = self.plots_dir / 'top_genres_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  💾 Saved plot: {output_path}")
        plt.close()

        return gr_genre_counter, imdb_genre_counter

    def create_genre_overlap_heatmap(self, gr_genres, imdb_genres):
        """Create heatmap showing genre overlap/mapping"""
        print("\n🗺️  Creating genre overlap heatmap...")

        if self.genre_mapping is None:
            print("  ⚠️  Skipping - genre mapping not found")
            return

        # Get top 15 genres from each
        top_gr_genres = [g for g, _ in gr_genres.most_common(15)]
        all_imdb_genres = sorted(list(imdb_genres.keys()))

        # Create mapping matrix
        mapping_dict = {m['book_genre']: m.get('movie_genres', []) for m in self.genre_mapping}

        matrix = np.zeros((len(top_gr_genres), len(all_imdb_genres)))

        for i, book_genre in enumerate(top_gr_genres):
            mapped_genres = mapping_dict.get(book_genre, [])
            for j, movie_genre in enumerate(all_imdb_genres):
                if movie_genre in mapped_genres:
                    matrix[i, j] = 1

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            xticklabels=all_imdb_genres,
            yticklabels=top_gr_genres,
            cmap='YlOrRd',
            cbar_kws={'label': 'Mapped (1) or Not (0)'},
            linewidths=0.5,
            linecolor='gray'
        )
        plt.xlabel('IMDB Movie Genres')
        plt.ylabel('Goodreads Book Genres (Top 15)')
        plt.title('Genre Mapping Heatmap: Books → Movies')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = self.plots_dir / 'genre_overlap_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved plot: {output_path}")
        plt.close()

    def run(self):
        """Run full validation pipeline"""
        print("\n" + "="*70)
        print("🔬 DATA VALIDATION & EDA REPORT")
        print("="*70)

        # Load data
        self.load_data()

        # Run checks
        all_issues = []

        # 1. Check nulls
        null_issues = self.check_nulls()
        all_issues.extend(null_issues)

        # 2. Check genre mapping coverage
        coverage_issues = self.check_genre_coverage()
        all_issues.extend(coverage_issues)

        # 3. Analyze description lengths
        self.analyze_description_lengths()

        # 4. Analyze top genres
        gr_genres, imdb_genres = self.analyze_top_genres()

        # 5. Create overlap heatmap
        self.create_genre_overlap_heatmap(gr_genres, imdb_genres)

        # Final verdict
        print("\n" + "="*70)
        print("🏁 FINAL VERDICT")
        print("="*70)

        if len(all_issues) == 0:
            print("\n✅ GO - All checks passed!")
            print("   Data quality is sufficient to proceed to Sprint 2 (embedding generation)")
            verdict = "GO"
        else:
            print(f"\n⚠️  {len(all_issues)} issue(s) found:")
            for i, issue in enumerate(all_issues, 1):
                print(f"   {i}. {issue}")

            # Determine if issues are blockers
            critical_issues = [i for i in all_issues if 'coverage' in i.lower() or 'mapping not found' in i.lower()]

            if len(critical_issues) > 0:
                print("\n❌ NO-GO - Critical issues must be resolved")
                verdict = "NO-GO"
            else:
                print("\n⚠️  CONDITIONAL GO - Minor issues present but not blocking")
                print("   Can proceed with caution")
                verdict = "CONDITIONAL GO"

        print("\n" + "="*70)
        print(f"\n📁 All plots saved to: {self.plots_dir}")
        print("="*70 + "\n")

        return verdict, all_issues


def main():
    """Entry point"""
    validator = DataValidator()
    verdict, issues = validator.run()
    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict == "GO" else 1)
