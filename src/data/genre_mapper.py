"""
Genre Mapping Pipeline using Claude API
Maps Goodreads book genres (461) to IMDB movie genres (~25)
"""
import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm.asyncio import tqdm
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import ast

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from project_config import (
    GOODREADS_RAW_CSV,
    IMDB_FILTERED_CSV,
    GENRE_MAPPING_JSON,
    GENRE_MAPPING_CSV,
    CLAUDE_MODEL,
    GENRE_MAPPING_BATCH_SIZE,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY
)

# Load environment variables
load_dotenv()

class GenreMapper:
    """Maps book genres to movie genres using Claude API"""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.system_prompt = (
            "You are a precise genre taxonomy mapper. Always respond with valid JSON only. "
            "No preamble, no explanation, no markdown."
        )

    def load_genres(self) -> tuple[List[str], List[str], pd.DataFrame]:
        """Load unique genres from both datasets"""
        print("📚 Loading datasets...")

        # Load Goodreads
        goodreads_df = pd.read_csv(GOODREADS_RAW_CSV, index_col=0)
        book_genres_set = set()

        for genres_str in goodreads_df['Genres']:
            try:
                genres_list = ast.literal_eval(genres_str)
                book_genres_set.update(genres_list)
            except:
                pass

        book_genres = sorted(list(book_genres_set))
        print(f"  ✓ Found {len(book_genres)} unique book genres")

        # Load IMDB
        imdb_df = pd.read_csv(IMDB_FILTERED_CSV, index_col=0)
        movie_genres_set = set()

        for genres_str in imdb_df['genres']:
            if pd.notna(genres_str):
                genres_list = [g.strip() for g in str(genres_str).split(',')]
                movie_genres_set.update(genres_list)

        movie_genres = sorted(list(movie_genres_set))
        print(f"  ✓ Found {len(movie_genres)} unique movie genres")
        print(f"  Movie genres: {movie_genres}")

        return book_genres, movie_genres, goodreads_df

    async def map_single_genre(
        self,
        book_genre: str,
        movie_genres: List[str],
        retry_count: int = 0
    ) -> Dict:
        """Map a single book genre to movie genres using Claude API"""

        user_prompt = f"""Map this book genre to the most compatible IMDB movie genres from the provided list.

Book genre: {book_genre}
Available IMDB genres: {movie_genres}

Rules:
- Return 1-3 IMDB genres maximum
- Only use genres from the provided list
- If no good match exists return []
- Respond with JSON only: {{"book_genre": "...", "movie_genres": [...], "confidence": "high/medium/low"}}
"""

        try:
            response = await self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            # Parse JSON response
            content = response.content[0].text.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            # Validate result
            if "book_genre" not in result or "movie_genres" not in result:
                raise ValueError(f"Invalid response format: {result}")

            return result

        except Exception as e:
            if retry_count < API_RETRY_ATTEMPTS:
                wait_time = API_RETRY_DELAY * (2 ** retry_count)  # Exponential backoff
                print(f"  ⚠️  Error mapping '{book_genre}': {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self.map_single_genre(book_genre, movie_genres, retry_count + 1)
            else:
                print(f"  ❌ Failed to map '{book_genre}' after {API_RETRY_ATTEMPTS} retries: {e}")
                return {
                    "book_genre": book_genre,
                    "movie_genres": [],
                    "confidence": "failed",
                    "error": str(e)
                }

    async def map_genres_batch(
        self,
        book_genres: List[str],
        movie_genres: List[str]
    ) -> List[Dict]:
        """Map all book genres to movie genres in batches"""

        all_results = []
        total_batches = (len(book_genres) + GENRE_MAPPING_BATCH_SIZE - 1) // GENRE_MAPPING_BATCH_SIZE

        print(f"\n🎬 Mapping {len(book_genres)} genres in {total_batches} batches...")

        for i in range(0, len(book_genres), GENRE_MAPPING_BATCH_SIZE):
            batch = book_genres[i:i + GENRE_MAPPING_BATCH_SIZE]
            batch_num = i // GENRE_MAPPING_BATCH_SIZE + 1

            print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} genres)")

            # Create tasks for this batch
            tasks = [
                self.map_single_genre(genre, movie_genres)
                for genre in batch
            ]

            # Run batch concurrently with progress bar
            batch_results = []
            for task in tqdm.as_completed(tasks, desc=f"  Batch {batch_num}", total=len(tasks)):
                result = await task
                batch_results.append(result)

            all_results.extend(batch_results)

            # Delay between batches (except last one)
            if i + GENRE_MAPPING_BATCH_SIZE < len(book_genres):
                await asyncio.sleep(API_RETRY_DELAY)

        return all_results

    def calculate_genre_stats(
        self,
        mapping_results: List[Dict],
        goodreads_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate statistics about genre mapping coverage"""

        print("\n📊 Calculating coverage statistics...")

        # Count books per genre
        genre_counts = {}
        for genres_str in goodreads_df['Genres']:
            try:
                genres_list = ast.literal_eval(genres_str)
                for genre in genres_list:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            except:
                pass

        # Create summary dataframe
        summary_data = []
        for result in mapping_results:
            book_genre = result['book_genre']
            summary_data.append({
                'book_genre': book_genre,
                'mapped_movie_genres': ', '.join(result.get('movie_genres', [])),
                'num_mapped_genres': len(result.get('movie_genres', [])),
                'confidence': result.get('confidence', 'unknown'),
                'num_books_with_genre': genre_counts.get(book_genre, 0),
                'error': result.get('error', '')
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('num_books_with_genre', ascending=False)

        return summary_df

    def print_summary(self, summary_df: pd.DataFrame, goodreads_df: pd.DataFrame):
        """Print summary statistics"""

        print("\n" + "="*70)
        print("📈 GENRE MAPPING SUMMARY")
        print("="*70)

        # Confidence breakdown
        confidence_counts = summary_df['confidence'].value_counts()
        print("\n🎯 Confidence Distribution:")
        for conf, count in confidence_counts.items():
            pct = (count / len(summary_df)) * 100
            print(f"  {conf:10s}: {count:3d} genres ({pct:5.1f}%)")

        # Coverage analysis
        high_conf = summary_df[summary_df['confidence'] == 'high']
        total_books = len(goodreads_df)
        books_with_high_conf_mapping = high_conf['num_books_with_genre'].sum()
        coverage_pct = (books_with_high_conf_mapping / total_books) * 100

        print(f"\n📚 Coverage Analysis:")
        print(f"  Total books: {total_books:,}")
        print(f"  Books with high-confidence mapping: {books_with_high_conf_mapping:,} ({coverage_pct:.1f}%)")

        # Top mapped genres
        print(f"\n🔝 Top 10 Book Genres by Book Count:")
        top10 = summary_df.head(10)
        for idx, row in top10.iterrows():
            mapped = row['mapped_movie_genres'] if row['mapped_movie_genres'] else '(none)'
            print(f"  {row['book_genre']:30s} → {mapped:30s} [{row['confidence']}] ({row['num_books_with_genre']} books)")

        # Unmapped genres
        unmapped = summary_df[summary_df['num_mapped_genres'] == 0]
        if len(unmapped) > 0:
            print(f"\n⚠️  Unmapped Genres: {len(unmapped)}")
            print(f"  Books affected: {unmapped['num_books_with_genre'].sum():,}")

        print("\n" + "="*70)

    async def run(self):
        """Main execution pipeline"""

        # Check if mapping already exists
        if GENRE_MAPPING_JSON.exists():
            print(f"✓ Genre mapping already exists at {GENRE_MAPPING_JSON}")
            print("  Loading from file...")
            with open(GENRE_MAPPING_JSON, 'r') as f:
                mapping_results = json.load(f)

            # Still need to load data for statistics
            _, _, goodreads_df = self.load_genres()

        else:
            # Load genres
            book_genres, movie_genres, goodreads_df = self.load_genres()

            # Map genres using Claude API
            mapping_results = await self.map_genres_batch(book_genres, movie_genres)

            # Save JSON mapping
            print(f"\n💾 Saving mapping to {GENRE_MAPPING_JSON}...")
            with open(GENRE_MAPPING_JSON, 'w') as f:
                json.dump(mapping_results, f, indent=2)
            print("  ✓ Saved")

        # Calculate and save statistics
        summary_df = self.calculate_genre_stats(mapping_results, goodreads_df)

        print(f"\n💾 Saving summary to {GENRE_MAPPING_CSV}...")
        summary_df.to_csv(GENRE_MAPPING_CSV, index=False)
        print("  ✓ Saved")

        # Print summary
        self.print_summary(summary_df, goodreads_df)

        return summary_df


async def main():
    """Entry point"""
    mapper = GenreMapper()
    await mapper.run()


if __name__ == "__main__":
    asyncio.run(main())
