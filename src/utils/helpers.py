"""
Shared utility functions for the cross-content recommendation system.
Provides logging, timing, data loading, and genre mapping utilities.
"""
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import ast

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from project_config import (
    GOODREADS_RAW_CSV,
    IMDB_FILTERED_CSV,
    GENRE_MAPPING_JSON,
    PROJECT_ROOT
)


def setup_logger(name: str = "recommendation_system", log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Set up a structured logger with timestamps that writes to both file and stdout.

    Args:
        name: Logger name (default: "recommendation_system")
        log_dir: Directory to save log files (default: PROJECT_ROOT/logs)

    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler with detailed format
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Logging to: {log_file}")

    return logger


@contextmanager
def timer(description: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks.

    Args:
        description: Description of the timed operation
        logger: Optional logger to write timing info to

    Example:
        with timer("Loading data", logger):
            data = load_large_file()
    """
    start_time = time.time()
    if logger:
        logger.info(f"⏱️  Starting: {description}")
    else:
        print(f"⏱️  Starting: {description}")

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60

        if minutes > 0:
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{elapsed:.2f}s"

        if logger:
            logger.info(f"✓ Completed: {description} (took {time_str})")
        else:
            print(f"✓ Completed: {description} (took {time_str})")


def load_datasets(logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both Goodreads and IMDB datasets with null-handling.

    Args:
        logger: Optional logger for progress messages

    Returns:
        Tuple of (goodreads_df, imdb_df)
    """
    if logger:
        logger.info("Loading datasets...")

    # Load Goodreads
    goodreads_df = pd.read_csv(GOODREADS_RAW_CSV, index_col=0)
    if logger:
        logger.info(f"  Loaded {len(goodreads_df):,} books from Goodreads")

    # Load IMDB
    imdb_df = pd.read_csv(IMDB_FILTERED_CSV, index_col=0)
    if logger:
        logger.info(f"  Loaded {len(imdb_df):,} movies from IMDB")

    # Handle nulls in critical columns
    # For descriptions - mark but don't drop yet (will drop during embedding generation)
    goodreads_df['Description'] = goodreads_df['Description'].fillna('')
    imdb_df['Description'] = imdb_df['Description'].fillna('')

    # For genres - replace nulls with empty string
    goodreads_df['Genres'] = goodreads_df['Genres'].fillna('[]')
    imdb_df['genres'] = imdb_df['genres'].fillna('')

    # Check for nulls
    gr_nulls = goodreads_df[['Book', 'Description', 'Genres']].isnull().sum().sum()
    imdb_nulls = imdb_df[['primaryTitle', 'Description', 'genres']].isnull().sum().sum()

    if gr_nulls > 0 and logger:
        logger.warning(f"  Goodreads has {gr_nulls} null values in critical columns")
    if imdb_nulls > 0 and logger:
        logger.warning(f"  IMDB has {imdb_nulls} null values in critical columns")

    return goodreads_df, imdb_df


def load_genre_mapping(logger: Optional[logging.Logger] = None) -> Dict[str, List[str]]:
    """
    Load genre mapping from JSON file.

    Args:
        logger: Optional logger for progress messages

    Returns:
        Dictionary mapping book genres to lists of movie genres
        Example: {"Fantasy": ["Fantasy", "Adventure"], "Thriller": ["Thriller", "Mystery"]}
    """
    if not GENRE_MAPPING_JSON.exists():
        error_msg = f"Genre mapping file not found at {GENRE_MAPPING_JSON}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(GENRE_MAPPING_JSON, 'r') as f:
        mapping_list = json.load(f)

    # Convert from list of dicts to simple dict
    mapping_dict = {}
    for item in mapping_list:
        book_genre = item['book_genre']
        movie_genres = item.get('movie_genres', [])
        mapping_dict[book_genre] = movie_genres

    if logger:
        logger.info(f"Loaded genre mapping for {len(mapping_dict)} book genres")
        high_conf = sum(1 for item in mapping_list if item.get('confidence') == 'high')
        logger.info(f"  {high_conf} high-confidence mappings ({high_conf/len(mapping_dict)*100:.1f}%)")

    return mapping_dict


def get_mapped_movie_genres(book_row: pd.Series, genre_mapping: Dict[str, List[str]]) -> List[str]:
    """
    Get all mapped movie genres for a book, handling multiple book genres.

    Args:
        book_row: Pandas Series with a 'Genres' column (string representation of list)
        genre_mapping: Dictionary mapping book genres to movie genres

    Returns:
        List of unique movie genres (union of all mappings)
    """
    try:
        # Parse genres string to list
        if isinstance(book_row['Genres'], str):
            book_genres = ast.literal_eval(book_row['Genres'])
        else:
            book_genres = []
    except:
        book_genres = []

    # Collect all mapped movie genres
    movie_genres = set()
    for book_genre in book_genres:
        mapped = genre_mapping.get(book_genre, [])
        movie_genres.update(mapped)

    return sorted(list(movie_genres))


def calculate_genre_overlap(book_genres: List[str], movie_genres: List[str]) -> float:
    """
    Calculate Jaccard similarity between book and movie genre sets.

    Args:
        book_genres: List of book genres
        movie_genres: List of movie genres

    Returns:
        Jaccard similarity (intersection over union), 0.0 to 1.0
    """
    if not book_genres or not movie_genres:
        return 0.0

    set_book = set(book_genres)
    set_movie = set(movie_genres)

    intersection = len(set_book & set_movie)
    union = len(set_book | set_movie)

    return intersection / union if union > 0 else 0.0


# Test functions
def test_logger():
    """Test the logger setup"""
    logger = setup_logger("test_logger")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    print("✓ Logger test complete")


def test_timer():
    """Test the timer context manager"""
    logger = setup_logger("timer_test")

    with timer("Quick operation", logger):
        time.sleep(0.5)

    with timer("Longer operation", logger):
        time.sleep(2.1)

    print("✓ Timer test complete")


def test_data_loading():
    """Test data loading"""
    logger = setup_logger("data_loading_test")

    with timer("Loading datasets", logger):
        goodreads_df, imdb_df = load_datasets(logger)

    print(f"\nGoodreads shape: {goodreads_df.shape}")
    print(f"IMDB shape: {imdb_df.shape}")
    print("\nGoodreads columns:", goodreads_df.columns.tolist())
    print("IMDB columns:", imdb_df.columns.tolist())
    print("\n✓ Data loading test complete")


def test_genre_mapping():
    """Test genre mapping"""
    logger = setup_logger("genre_mapping_test")

    genre_mapping = load_genre_mapping(logger)

    print(f"\nTotal genre mappings: {len(genre_mapping)}")
    print("\nSample mappings:")
    for i, (book_genre, movie_genres) in enumerate(list(genre_mapping.items())[:5]):
        print(f"  {book_genre} → {movie_genres}")

    # Test with actual book
    goodreads_df, _ = load_datasets()
    sample_book = goodreads_df.iloc[0]
    print(f"\nTest book: {sample_book['Book']}")
    print(f"  Book genres: {sample_book['Genres']}")

    mapped_genres = get_mapped_movie_genres(sample_book, genre_mapping)
    print(f"  Mapped movie genres: {mapped_genres}")

    print("\n✓ Genre mapping test complete")


def main():
    """Run all tests"""
    print("="*70)
    print("TESTING HELPER UTILITIES")
    print("="*70 + "\n")

    test_logger()
    print()

    test_timer()
    print()

    test_data_loading()
    print()

    test_genre_mapping()

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    main()
