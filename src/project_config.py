"""
Configuration file with all project paths and constants.
Use these paths in all scripts to avoid hardcoding.
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Raw data files
GOODREADS_RAW_CSV = RAW_DATA_DIR / "goodreads" / "goodreads_data.csv"
IMDB_FILTERED_CSV = RAW_DATA_DIR / "imdb_filtered" / "IMDB.csv"
IMDB_BASICS_TSV = RAW_DATA_DIR / "imdb" / "title.basics.tsv" / "data.tsv"
IMDB_RATINGS_TSV = RAW_DATA_DIR / "imdb" / "title.ratings.tsv" / "data.tsv"

# Processed data files
GOODREADS_PREPROCESSED = PROCESSED_DATA_DIR / "goodreads_preprocessed"
IMDB_PREPROCESSED = PROCESSED_DATA_DIR / "imdb_preprocessed"
GOODREADS_GENRES_FREQ = PROCESSED_DATA_DIR / "goodreads_genres_freq"
IMDB_GENRES_FREQ = PROCESSED_DATA_DIR / "imdb_genres_freq"

# Genre mapping files
GENRE_MAPPING_JSON = PROCESSED_DATA_DIR / "genre_mapping.json"
GENRE_MAPPING_CSV = PROCESSED_DATA_DIR / "genre_mapping_summary.csv"

# Embeddings
GOODREADS_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "goodreads_embeddings.npy"
IMDB_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "imdb_embeddings.npy"
FAISS_INDEX_FILE = EMBEDDINGS_DIR / "combined_index.faiss"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
PROJECTION_HEAD_DIR = MODELS_DIR / "projection_head"
# Note: Sentence transformers will be downloaded via library, not stored in repo

# Output directories
PLOTS_DIR = PROJECT_ROOT / "plots"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_ARCHIVE_DIR = NOTEBOOKS_DIR / "archive"

# App directory
APP_DIR = PROJECT_ROOT / "app"

# Model hyperparameters
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"  # Model to download via sentence-transformers
EMBEDDING_DIM = 768  # all-mpnet-base-v2 dimension
PROJECTION_DIM = 256  # Projected dimension for contrastive learning
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# API settings
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GENRE_MAPPING_BATCH_SIZE = 20
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 1  # seconds

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR,
    MODELS_DIR, PLOTS_DIR, NOTEBOOKS_DIR, NOTEBOOKS_ARCHIVE_DIR,
    APP_DIR, PROJECTION_HEAD_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
