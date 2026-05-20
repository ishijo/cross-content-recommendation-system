"""
Configuration file with all project paths and constants.

Supports two modes automatically:
  - Local development: data lives under PROJECT_ROOT/data/ and models/
  - Cloud (Streamlit Cloud): data downloaded from HuggingFace to
    ~/.cache/cross-content-recommender/ by hf_data_loader.py

Mode selection is automatic: if the local Goodreads CSV exists, use local
paths; otherwise fall back to the HuggingFace cache directory.

Use these constants in all scripts — never hardcode paths directly.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed roots — never change
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent          # repo root (always local)
HF_CACHE_ROOT = Path.home() / ".cache" / "cross-content-recommender"

# ---------------------------------------------------------------------------
# Auto-detect: local dev vs cloud
# ---------------------------------------------------------------------------
_LOCAL_CSV = PROJECT_ROOT / "data" / "raw" / "goodreads" / "goodreads_data.csv"
IS_CLOUD = not _LOCAL_CSV.exists()

# Data lives under PROJECT_ROOT locally, under HF_CACHE_ROOT on cloud
DATA_ROOT = HF_CACHE_ROOT if IS_CLOUD else PROJECT_ROOT

# ---------------------------------------------------------------------------
# Data directories  (may be local OR in HF cache)
# ---------------------------------------------------------------------------
DATA_DIR = DATA_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ---------------------------------------------------------------------------
# Raw data files
# ---------------------------------------------------------------------------
GOODREADS_RAW_CSV = RAW_DATA_DIR / "goodreads" / "goodreads_data.csv"
IMDB_FILTERED_CSV = RAW_DATA_DIR / "imdb_filtered" / "IMDB.csv"
IMDB_BASICS_TSV = RAW_DATA_DIR / "imdb" / "title.basics.tsv" / "data.tsv"
IMDB_RATINGS_TSV = RAW_DATA_DIR / "imdb" / "title.ratings.tsv" / "data.tsv"

# ---------------------------------------------------------------------------
# Processed data files
# ---------------------------------------------------------------------------
GOODREADS_PREPROCESSED = PROCESSED_DATA_DIR / "goodreads_preprocessed"
IMDB_PREPROCESSED = PROCESSED_DATA_DIR / "imdb_preprocessed"
GOODREADS_GENRES_FREQ = PROCESSED_DATA_DIR / "goodreads_genres_freq"
IMDB_GENRES_FREQ = PROCESSED_DATA_DIR / "imdb_genres_freq"

GENRE_MAPPING_JSON = PROCESSED_DATA_DIR / "genre_mapping.json"
GENRE_MAPPING_CSV = PROCESSED_DATA_DIR / "genre_mapping_summary.csv"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
GOODREADS_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "goodreads_embeddings.npy"
IMDB_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "imdb_embeddings.npy"
FAISS_INDEX_FILE = EMBEDDINGS_DIR / "combined_index.faiss"

# Sprint 4A — bidirectional FAISS indices and embeddings
BOOK_FAISS_INDEX = EMBEDDINGS_DIR / "book_faiss.index"
BOOK_FAISS_PROJECTED_INDEX = EMBEDDINGS_DIR / "book_faiss_projected.index"
BOOK_EMBEDDINGS_PROJECTED_NPY = EMBEDDINGS_DIR / "book_embeddings_projected.npy"
BIDIRECTIONAL_EVAL_RESULTS = PROCESSED_DATA_DIR / "bidirectional_evaluation_results.json"

# ---------------------------------------------------------------------------
# Model directories  (may be local OR in HF cache)
# ---------------------------------------------------------------------------
MODELS_DIR = DATA_ROOT / "models"
PROJECTION_HEAD_DIR = MODELS_DIR / "projection_head"

# ---------------------------------------------------------------------------
# Output / code dirs  (always local — not used on cloud)
# ---------------------------------------------------------------------------
PLOTS_DIR = PROJECT_ROOT / "plots"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_ARCHIVE_DIR = NOTEBOOKS_DIR / "archive"
APP_DIR = PROJECT_ROOT / "app"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
PROJECTION_DIM = 128   # Actual output dim of trained projection head
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # Cost-efficient for explanations
GENRE_MAPPING_BATCH_SIZE = 20
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 1  # seconds

# ---------------------------------------------------------------------------
# HuggingFace Hub settings
# ---------------------------------------------------------------------------
HF_REPO_ID = "ishijo/cross-content-recommender-data"
HF_REPO_TYPE = "dataset"

# ---------------------------------------------------------------------------
# Ensure writable directories exist
# ---------------------------------------------------------------------------
for _d in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR,
    MODELS_DIR, PROJECTION_HEAD_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# Local-only dirs (safe to skip creation on cloud)
for _d in [PLOTS_DIR, NOTEBOOKS_DIR, APP_DIR]:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
