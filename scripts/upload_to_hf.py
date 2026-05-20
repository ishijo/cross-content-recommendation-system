"""
Upload all data files to HuggingFace Hub dataset repository.

Target repo: ishijo/cross-content-recommender-data

Files uploaded (HF repo path → local source):
  data/goodreads_data.csv        ← data/raw/goodreads/goodreads_data.csv
  data/IMDB.csv                  ← data/raw/imdb_filtered/IMDB.csv
  data/genre_mapping.json        ← data/processed/genre_mapping.json
  data/genre_mapping_summary.csv ← data/processed/genre_mapping_summary.csv
  embeddings/book_embeddings.npy ← data/embeddings/book_embeddings.npy
  embeddings/movie_embeddings.npy← data/embeddings/movie_embeddings.npy
  embeddings/book_embeddings_projected.npy ← data/embeddings/...
  embeddings/book_ids.json       ← data/embeddings/book_ids.json
  embeddings/movie_ids.json      ← data/embeddings/movie_ids.json
  embeddings/embedding_metadata.json ← data/embeddings/embedding_metadata.json
  models/best_model.pt           ← models/projection_head/best_model.pt
  models/training_config.json    ← models/projection_head/training_config.json

FAISS indices are NOT uploaded — they are rebuilt from embeddings at startup.

Usage:
    HF_TOKEN=hf_xxx python scripts/upload_to_hf.py
    # or set HF_TOKEN in .env
"""

import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from project_config import PROJECT_ROOT as _ROOT

REPO_ID = "ishijo/cross-content-recommender-data"
REPO_TYPE = "dataset"

# ---------------------------------------------------------------------------
# File manifest: (local_path, hf_repo_path)
# ---------------------------------------------------------------------------
FILES = [
    # Raw data
    (_ROOT / "data" / "raw" / "goodreads" / "goodreads_data.csv",
     "data/goodreads_data.csv"),
    (_ROOT / "data" / "raw" / "imdb_filtered" / "IMDB.csv",
     "data/IMDB.csv"),
    # Processed data
    (_ROOT / "data" / "processed" / "genre_mapping.json",
     "data/genre_mapping.json"),
    (_ROOT / "data" / "processed" / "genre_mapping_summary.csv",
     "data/genre_mapping_summary.csv"),
    # Embeddings
    (_ROOT / "data" / "embeddings" / "book_embeddings.npy",
     "embeddings/book_embeddings.npy"),
    (_ROOT / "data" / "embeddings" / "movie_embeddings.npy",
     "embeddings/movie_embeddings.npy"),
    (_ROOT / "data" / "embeddings" / "book_embeddings_projected.npy",
     "embeddings/book_embeddings_projected.npy"),
    (_ROOT / "data" / "embeddings" / "book_ids.json",
     "embeddings/book_ids.json"),
    (_ROOT / "data" / "embeddings" / "movie_ids.json",
     "embeddings/movie_ids.json"),
    (_ROOT / "data" / "embeddings" / "embedding_metadata.json",
     "embeddings/embedding_metadata.json"),
    # Model weights
    (_ROOT / "models" / "projection_head" / "best_model.pt",
     "models/best_model.pt"),
    (_ROOT / "models" / "projection_head" / "training_config.json",
     "models/training_config.json"),
]


def get_token() -> str:
    """
    Get HuggingFace token from environment or CLI cache.

    Checks in order:
    1. HF_TOKEN env var
    2. HUGGINGFACE_TOKEN env var
    3. huggingface-cli cached token (~/.cache/huggingface/token)
    4. .env file (loaded via python-dotenv)
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Try huggingface_hub's own cached token (set via `huggingface-cli login`)
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("  (Using token from huggingface-cli cache)")
            return token
    except Exception:
        pass

    print("ERROR: No HuggingFace token found.")
    print("  Option 1: export HF_TOKEN=hf_your_token_here")
    print("  Option 2: Add HF_TOKEN=hf_... to your .env file")
    print("  Option 3: Run `huggingface-cli login` first")
    sys.exit(1)


def ensure_repo_exists(api, token: str) -> None:
    """Create dataset repo if it doesn't exist yet."""
    from huggingface_hub import create_repo
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            exist_ok=True,
            private=False,
        )
        print(f"✓ Repo ready: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"  Note: {e}")


def upload_files(api, token: str) -> tuple[list, list, list]:
    """
    Upload all files in the manifest.

    Returns:
        (uploaded, skipped, failed) lists of HF repo paths
    """
    # Get existing files
    try:
        existing = set(api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))
        print(f"  Found {len(existing)} existing files in repo.")
    except Exception:
        existing = set()

    uploaded, skipped, failed = [], [], []

    for local_path, hf_path in FILES:
        local_path = Path(local_path)

        # Check local file exists
        if not local_path.exists():
            print(f"  ⚠  LOCAL FILE MISSING, skipping: {local_path}")
            failed.append(hf_path)
            continue

        size_mb = local_path.stat().st_size / 1_048_576

        # Skip if already on Hub
        if hf_path in existing:
            print(f"  ⏭  Skip (exists):   {hf_path}  ({size_mb:.1f} MB)")
            skipped.append(hf_path)
            continue

        # Upload
        print(f"  ⬆  Uploading:       {hf_path}  ({size_mb:.1f} MB) ...", end="", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
            )
            print(" ✓")
            uploaded.append(hf_path)
        except Exception as e:
            print(f" ✗ FAILED: {e}")
            failed.append(hf_path)

    return uploaded, skipped, failed


def print_summary(uploaded: list, skipped: list, failed: list) -> None:
    """Print upload summary."""
    total = len(uploaded) + len(skipped) + len(failed)
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"  Total files:  {total}")
    print(f"  Uploaded:     {len(uploaded)}")
    print(f"  Skipped:      {len(skipped)}")
    print(f"  Failed:       {len(failed)}")

    if uploaded:
        print("\nUploaded:")
        for f in uploaded:
            print(f"    ✓ {f}")

    if skipped:
        print("\nSkipped (already on Hub):")
        for f in skipped:
            print(f"    ⏭  {f}")

    if failed:
        print("\nFailed:")
        for f in failed:
            print(f"    ✗ {f}")

    print("\n" + "=" * 60)
    if not failed:
        print(f"✅ All done! View at: https://huggingface.co/datasets/{REPO_ID}")
    else:
        print(f"⚠️  {len(failed)} file(s) failed — check paths and re-run.")
    print("=" * 60)


def main():
    """Main entry point."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  Run: pip install huggingface_hub")
        sys.exit(1)

    print("=" * 60)
    print("HUGGINGFACE UPLOAD — Cross-Content Recommender Data")
    print("=" * 60)
    print(f"  Repo: {REPO_ID}")
    print(f"  Files to process: {len(FILES)}")
    print()

    token = get_token()
    api = HfApi()

    ensure_repo_exists(api, token)
    print()

    uploaded, skipped, failed = upload_files(api, token)
    print_summary(uploaded, skipped, failed)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
