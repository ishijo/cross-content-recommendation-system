"""
HuggingFace Hub data loader for cloud deployment.

Downloads all required data files from the HuggingFace dataset repo
to a local cache directory, then rebuilds FAISS indices from embeddings.

The cache directory mirrors the project's data structure so that
project_config.py constants resolve correctly in cloud mode.

Usage (standalone test):
    python src/utils/hf_data_loader.py

Usage (from app):
    from utils.hf_data_loader import load_all_data, build_faiss_indices_if_needed
    paths = load_all_data()
    build_faiss_indices_if_needed(paths["embeddings_dir"], paths["models_dir"])
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ID = "ishijo/cross-content-recommender-data"
REPO_TYPE = "dataset"

# HF repo path → local subpath (relative to cache root)
# local subpath is designed to mirror project_config expectations exactly
_DOWNLOAD_MAP: Dict[str, str] = {
    "data/goodreads_data.csv":               "data/raw/goodreads/goodreads_data.csv",
    "data/IMDB.csv":                          "data/raw/imdb_filtered/IMDB.csv",
    "data/genre_mapping.json":               "data/processed/genre_mapping.json",
    "data/genre_mapping_summary.csv":        "data/processed/genre_mapping_summary.csv",
    "embeddings/book_embeddings.npy":        "data/embeddings/book_embeddings.npy",
    "embeddings/movie_embeddings.npy":       "data/embeddings/movie_embeddings.npy",
    "embeddings/book_embeddings_projected.npy": "data/embeddings/book_embeddings_projected.npy",
    "embeddings/book_ids.json":              "data/embeddings/book_ids.json",
    "embeddings/movie_ids.json":             "data/embeddings/movie_ids.json",
    "embeddings/embedding_metadata.json":    "data/embeddings/embedding_metadata.json",
    "models/best_model.pt":                  "models/projection_head/best_model.pt",
    "models/training_config.json":           "models/projection_head/training_config.json",
}


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

def get_cache_dir() -> Path:
    """
    Return the local cache root for all downloaded data.

    Creates the directory if it does not exist.

    Returns:
        Path: ~/.cache/cross-content-recommender
    """
    cache = Path("~/.cache/cross-content-recommender").expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ---------------------------------------------------------------------------
# Single-file download
# ---------------------------------------------------------------------------

def download_file(
    repo_id: str,
    filename: str,
    local_path: Path,
    force: bool = False,
) -> Path:
    """
    Download a single file from the HuggingFace Hub.

    Uses hf_hub_download() with local caching. If the file already exists
    at local_path and force=False, the download is skipped.

    Args:
        repo_id:    HF repo id (e.g. "ishijo/cross-content-recommender-data")
        filename:   Path within the repo (e.g. "embeddings/book_embeddings.npy")
        local_path: Where to save the file on disk
        force:      Re-download even if the file already exists

    Returns:
        Path to the local file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for cloud deployment.\n"
            "Install with: pip install huggingface_hub"
        ) from e

    if local_path.exists() and not force:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    size_label = ""
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may save with a different name — move if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != local_path and downloaded_path.exists():
            downloaded_path.rename(local_path)
    except Exception:
        # Fallback: download directly to the exact path
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=REPO_TYPE,
        )
        import shutil
        shutil.copy2(tmp, str(local_path))

    return local_path


# ---------------------------------------------------------------------------
# Bulk download
# ---------------------------------------------------------------------------

def load_all_data(force_refresh: bool = False) -> Dict[str, Path]:
    """
    Download all required data files from HuggingFace Hub to the local cache.

    Files are downloaded only if they do not already exist in the cache
    (or if force_refresh=True).

    Args:
        force_refresh: Re-download all files even if already cached

    Returns:
        Dictionary of named paths:
        {
          "goodreads_csv": Path,
          "imdb_csv": Path,
          "genre_mapping_json": Path,
          "book_embeddings": Path,
          "movie_embeddings": Path,
          "book_embeddings_projected": Path,
          "book_ids": Path,
          "movie_ids": Path,
          "best_model_pt": Path,
          "embeddings_dir": Path,
          "models_dir": Path,
          "cache_root": Path,
        }
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    cache_root = get_cache_dir()
    t0 = time.time()

    print(f"[HF Loader] Cache: {cache_root}")
    print(f"[HF Loader] Repo:  https://huggingface.co/datasets/{REPO_ID}")
    print()

    downloaded_count = 0
    skipped_count = 0
    failed = []

    for hf_path, rel_local in _DOWNLOAD_MAP.items():
        local_path = cache_root / rel_local
        size_label = f"  ({local_path.stat().st_size / 1e6:.1f} MB)" if local_path.exists() else ""

        if local_path.exists() and not force_refresh:
            print(f"  ✓ Cached:    {rel_local}{size_label}")
            skipped_count += 1
            continue

        print(f"  ⬇ Downloading: {hf_path} ...", end="", flush=True)
        t1 = time.time()
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # hf_hub_download saves to HF's own cache; we copy to our cache
            import shutil
            tmp_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_path,
                repo_type=REPO_TYPE,
            )
            shutil.copy2(tmp_path, str(local_path))

            elapsed = time.time() - t1
            size_mb = local_path.stat().st_size / 1e6
            print(f" ✓  ({size_mb:.1f} MB, {elapsed:.1f}s)")
            downloaded_count += 1

        except Exception as e:
            print(f" ✗  FAILED: {e}")
            failed.append(hf_path)

    elapsed_total = time.time() - t0
    print()
    print(f"[HF Loader] Done — {downloaded_count} downloaded, {skipped_count} cached, "
          f"{len(failed)} failed  ({elapsed_total:.1f}s total)")

    if failed:
        print(f"[HF Loader] WARNING: {len(failed)} file(s) failed to download: {failed}")

    emb_dir = cache_root / "data" / "embeddings"
    mdl_dir = cache_root / "models" / "projection_head"

    return {
        "cache_root": cache_root,
        "goodreads_csv":              cache_root / "data" / "raw" / "goodreads" / "goodreads_data.csv",
        "imdb_csv":                   cache_root / "data" / "raw" / "imdb_filtered" / "IMDB.csv",
        "genre_mapping_json":         cache_root / "data" / "processed" / "genre_mapping.json",
        "book_embeddings":            emb_dir / "book_embeddings.npy",
        "movie_embeddings":           emb_dir / "movie_embeddings.npy",
        "book_embeddings_projected":  emb_dir / "book_embeddings_projected.npy",
        "book_ids":                   emb_dir / "book_ids.json",
        "movie_ids":                  emb_dir / "movie_ids.json",
        "best_model_pt":              mdl_dir / "best_model.pt",
        "embeddings_dir":             emb_dir,
        "models_dir":                 mdl_dir,
    }


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_indices_if_needed(
    embeddings_dir: Path,
    models_dir: Path,
    batch_size: int = 256,
) -> None:
    """
    Rebuild all four FAISS indices from downloaded embeddings if not cached.

    Indices built:
    1. book_faiss.index         — 768-dim baseline index for books
    2. movie_faiss.index        — 768-dim baseline index for movies/shows
    3. book_faiss_projected.index  — 128-dim contrastive index for books
    4. movie_faiss_projected.index — 128-dim contrastive index for movies/shows

    Building from embeddings is fast (<30s on CPU) and avoids storing large
    binary FAISS index files on the Hub.

    Args:
        embeddings_dir: Directory containing .npy embedding files
        models_dir:     Directory containing best_model.pt (for projected indices)
        batch_size:     Batch size when projecting movies through the projection head
    """
    try:
        import faiss
        import numpy as np
    except ImportError as e:
        raise ImportError("faiss-cpu and numpy are required.") from e

    embeddings_dir = Path(embeddings_dir)
    models_dir = Path(models_dir)

    print("[FAISS Builder] Building indices from embeddings...")

    # ---- 1. Baseline book index (768-dim) ----
    book_idx_path = embeddings_dir / "book_faiss.index"
    if book_idx_path.exists():
        print(f"  ✓ Exists: book_faiss.index")
    else:
        print("  Building book_faiss.index (768-dim) ...", end="", flush=True)
        embs = np.load(embeddings_dir / "book_embeddings.npy")
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs.astype("float32"))
        faiss.write_index(idx, str(book_idx_path))
        print(f" ✓  ({idx.ntotal} vectors)")

    # ---- 2. Baseline movie index (768-dim) ----
    movie_idx_path = embeddings_dir / "movie_faiss.index"
    if movie_idx_path.exists():
        print(f"  ✓ Exists: movie_faiss.index")
    else:
        print("  Building movie_faiss.index (768-dim) ...", end="", flush=True)
        embs = np.load(embeddings_dir / "movie_embeddings.npy")
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs.astype("float32"))
        faiss.write_index(idx, str(movie_idx_path))
        print(f" ✓  ({idx.ntotal} vectors)")

    # ---- 3. Projected book index (128-dim) ----
    book_proj_idx_path = embeddings_dir / "book_faiss_projected.index"
    if book_proj_idx_path.exists():
        print(f"  ✓ Exists: book_faiss_projected.index")
    else:
        proj_embs_path = embeddings_dir / "book_embeddings_projected.npy"
        if proj_embs_path.exists():
            print("  Building book_faiss_projected.index (128-dim) ...", end="", flush=True)
            embs = np.load(proj_embs_path)
            idx = faiss.IndexFlatIP(128)
            idx.add(embs.astype("float32"))
            faiss.write_index(idx, str(book_proj_idx_path))
            print(f" ✓  ({idx.ntotal} vectors)")
        else:
            print("  ⚠  book_embeddings_projected.npy not found — skipping projected book index")

    # ---- 4. Projected movie index (128-dim) ----
    # This index requires running inference through the PyTorch projection head.
    # We skip it here and let ContrastiveRecommender.load_projected_index()
    # build it at startup — that method already has build-on-demand logic
    # and is the authoritative owner of this index.
    movie_proj_idx_path = embeddings_dir / "movie_faiss_projected.index"
    if movie_proj_idx_path.exists():
        print(f"  ✓ Exists: movie_faiss_projected.index")
    else:
        print("  ↷ movie_faiss_projected.index — will be built by ContrastiveRecommender at startup")

    print("[FAISS Builder] All indices ready.")


def _build_projected_movie_index(
    embeddings_dir: Path,
    models_dir: Path,
    batch_size: int = 256,
) -> None:
    """
    Project movie embeddings through the trained head and build 128-dim FAISS index.

    Requires best_model.pt in models_dir.

    Args:
        embeddings_dir: Directory with movie_embeddings.npy
        models_dir:     Directory with best_model.pt
        batch_size:     Projection batch size
    """
    import faiss
    import numpy as np

    model_path = models_dir / "best_model.pt"
    if not model_path.exists():
        print(f"  ⚠  best_model.pt not found at {model_path} — skipping projected movie index")
        return

    try:
        import torch
    except ImportError:
        print("  ⚠  PyTorch not available — skipping projected movie index")
        return

    # Load ProjectionHead — add src to path if needed
    _src_dir = Path(__file__).parent.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    try:
        from models.projection_head import ProjectionHead
    except ImportError:
        print("  ⚠  Could not import ProjectionHead — skipping projected movie index")
        return

    print("  Building movie_faiss_projected.index (128-dim) ...", end="", flush=True)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    head = ProjectionHead(input_dim=768, hidden_dims=[512, 256], output_dim=128, dropout=0.2)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval().to(device)

    movie_embs = np.load(embeddings_dir / "movie_embeddings.npy")
    projected = []

    with torch.no_grad():
        for i in range(0, len(movie_embs), batch_size):
            batch = torch.from_numpy(movie_embs[i : i + batch_size]).float().to(device)
            out = head(batch)
            projected.append(out.cpu().numpy())

    projected_arr = np.vstack(projected)

    idx = faiss.IndexFlatIP(128)
    idx.add(projected_arr.astype("float32"))
    out_path = embeddings_dir / "movie_faiss_projected.index"
    faiss.write_index(idx, str(out_path))

    print(f" ✓  ({idx.ntotal} vectors, {time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HF Data Loader — standalone test")
    print("=" * 60)

    paths = load_all_data(force_refresh=False)

    print("\nLocal paths:")
    for k, v in paths.items():
        exists = "✓" if Path(v).exists() else "✗"
        print(f"  {exists} {k}: {v}")

    print("\nBuilding FAISS indices...")
    build_faiss_indices_if_needed(paths["embeddings_dir"], paths["models_dir"])

    print("\n✅ HF data loader test complete.")
