"""
Quick verification script to check Sprint 2 outputs are valid.
Run this to verify everything is working before debugging further.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append('src')
from project_config import EMBEDDINGS_DIR, PROCESSED_DATA_DIR

def check_embeddings():
    """Verify embeddings exist and look reasonable"""
    print("\n" + "="*70)
    print("1. CHECKING EMBEDDINGS")
    print("="*70)

    files_to_check = [
        EMBEDDINGS_DIR / "book_embeddings.npy",
        EMBEDDINGS_DIR / "movie_embeddings.npy",
        EMBEDDINGS_DIR / "book_ids.json",
        EMBEDDINGS_DIR / "movie_ids.json",
        EMBEDDINGS_DIR / "embedding_metadata.json"
    ]

    all_exist = True
    for file_path in files_to_check:
        if file_path.exists():
            print(f"  ✓ {file_path.name} exists")
        else:
            print(f"  ✗ {file_path.name} MISSING")
            all_exist = False

    if not all_exist:
        print("\n  ❌ Some embedding files are missing!")
        print("  Run: python src/models/embedding_generator.py")
        return False

    # Load and check
    book_emb = np.load(EMBEDDINGS_DIR / "book_embeddings.npy")
    movie_emb = np.load(EMBEDDINGS_DIR / "movie_embeddings.npy")

    print(f"\n  Book embeddings: {book_emb.shape}")
    print(f"  Movie embeddings: {movie_emb.shape}")

    # Check norms
    book_norms = np.linalg.norm(book_emb, axis=1)
    movie_norms = np.linalg.norm(movie_emb, axis=1)

    book_norm_mean = book_norms.mean()
    movie_norm_mean = movie_norms.mean()

    print(f"\n  Book embedding norms: {book_norm_mean:.4f} (should be ~1.0)")
    print(f"  Movie embedding norms: {movie_norm_mean:.4f} (should be ~1.0)")

    if abs(book_norm_mean - 1.0) > 0.1 or abs(movie_norm_mean - 1.0) > 0.1:
        print("\n  ⚠️  WARNING: Embeddings are not normalized!")
        print("  This will cause poor similarity scores.")
        return False

    # Check for NaNs
    if np.isnan(book_emb).any():
        print("\n  ❌ Book embeddings contain NaNs!")
        return False
    if np.isnan(movie_emb).any():
        print("\n  ❌ Movie embeddings contain NaNs!")
        return False

    print("\n  ✓ Embeddings look valid")
    return True


def check_genre_mapping():
    """Verify genre mapping exists and is reasonable"""
    print("\n" + "="*70)
    print("2. CHECKING GENRE MAPPING")
    print("="*70)

    genre_mapping_file = PROCESSED_DATA_DIR / "genre_mapping.json"

    if not genre_mapping_file.exists():
        print(f"  ✗ {genre_mapping_file} MISSING")
        print("\n  Run Sprint 1: python src/data/genre_mapper.py")
        return False

    with open(genre_mapping_file, 'r') as f:
        mapping_list = json.load(f)

    print(f"  ✓ Genre mapping exists ({len(mapping_list)} book genres)")

    # Convert to dict
    mapping_dict = {item['book_genre']: item.get('movie_genres', []) for item in mapping_list}

    # Check how many are empty
    empty_count = sum(1 for v in mapping_dict.values() if len(v) == 0)
    empty_pct = (empty_count / len(mapping_dict)) * 100

    print(f"  Empty mappings: {empty_count}/{len(mapping_dict)} ({empty_pct:.1f}%)")

    if empty_pct > 50:
        print("\n  ⚠️  WARNING: >50% of genres have no mapping!")
        print("  This will cause low evaluation scores.")

    # Show some examples
    print("\n  Sample mappings:")
    for i, (bg, mg) in enumerate(list(mapping_dict.items())[:5]):
        status = "✓" if mg else "✗"
        print(f"    {status} {bg} → {mg}")

    return True


def check_faiss_index():
    """Verify FAISS index exists"""
    print("\n" + "="*70)
    print("3. CHECKING FAISS INDEX")
    print("="*70)

    index_file = EMBEDDINGS_DIR / "movie_faiss.index"

    if not index_file.exists():
        print(f"  ✗ {index_file} MISSING")
        print("\n  Run: python src/models/faiss_indexer.py")
        return False

    print(f"  ✓ FAISS index exists")

    # Try to load it
    try:
        import faiss
        index = faiss.read_index(str(index_file))
        print(f"  ✓ Index loaded successfully ({index.ntotal} vectors)")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load index: {e}")
        return False


def check_metadata():
    """Check embedding metadata for anomalies"""
    print("\n" + "="*70)
    print("4. CHECKING METADATA")
    print("="*70)

    metadata_file = EMBEDDINGS_DIR / "embedding_metadata.json"

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"  Model: {metadata['model_name']}")
    print(f"  Books: {metadata['n_books']:,}")
    print(f"  Movies: {metadata['n_movies']:,}")

    if 'similarity_stats' in metadata:
        stats = metadata['similarity_stats']
        print(f"\n  Similarity Statistics:")
        print(f"    Within-book: {stats['within_book_mean']:.4f}")
        print(f"    Within-movie: {stats['within_movie_mean']:.4f}")
        print(f"    Cross-domain: {stats['cross_domain_mean']:.4f}")
        print(f"    Gap: {stats['similarity_gap']:.4f} ({stats['similarity_gap_percent']:.1f}%)")

        # Check if values are reasonable
        if stats['within_book_mean'] < 0.3:
            print("\n  ⚠️  WARNING: Within-book similarity is very low (<0.3)")
            print("  Expected: 0.45-0.50. Embeddings may be poor quality.")

        if stats['within_movie_mean'] < 0.3:
            print("\n  ⚠️  WARNING: Within-movie similarity is very low (<0.3)")
            print("  Expected: 0.45-0.50. Embeddings may be poor quality.")

        if stats['similarity_gap_percent'] > 25:
            print("\n  ⚠️  WARNING: Similarity gap is large (>25%)")
            print("  This is higher than expected (~15-20%).")

    return True


def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("SPRINT 2 VERIFICATION")
    print("="*70)

    checks = [
        ("Embeddings", check_embeddings),
        ("Genre Mapping", check_genre_mapping),
        ("FAISS Index", check_faiss_index),
        ("Metadata", check_metadata),
    ]

    results = []
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ❌ {name} check failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = all(passed for _, passed in results)

    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    if all_passed:
        print("\n✅ All checks passed! Sprint 2 outputs look valid.")
        print("\nIf evaluation still fails, run:")
        print("  python debug_evaluation.py")
    else:
        print("\n❌ Some checks failed. Fix the issues above first.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
