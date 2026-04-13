# Sprint 3: Contrastive Learning Implementation Plan

## Overview

Build a contrastive learning system to improve cross-domain book-to-movie recommendations by training projection heads that map 768-dim embeddings to 128-dim aligned space, reducing the current 29.7% similarity gap.

## Architecture Summary

```
Raw Data (Sprints 1-2) → [TASK 1] Contrastive Pairs → [TASK 2] Train Projection Head
                                                              ↓
                                                    models/projection_head/
                                                              ↓
                                                    [TASK 3] Contrastive Recommender
                                                              ↓
                                            ┌─────────────────┴─────────────────┐
                                            ↓                                   ↓
                                    [TASK 4] Streamlit App        [TASK 5] Evaluation
```

## Key Design Decisions ✓ User Confirmed

1. **Shared Projection Head** ✓: Single MLP for both books and movies (same embedding space, simpler, ~1M params)
2. **Memory Strategy**: Load all embeddings into memory (~50MB, fast access)
3. **Hard Negatives in Loss** ✓: Pre-mine during dataset construction, include in InfoNCE loss alongside in-batch negatives
4. **Two FAISS Indices**: Maintain 768-dim (baseline) and 128-dim (contrastive) for comparison
5. **InfoNCE Loss** ✓: In-batch negatives (batch_size - 1) + 1 hard negative per sample in denominator
6. **Claude API Optional** ✓: Graceful degradation - app works without API key, shows recommendations without explanations
7. **Backward Compatible**: ContrastiveRecommender extends BaselineRecommender

---

## TASK 1: Contrastive Dataset Builder

**File:** `src/data/contrastive_dataset.py`

### Implementation

```python
class ContrastivePairBuilder:
    def __init__(self):
        - Load book/movie CSVs, embeddings (9,923 books, 7,662 movies)
        - Load genre_mapping.json
        - Build temporary FAISS index for hard negative mining

    def find_positive_pairs() -> List[Tuple[int, int]]:
        - For each book, get mapped movie genres
        - Find movies sharing ≥1 genre
        - Sample up to 5 movies per book (stratify by rating)
        - Return list of (book_idx, movie_idx) pairs
        - Expected: ~30,000-40,000 pairs

    def find_hard_negatives(positive_pairs) -> Dict:
        - For each (book_idx, movie_pos_idx):
          * Query FAISS with book embedding → top 50 similar movies
          * Filter out genre-compatible movies
          * Select highest-scoring genre-incompatible movie
        - Return dict: {f"{book_idx}_{movie_pos_idx}": movie_neg_idx}
        - Fallback to random if no hard negatives found

    def save_pairs(pairs_data) -> None:
        - Save to data/processed/contrastive_pairs.json
        - Format: {positive_pairs: [[book_idx, movie_idx], ...],
                   hard_negatives: {"23_456": 1234, ...},
                   stats: {...}}
```

**Output:** `data/processed/contrastive_pairs.json` with pair indices and statistics

---

## TASK 2: Projection Head Training

**File:** `src/models/projection_head.py`

### Architecture

```python
class ProjectionHead(nn.Module):
    - Layer 1: Linear(768 → 512) → BatchNorm1d → ReLU → Dropout(0.2)
    - Layer 2: Linear(512 → 256) → BatchNorm1d → ReLU → Dropout(0.2)
    - Layer 3: Linear(256 → 128)
    - Output: L2-normalize to unit sphere

    forward(x): Returns (batch_size, 128) L2-normalized embeddings

class InfoNCELoss(nn.Module):
    - Temperature: 0.07
    - For each batch:
      * Positive similarity: dot(book_proj, movie_pos_proj) / τ
      * In-batch negatives: dot(book_proj, all_other_movie_proj) / τ
      * Hard negative: dot(book_proj, movie_neg_proj) / τ
      * Loss: -log(exp(pos_sim) / sum(exp(all_sim)))

class ContrastiveDataset(torch.utils.data.Dataset):
    - Load embeddings into memory at __init__
    - __getitem__(idx): Return (book_emb[book_idx],
                                 movie_emb[movie_pos_idx],
                                 movie_emb[movie_neg_idx])
    - 80/20 train/val split stratified by book_id

class ContrastiveTrainer:
    - Optimizer: Adam(lr=1e-4, weight_decay=1e-5)
    - Scheduler: CosineAnnealingLR(T_max=epochs)
    - Batch size: 256
    - Early stopping: patience=5 on val_loss
    - Log every epoch: train_loss, val_loss
    - Save best model: models/projection_head/best_model.pt
```

### Training Loop

1. Load contrastive_pairs.json
2. Create ContrastiveDataset with embeddings in memory
3. Split 80/20 train/val, create DataLoaders (num_workers=4)
4. Train for max 50 epochs with early stopping
5. Compute similarity gap: mean cosine sim of positive pairs before vs after training
6. Evaluate Precision@10 on 200-book sample
7. Generate plots/contrastive_training_curves.png (train/val loss curves)
8. Save training_config.json

**Outputs:**
- `models/projection_head/best_model.pt` (checkpoint dict with model_state_dict, optimizer_state_dict, epoch, val_loss, config)
- `models/projection_head/training_config.json`
- `plots/contrastive_training_curves.png`

---

## TASK 3: Contrastive Recommender

**File:** `src/models/contrastive_recommender.py`

### Implementation

```python
class ContrastiveRecommender(BaselineRecommender):
    """Extends baseline with projection head capabilities"""

    def __init__(self):
        super().__init__()  # Inherit all baseline functionality
        self.projection_head = None
        self.projected_movie_index = None  # 128-dim FAISS IndexFlatIP
        self.projected_movie_embeddings = None

    def load_projection_head(model_path=None):
        - Load checkpoint from models/projection_head/best_model.pt
        - Initialize ProjectionHead, load state_dict
        - Set to eval mode

    def project_book(book_embedding) -> np.ndarray:
        - Project (768,) → (128,) using trained head
        - Return L2-normalized numpy array

    def build_projected_movie_index():
        - Project all movie embeddings in batches (256 at a time)
        - Store in self.projected_movie_embeddings (7,662 × 128)
        - Build FAISS IndexFlatIP(128)
        - Save to data/embeddings/movie_faiss_projected.index

    def recommend_movies(book_title, k=10, genre_filter=True,
                        genre_boost=0.2, use_projection=False):
        - If use_projection=False: Call super().recommend_movies()
        - If use_projection=True:
          * Get book embedding, project to 128-dim
          * Search projected_movie_index for top 50 candidates
          * Re-rank: final_score = projected_cosine_sim + 0.2 × genre_overlap
          * Return top k as DataFrame (same schema as baseline)

    def compare_baseline_vs_contrastive(book_title, k=5):
        - Get recommendations from both methods
        - Return (baseline_df, contrastive_df) for side-by-side comparison
```

**Inheritance Pattern:** Extends BaselineRecommender → No code duplication, maintains compatibility

**Output:** `data/embeddings/movie_faiss_projected.index` (128-dim FAISS index)

---

## TASK 4: Streamlit App

**File:** `app/main.py` (complete rewrite)

### UI Components

**Sidebar:**
- Model toggle: "Baseline" vs "Contrastive Learning" (radio buttons)
- K slider: 5-20 recommendations (default 10)
- Genre filter: Checkbox (default True)
- API key input: Text input (password, optional)

**Main Area:**
- Title: "🎬 Cross-Content Movie Recommendations"
- Text input: Book title
- Button: "🔍 Find Movies"
- Results: Expandable movie cards (top 3 expanded by default)

### Implementation

```python
@st.cache_resource
def load_recommender():
    - Load ContrastiveRecommender
    - Try to load projection head (catch errors → use baseline only)
    - Return (recommender, has_projection_flag)

@st.cache_data(ttl=3600)
def generate_explanation(book_title, movie_title, api_key):
    - Call Claude API (claude-haiku-4-5-20251001)
    - Prompt: "Explain in 2 sentences why reader of {book} would enjoy {movie}"
    - Return explanation text (cached for 1 hour)

def main():
    - Load recommender with caching
    - Render sidebar controls
    - On "Find Movies" click:
      * Get recommendations (use_projection based on toggle)
      * Store in session_state
      * Display book details + movie cards
    - For each movie card:
      * Similarity progress bar
      * Genres, rating, description
      * Claude API explanation (if API key provided)
```

### Error Handling

- **Book not found:** Suggest 3 similar titles using partial string match
- **API key missing:** Show recommendations without explanations + info message
- **Projection head missing:** Warn user, fall back to baseline mode
- **API rate limit:** Use cached results, inform user

**Outputs:** Running Streamlit app at http://localhost:8501

---

## TASK 5: Final Evaluation

**File:** `src/evaluation/final_evaluation.py`

### Implementation

```python
class FinalEvaluator:
    def __init__(self):
        - Setup logger
        - Load ContrastiveRecommender with projection head

    def sample_books(n=500, seed=42):
        - Sample 500 books randomly (reproducible)

    def evaluate_system(use_projection, k=10) -> Dict:
        - For each sampled book:
          * Get k recommendations
          * Calculate relevant movies (genre overlap)
          * Compute: Precision@10, nDCG@10
          * Track all recommended IDs for coverage
          * Compute diversity (mean pairwise cosine distance in recommendation list)
        - Return aggregated metrics

    def _compute_diversity(recommendations, use_projection):
        - Get embeddings (projected or original)
        - Compute mean pairwise cosine distance
        - Higher = more diverse

    def compare_systems(k=10) -> Dict:
        - Evaluate baseline (use_projection=False)
        - Evaluate contrastive (use_projection=True)
        - Compute percent improvements
        - Return full comparison

    def plot_comparison(results, output_path):
        - Generate 4-panel bar chart (2×2 grid)
        - Metrics: Precision@10, nDCG@10, Coverage, Diversity
        - Side-by-side bars: Baseline vs Contrastive
        - Add value labels on bars
        - Save to plots/final_evaluation_comparison.png

    def print_summary(results):
        - Print formatted table with both systems
        - Show absolute values + percent improvements
```

### Metrics

1. **Precision@10:** Fraction of top 10 sharing genres with book
2. **nDCG@10:** Normalized ranking quality (binary relevance)
3. **Coverage:** Fraction of movie catalog recommended across all queries
4. **Diversity:** Mean intra-list pairwise distance (1 - cosine_similarity)

**Outputs:**
- `data/processed/final_evaluation_results.json`
- `plots/final_evaluation_comparison.png`
- Console summary report

---

## Critical Files to Modify/Create

### New Files
1. `src/data/contrastive_dataset.py` - Dataset builder
2. `src/models/projection_head.py` - Model + training
3. `src/models/contrastive_recommender.py` - Upgraded recommender
4. `app/main.py` - Complete rewrite
5. `src/evaluation/final_evaluation.py` - Comparative evaluation

### Files to Reference (No Changes)
- `src/models/baseline_recommender.py` - Parent class, method signatures
- `src/models/faiss_indexer.py` - FAISS patterns
- `src/utils/helpers.py` - Genre utilities
- `src/evaluation/metrics.py` - Metric functions
- `src/project_config.py` - Path constants

### Update Requirements
- Verify `torch==2.2.2` and `anthropic==0.40.0` in requirements.txt (already present)

---

## Execution Order

```bash
# Step 1: Build training dataset
python src/data/contrastive_dataset.py
# Expected: ~30-40k pairs, prints stats

# Step 2: Train projection head (10-20 min)
python src/models/projection_head.py
# Expected: Best val_loss, similarity gap improvement

# Step 3: Build projected index
python src/models/contrastive_recommender.py
# Expected: 128-dim FAISS index created

# Step 4: Launch Streamlit app
streamlit run app/main.py
# Test both baseline and contrastive modes

# Step 5: Run full evaluation
python src/evaluation/final_evaluation.py
# Expected: Comparison metrics, plots
```

---

## Verification Steps

### After Task 1:
- [ ] `data/processed/contrastive_pairs.json` exists
- [ ] Stats show ~30-40k positive pairs, positive/negative ratio ≈ 1:1
- [ ] Coverage: 80%+ of books have pairs

### After Task 2:
- [ ] `models/projection_head/best_model.pt` exists (~5MB)
- [ ] Training curves show convergence
- [ ] Similarity gap improvement > 10%
- [ ] Precision@10 on sample > baseline

### After Task 3:
- [ ] `data/embeddings/movie_faiss_projected.index` exists
- [ ] Can recommend with use_projection=True
- [ ] Output schema matches baseline exactly

### After Task 4:
- [ ] Streamlit app runs without errors
- [ ] Both model modes work
- [ ] Claude explanations generate (with API key)
- [ ] Error handling works (invalid book, no API key)

### After Task 5:
- [ ] Evaluation completes on 500 books
- [ ] Contrastive shows improvement over baseline
- [ ] Plots and JSON results saved
- [ ] Summary prints clean comparison table

---

## Expected Performance Improvements

Based on similar contrastive learning systems:

- **Precision@10:** +5-15% improvement (from 99.83% baseline, likely saturation)
- **nDCG@10:** +5-15% improvement (from 99.86% baseline)
- **Coverage:** +10-20% (contrastive learning encourages diversity)
- **Diversity:** +15-30% (reduced embedding collapse)

**Note:** Baseline performance (99.8%+) is inflated due to broad genre overlap proxy. Contrastive learning should provide better fine-grained alignment beyond just genre matching.

---

## Design Decisions Rationale

1. **Shared Projection Head:** Books and movies use same 768-dim model → single head is simpler and sufficient

2. **InfoNCE with Hard Negatives:** Combines scalability (in-batch) with difficulty (hard negatives) for better learning signal

3. **Extend BaselineRecommender:** Code reuse, backward compatibility, easy A/B testing

4. **Two FAISS Indices:** Maintains baseline for comparison without breaking existing functionality

5. **Memory Loading:** 50MB embeddings fit in RAM → faster than disk I/O

6. **Streamlit Caching:** @st.cache_resource for models (expensive), @st.cache_data for API (ephemeral)

7. **128-dim Output:** Standard for contrastive learning (CLIP uses 512, but 128 is sufficient for 7k movies)

---

## Risk Mitigation

**Risk:** Training doesn't converge
- **Mitigation:** Start with higher LR (1e-3), add gradient clipping, reduce batch size

**Risk:** GPU memory overflow
- **Mitigation:** Reduce batch size to 128, use CPU training (slower but works)

**Risk:** Contrastive performs worse than baseline
- **Mitigation:** This is informative! Document why, analyze failure modes, keep baseline as default

**Risk:** No hard negatives found for some pairs
- **Mitigation:** Fallback to random negatives, track success rate in stats

**Risk:** Claude API costs too high
- **Mitigation:** Cache all responses, make API key optional, use Haiku (cheapest model)

---

## Timeline Estimate

- **Task 1:** 4-6 hours (implementation + testing)
- **Task 2:** 6-8 hours (implementation + training + debugging)
- **Task 3:** 3-4 hours (implementation + testing)
- **Task 4:** 4-5 hours (UI + UX + testing)
- **Task 5:** 3-4 hours (evaluation + analysis)

**Total:** 20-27 hours focused development

---

## Deliverables Summary

### Code (5 new files)
- src/data/contrastive_dataset.py
- src/models/projection_head.py
- src/models/contrastive_recommender.py
- app/main.py (rewrite)
- src/evaluation/final_evaluation.py

### Data/Models (4 artifacts)
- data/processed/contrastive_pairs.json
- models/projection_head/best_model.pt
- models/projection_head/training_config.json
- data/embeddings/movie_faiss_projected.index

### Evaluation (3 artifacts)
- plots/contrastive_training_curves.png
- plots/final_evaluation_comparison.png
- data/processed/final_evaluation_results.json

### User-Facing
1. Dataset stats (pairs, ratio, coverage)
2. Training results (val loss, epoch, similarity gap improvement)
3. Final evaluation table (P@10, nDCG@10 for both models)
4. 3 good + 3 bad recommendations with analysis
5. Localhost URL for Streamlit app
6. Design decisions documented
7. Sprint 4 recommendations (polish, deployment, portfolio)
