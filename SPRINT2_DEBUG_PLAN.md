# SPRINT 2 - DEBUG & FIX PLAN

**Status:** 🚨 CRITICAL ISSUES DETECTED
**Issues:**
1. Evaluation failed (0 books evaluated)
2. Low similarity scores (0.33 instead of 0.45)

---

## 🎯 IMMEDIATE ACTION PLAN

Run these 3 scripts in order:

### **STEP 1: Verify Outputs**
```bash
python verify_sprint2.py
```

**What this checks:**
- ✓ Embeddings exist and are normalized
- ✓ Genre mapping exists and isn't empty
- ✓ FAISS index loads correctly
- ✓ Similarity scores are reasonable

**Expected output:**
- All checks should pass (✓)
- Embedding norms should be ~1.0
- Within-domain similarity should be >0.40

**If fails:** Fix the specific issue it identifies before proceeding.

---

### **STEP 2: Diagnose Root Cause**
```bash
python debug_evaluation.py
```

**What this checks:**
- Genre mapping validity (do mapped genres actually exist in IMDB?)
- Embedding quality (norms, NaNs, actual similarities)
- Single book evaluation walkthrough
- Why evaluation loop is failing

**Expected output:**
- Should show exactly where the pipeline breaks
- Will identify if it's genre mapping, embeddings, or matching logic

**Key things to look for:**
```
❌ "No mapped movie genres!" → Genre mapping problem
❌ "ZERO relevant movies found!" → Genre matching problem
❌ "Invalid genre mappings!" → Genre mapping doesn't match IMDB
⚠️  "Embedding norms != 1.0" → Normalization problem
```

---

### **STEP 3: Run Fixed Evaluation**
```bash
python quick_fix_evaluation.py
```

**What this does:**
- Runs evaluation with better error handling
- Logs exactly why each book fails
- Shows debug counters (how many failed for what reason)
- Produces actual metrics if ANY books succeed

**Expected output:**
```
✓ Successful: 150-180
✗ Failed (no mapped genres): 10-20
✗ Failed (no relevant movies): 10-20
✗ Failed (exceptions): 0-5

Precision@10: 0.30-0.40 (if successful)
```

---

## 🔍 LIKELY ROOT CAUSES

Based on your symptoms, here's what's probably wrong:

### **Issue 1: Evaluation Failing (0 books evaluated)**

**Symptom:** All metrics are 0.0000, 0 books evaluated

**Most likely causes (in order of probability):**

1. **Genre mapping is broken** (80% likely)
   - Mapped genres don't exist in IMDB data
   - Example: Mapping says "Fantasy" but IMDB has "Sci-Fi" (case mismatch)
   - Fix: Check `debug_evaluation.py` output for "invalid genre mappings"

2. **Book genres aren't being parsed** (15% likely)
   - `ast.literal_eval()` is failing silently
   - Genres column has unexpected format
   - Fix: Check `debug_evaluation.py` output for parse errors

3. **Relevance threshold too strict** (5% likely)
   - No movies share genres with books
   - Fix: Lower the threshold or use fuzzy matching

---

### **Issue 2: Low Similarity (0.33 instead of 0.45)**

**Symptom:** Within-domain similarity is 0.3368 instead of expected 0.45-0.50

**Most likely causes:**

1. **Embeddings aren't normalized** (60% likely)
   - Check: Run `verify_sprint2.py` - it will show norms
   - If norms != 1.0, embeddings weren't normalized
   - Fix: Regenerate embeddings with normalization enabled

2. **Input format is wrong** (30% likely)
   - Genres aren't being included in input string
   - Only description is being encoded
   - Check: Look at embedding_generator.py logs for input examples

3. **Model loaded incorrectly** (10% likely)
   - Wrong model downloaded
   - Model corrupted
   - Check: `verify_sprint2.py` will show model name in metadata

---

## 🛠️ HOW TO FIX EACH ISSUE

### **Fix 1: Genre Mapping is Broken**

**If `debug_evaluation.py` shows "invalid genre mappings":**

The mapped genres don't match actual IMDB genres. This happens because:
- Genre mapping from Sprint 1 uses different capitalization
- IMDB genres changed
- Mapping has typos

**Solution:**
```python
# Create a genre normalization script
# Maps "Fantasy" → "fantasy", handles case-insensitive matching
```

I can create a `fix_genre_mapping.py` script if this is the issue.

---

### **Fix 2: Embeddings Not Normalized**

**If `verify_sprint2.py` shows norms != 1.0:**

Embeddings weren't L2-normalized during generation.

**Solution:**
```bash
# Delete existing embeddings
rm data/embeddings/*.npy

# Regenerate with normalization
# (Already enabled in embedding_generator.py, but double-check)
python src/models/embedding_generator.py
```

Check that `embedding_generator.py` line 82 has:
```python
normalize_embeddings=True  # Must be True!
```

---

### **Fix 3: Genre Parsing Fails**

**If `debug_evaluation.py` shows "Failed to parse genres":**

The Genres column format is unexpected.

**Solution:**
```python
# Check actual format
import pandas as pd
df = pd.read_csv('data/raw/goodreads/goodreads_data.csv', index_col=0)
print(df['Genres'].head())

# If it's NOT ['Genre1', 'Genre2'] format, fix parsing logic
```

I can create a `fix_genre_parsing.py` script if needed.

---

## 📊 WHAT TO REPORT BACK

After running all 3 scripts, please share:

### **From `verify_sprint2.py`:**
```
✓/✗ Embeddings check
✓/✗ Genre Mapping check
✓/✗ FAISS Index check
✓/✗ Metadata check

Embedding norms: X.XXXX (should be ~1.0)
Within-book similarity: X.XXXX (should be >0.40)
```

### **From `debug_evaluation.py`:**
```
Total genre mappings: XXX
Empty mappings: XX%
Unique IMDB genres: [list]
Invalid genre mappings: XXX
Relevant movies found for Harry Potter: XXX
```

### **From `quick_fix_evaluation.py`:**
```
✓ Successful: XXX
✗ Failed (no recommendations): XXX
✗ Failed (no mapped genres): XXX
✗ Failed (no relevant movies): XXX

Precision@10: X.XXXX
NDCG@10: X.XXXX
```

---

## 🎯 SUCCESS CRITERIA

After fixes, you should see:

✅ **`verify_sprint2.py`:**
- All checks pass
- Norms = 0.98-1.02
- Within-domain similarity > 0.40

✅ **`debug_evaluation.py`:**
- <10% empty genre mappings
- 0 invalid genre mappings
- >1000 relevant movies for typical books

✅ **`quick_fix_evaluation.py`:**
- >150 successful evaluations
- Precision@10 > 0.25
- NDCG@10 > 0.35

---

## 📝 NEXT STEPS

1. Run all 3 scripts
2. Report back the key numbers above
3. I'll create targeted fix scripts based on what breaks
4. Re-run evaluation
5. Document final metrics for Sprint 3 planning

---

**TL;DR:**
```bash
# Run these 3 commands:
python verify_sprint2.py       # Check outputs are valid
python debug_evaluation.py     # Find root cause
python quick_fix_evaluation.py # Get actual metrics

# Then report back what failed
```
