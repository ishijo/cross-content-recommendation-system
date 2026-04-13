#!/bin/bash
# Sprint 2 Execution Script
# Runs all 5 modules in sequence

set -e  # Exit on error

echo "==============================================================================="
echo "                      SPRINT 2 - FULL EXECUTION"
echo "==============================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/project_config.py" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Task 1: Test utilities
echo "📋 TASK 1: Testing Helper Utilities"
echo "==============================================================================="
python3 src/utils/helpers.py
if [ $? -ne 0 ]; then
    echo "❌ Helper utilities test failed"
    exit 1
fi
echo ""
echo "✅ Task 1 complete"
echo ""
sleep 2

# Task 2: Generate embeddings
echo "📋 TASK 2: Generating SBERT Embeddings"
echo "==============================================================================="
python3 src/models/embedding_generator.py
if [ $? -ne 0 ]; then
    echo "❌ Embedding generation failed"
    exit 1
fi
echo ""
echo "✅ Task 2 complete"
echo ""
sleep 2

# Task 3: Build FAISS index
echo "📋 TASK 3: Building FAISS Index"
echo "==============================================================================="
python3 src/models/faiss_indexer.py
if [ $? -ne 0 ]; then
    echo "❌ FAISS indexing failed"
    exit 1
fi
echo ""
echo "✅ Task 3 complete"
echo ""
sleep 2

# Task 4: Test baseline recommender
echo "📋 TASK 4: Testing Baseline Recommender"
echo "==============================================================================="
python3 src/models/baseline_recommender.py
if [ $? -ne 0 ]; then
    echo "❌ Baseline recommender test failed"
    exit 1
fi
echo ""
echo "✅ Task 4 complete"
echo ""
sleep 2

# Task 5: Run evaluation
echo "📋 TASK 5: Running Full Evaluation"
echo "==============================================================================="
python3 src/evaluation/metrics.py
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed"
    exit 1
fi
echo ""
echo "✅ Task 5 complete"
echo ""

# Final summary
echo ""
echo "==============================================================================="
echo "                    ✅ SPRINT 2 COMPLETE!"
echo "==============================================================================="
echo ""
echo "📊 Results saved to:"
echo "   - data/embeddings/embedding_metadata.json"
echo "   - data/processed/baseline_metrics.json"
echo "   - plots/baseline_evaluation.png"
echo "   - logs/ (detailed execution logs)"
echo ""
echo "📋 Next: Review results and plan Sprint 3 (contrastive learning)"
echo "==============================================================================="
