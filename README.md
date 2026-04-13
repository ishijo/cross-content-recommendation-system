# Cross-Content Recommendation System

A recommendation system that matches books to movies using sentence embeddings and contrastive learning. Given a book you loved, find compatible movies to watch!

## 🎯 Project Overview

- **Input:** 10,000 books from Goodreads (461 unique genres)
- **Target:** 10,000 movies from IMDB (~25 genres)
- **Approach:** Semantic similarity using SBERT embeddings + contrastive projection head trained on genre-overlap supervision
- **Goal:** Cross-domain recommendations with genre alignment

## 📁 Project Structure

```
cross-content-recommendation-system/
├── data/
│   ├── raw/              # Original datasets (Kaggle downloads)
│   ├── processed/        # Cleaned dataframes, genre mappings
│   └── embeddings/       # Generated embeddings and FAISS index
├── notebooks/
│   └── archive/          # Historical analysis notebooks
├── src/
│   ├── data/             # Data processing scripts
│   ├── models/           # ML model code
│   ├── evaluation/       # Metrics and evaluation
│   └── utils/            # Shared utilities
├── app/                  # Streamlit web application
├── plots/                # EDA and evaluation visualizations
├── models/               # Saved model weights
└── tests/                # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd cross-content-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for genre similarity)
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=your_key_here
```

### 3. Download Data

The datasets are automatically downloaded from Kaggle on first run, or you can download manually:

**Kaggle API Setup (one-time):**
1. Get API token from [kaggle.com/account](https://www.kaggle.com/account)
2. Place `kaggle.json` in `~/.kaggle/`
3. Run: `chmod 600 ~/.kaggle/kaggle.json`

**Manual Download:**
```bash
python download_data.py
```

**Data Sources:**
- [IMDB Data](https://www.kaggle.com/datasets/ishikajohari/imdb-data-with-descriptions)
- [Goodreads Data](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data)

### 4. Run Data Pipeline

**Step 1: Genre Mapping (uses Claude API)**
```bash
python src/data/genre_mapper.py
```
This maps 461 book genres to ~25 movie genres using Claude API with retry logic and progress tracking.

**Step 2: Data Validation**
```bash
python src/data/data_validation.py
```
Validates data quality, generates EDA plots, and provides GO/NO-GO verdict.

### 5. Run Application

```bash
# Local data mode
streamlit run app/main.py

# S3 cloud storage mode
streamlit run app/main_s3.py
```

## 📊 Data Pipeline

1. **Raw Data** (`data/raw/`)
   - Goodreads: 10k books with descriptions, 461 genres
   - IMDB: 10k movies with descriptions, ~25 genres

2. **Genre Mapping** (`src/data/genre_mapper.py`)
   - Uses Claude API to map book genres → movie genres
   - Outputs: `genre_mapping.json`, `genre_mapping_summary.csv`

3. **Validation** (`src/data/data_validation.py`)
   - Checks for nulls in critical columns
   - Verifies ≥80% genre mapping coverage
   - Generates EDA plots in `plots/`

4. **Preprocessing** (notebooks)
   - Clean text, handle missing values
   - Create pickle files for quick loading

## 🔬 Machine Learning Pipeline

### Current Status: Sprint 1 Complete ✅

**Sprint 1 - Data Foundation**
- ✅ Repository restructuring
- ✅ Genre mapping with Claude API
- ✅ Data validation and EDA
- ✅ Fixed broken notebook paths

**Sprint 2 - Embedding Generation** (Next)
- Generate SBERT embeddings for all books/movies
- Create FAISS index for fast similarity search
- Implement baseline content-based recommendations

**Sprint 3 - Contrastive Learning** (Future)
- Train projection head on genre-overlap pairs
- Fine-tune for cross-domain alignment
- Evaluate with precision@K, NDCG metrics

**Sprint 4 - Production Deployment** (Future)
- Optimize inference pipeline
- Build interactive Streamlit UI
- Deploy to cloud (Streamlit Cloud / AWS)

## 🧪 Running Notebooks

All notebooks have been moved to `notebooks/archive/` with fixed paths:

```bash
cd notebooks
jupyter notebook archive/preprocessing.ipynb
```

**Available notebooks:**
- `preprocessing.ipynb` - Data cleaning
- `descriptions.ipynb` - Description similarity experiments
- `genres.ipynb` - Genre analysis and visualization
- `genreSimilarity.ipynb` - Genre similarity using spaCy
- `getDescription.ipynb` - Web scraping (historical)

## 🔧 Configuration

All paths are centralized in `src/config.py`. No hardcoded paths in scripts!

## 📈 Evaluation

Metrics tracked:
- Precision@K, Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Genre alignment accuracy
- Cross-domain coverage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is for educational and research purposes.

## 🔗 Links

- [GitHub Repository](https://github.com/ishijo/cross-content-recommendation-system)
- [IMDB Dataset](https://www.kaggle.com/datasets/ishikajohari/imdb-data-with-descriptions)
- [Goodreads Dataset](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data)
- [Sentence Transformers](https://www.sbert.net/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with:** Python, Sentence Transformers, Streamlit, Claude API, FAISS
