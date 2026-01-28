# Cross-Content Recommendation System

A recommendation system that works across multiple content types (movies and books) using content-based filtering.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

The datasets are hosted on Kaggle and are too large to store in this repository. Download them using one of the methods below:

#### Option A: Manual Download

1. **IMDB Data**: Download from [Kaggle - IMDB Data with Descriptions](https://www.kaggle.com/datasets/ishikajohari/imdb-data-with-descriptions)
2. **Books Data**: Download from [Kaggle - Best Books 10K Multi-Genre Data](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data)

Place the downloaded files in the `data/` directory:
```
data/
├── imdb/
│   ├── title.basics.tsv/
│   ├── title.ratings.tsv/
│   ├── title.akas.tsv/
│   ├── title.principals.tsv/
│   └── name.basics.tsv/
└── goodreads/
    └── goodreads_data.csv
```

#### Option B: Using Kaggle API

Install the Kaggle API and download automatically:

```bash
# Install Kaggle CLI
pip install kaggle

# Download datasets
kaggle datasets download -d ishikajohari/imdb-data-with-descriptions -p data/imdb --unzip
kaggle datasets download -d ishikajohari/best-books-10k-multi-genre-data -p data/goodreads --unzip
```

**Note**: You'll need to set up Kaggle API credentials. See [Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials) for details.

## Usage

```bash
streamlit run main.py
```

## Project Structure

- `src/` - Source code for data processing and model
- `data/` - Data directory (gitignored, download separately)
- `main.py` - Main application entry point
- `reports/` - Analysis and reports

## Data

This project uses:
- **IMDB Dataset**: Movie titles, ratings, and descriptions
- **Goodreads Dataset**: Book titles, ratings, and multi-genre information

Both datasets are publicly available on Kaggle (links above).
