"""
Load datasets directly from AWS S3.
No local storage required - data is read directly into memory.
"""

import pandas as pd
import boto3
from io import StringIO, BytesIO
import streamlit as st

# S3 Configuration
BUCKET_NAME = "your-bucket-name"  # Change this
AWS_REGION = "us-east-1"  # Change this

# Option 1: Public bucket (no auth needed)
def load_from_public_s3(s3_key):
    """Load CSV from public S3 bucket using HTTPS."""
    url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    df = pd.read_csv(url)
    return df

# Option 2: Private bucket (requires AWS credentials)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_from_private_s3(s3_key):
    """Load CSV from private S3 bucket using boto3."""
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    # Get object from S3
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)

    # Read CSV into pandas
    if s3_key.endswith('.tsv'):
        df = pd.read_csv(BytesIO(response['Body'].read()), sep='\t')
    else:
        df = pd.read_csv(BytesIO(response['Body'].read()))

    return df

# Option 3: Load with s3fs (recommended for large files)
@st.cache_data(ttl=3600)
def load_with_s3fs(s3_path):
    """
    Load CSV using s3fs - efficient for large files.
    s3_path example: 's3://bucket-name/path/to/file.csv'

    Requires: pip install s3fs
    """
    df = pd.read_csv(s3_path)
    return df

# Main loading function
@st.cache_data(ttl=3600)
def load_imdb_data():
    """Load IMDB datasets from S3."""
    try:
        # Load different IMDB files
        title_basics = load_from_private_s3("datasets/imdb/title.basics.tsv/data.tsv")
        title_ratings = load_from_private_s3("datasets/imdb/title.ratings.tsv/data.tsv")

        return {
            'basics': title_basics,
            'ratings': title_ratings
        }
    except Exception as e:
        st.error(f"Error loading IMDB data from S3: {e}")
        return None

@st.cache_data(ttl=3600)
def load_goodreads_data():
    """Load Goodreads dataset from S3."""
    try:
        df = load_from_private_s3("datasets/goodreads/goodreads_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading Goodreads data from S3: {e}")
        return None

# Example usage
if __name__ == "__main__":
    print("Loading data from S3...")

    # Test loading
    imdb_data = load_imdb_data()
    goodreads_data = load_goodreads_data()

    if imdb_data:
        print(f"✓ IMDB data loaded: {imdb_data['basics'].shape}")
    if goodreads_data is not None:
        print(f"✓ Goodreads data loaded: {goodreads_data.shape}")
