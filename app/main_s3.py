"""
Streamlit app using S3 for data storage.
Copy this to main.py if you want to use S3 instead of local storage.
"""

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import S3 data loader
from utils import s3_data_loader

st.markdown('Please find the GitHub Repository for this project [here](https://github.com/ishijo/cross-content-recommendation-system).')
st.title('Cross Content Recommendation System')
st.write(' ')

# Load data from S3 (cached automatically)
with st.spinner('Loading data from S3...'):
    imdb_data = s3_data_loader.load_imdb_data()
    goodreads_data = s3_data_loader.load_goodreads_data()

if imdb_data is None or goodreads_data is None:
    st.error("Failed to load data from S3. Please check your configuration.")
    st.stop()

st.success("✓ Data loaded from S3")

def main():
    st.selectbox('Recommend me more of - ',('a','b','c'),placeholder="")
    st.write(' ')

    st.write('In ...')
    col1, col2, col3 = st.columns(3, gap="large")
    col1.button('Movies', use_container_width=True)
    col2.button('Books', use_container_width=True)
    col3.button('Both', use_container_width=True)
    st.write(' ')

    st.button('Recommend', use_container_width=True)

    # Example: Display data info
    if st.checkbox('Show data info'):
        st.write(f"IMDB basics shape: {imdb_data['basics'].shape}")
        st.write(f"IMDB ratings shape: {imdb_data['ratings'].shape}")
        st.write(f"Goodreads shape: {goodreads_data.shape}")

if __name__=='__main__':
    main()
