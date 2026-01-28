import pandas as pd
import numpy as np
import re
import os
import streamlit as st
import streamlit.components.v1 as components

st.markdown('Please find the GitHub Repository for this project [here](https://github.com/ishijo/cross-content-recommendation-system).')
#st.image('./data/Cover-Img.png')
st.title('Cross Content Recommendation System')
st.write(' ')

#pickle_in = open("lol.pkl","rb")
def main():
    
    st.selectbox('Recommend me more of - ',('a','b','c'),placeholder="",)
    st.write(' ')

    st.write('In ...')
    col1, col2, col3 = st.columns(3,gap = "large")
    col1.button('Movies',use_container_width=True)
    col2.button('Books',use_container_width=True)
    col3.button('Both',use_container_width=True)
    st.write(' ')

    st.button('Recommend',use_container_width=True)

    

if __name__=='__main__':
    main()
