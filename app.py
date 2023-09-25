from operator import index
import streamlit as st
from pycaret import classification
from pycaret import regression
from pycaret import clustering
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar:
    st.image("./img.png")
    st.title("AutoML")
    choice = st.radio(
        "Navigation", ["Upload", "Exploratory data analysis", "Model Training", "Download Model"])
    st.info("This applications runs through all possible machine learning models for any given Machine Learning Technique and finds the most optimal model for your dataset.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)


if choice == "Exploratory data analysis":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
