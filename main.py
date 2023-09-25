import streamlit as st
from pycaret import classification
from pycaret import regression
from pycaret import clustering
import pandas as pd
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


if choice == "Model Training":
    model = st.selectbox(
        "Select Model", ["Classification", "Regression", "Clustering"])
    if model == "Classification":
        st.title("Classification")
        target = st.selectbox("Select Target Variable", df.columns)
        if st.button("Run"):
            st.write("Running...")
            classification.setup(
                data=df, target=target)
            setup_df = classification.pull()
            st.dataframe(setup_df)
            best_model = classification.compare_models()
            compare_df = classification.pull()
            st.dataframe(compare_df)
            classification.save_model(best_model, 'best_model')

    if model == "Regression":
        st.title("Regression")
        target = st.selectbox("Select Target Variable", df.columns)
        if st.button("Run"):
            st.write("Running...")
            regression.setup(
                data=df, target=target)
            setup_df = regression.pull()
            st.dataframe(setup_df)
            best_model = regression.compare_models()
            compare_df = regression.pull()
            st.dataframe(compare_df)
            regression.save_model(best_model, 'best_model')

    if model == "Clustering":
        st.title("Clustering")
        model_choice = st.selectbox("Select Model", ["K-Means Clustering", "Affinity Propagation", "Mean Shift Clustering", "Spectral Clustering",
                                                     "Agglomerative Clustering", "DBSCAN", "OPTICS", "Birch", "K-Modes Clustering",])
        if st.button("Run"):

            clustering.setup(
                data=df)
            setup_df = clustering.pull()
            st.dataframe(setup_df)
            if model_choice == "K-Means Clustering":
                best_model = clustering.create_model("kmeans")
                compare_df = clustering.pull()
                kmeans_model = clustering.assign_model(best_model)
                st.dataframe(kmeans_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "Affinity Propagation":
                best_model = clustering.create_model("ap")
                compare_df = clustering.pull()
                ap_model = clustering.assign_model(best_model)
                st.dataframe(ap_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "Mean Shift Clustering":
                best_model = clustering.create_model("meanshift")
                compare_df = clustering.pull()
                meanshift_model = clustering.assign_model(best_model)
                st.dataframe(meanshift_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "Spectral Clustering":
                best_model = clustering.create_model("sc")
                compare_df = clustering.pull()
                sc_model = clustering.assign_model(best_model)
                st.dataframe(sc_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "Agglomerative Clustering":
                best_model = clustering.create_model("hclust")
                compare_df = clustering.pull()
                hclust_model = clustering.assign_model(best_model)
                st.dataframe(hclust_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "DBSCAN":
                best_model = clustering.create_model("dbscan")
                compare_df = clustering.pull()
                dbscan_model = clustering.assign_model(best_model)
                st.dataframe(dbscan_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "OPTICS":
                best_model = clustering.create_model("optics")
                compare_df = clustering.pull()
                optics_model = clustering.assign_model(best_model)
                st.dataframe(optics_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "Birch":
                best_model = clustering.create_model("birch")
                compare_df = clustering.pull()
                birch_model = clustering.assign_model(best_model)
                st.dataframe(birch_model)
                clustering.save_model(best_model, 'best_model')
            if model_choice == "K-Modes Clustering":
                best_model = clustering.create_model("kmodes")
                compare_df = clustering.pull()
                kmodes_model = clustering.assign_model(best_model)
                st.dataframe(kmodes_model)
                clustering.save_model(best_model, 'best_model')


if choice == "Download Model":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
