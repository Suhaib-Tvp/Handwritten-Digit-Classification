
import streamlit as st

st.set_page_config(page_title="Spam Detection App", layout="centered")

st.title("Email Spam Detection App")

st.write("Select a model from the sidebar to classify emails.")

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the model", ["Logistic Regression", "Naive Bayes"])

if app_mode == "Logistic Regression":
    import logistic_app
elif app_mode == "Naive Bayes":
    import naive_bayes_app
