import streamlit as st

st.title("Mr.Analyst - Your Personal AI Analyst")

st.write("""
Welcome to Mr.Analyst! This application is designed to help you analyze data using the power of AI. You can upload your datasets, ask questions, and get insights in a user-friendly interface.
""")



st.write("""Kindly upload your file to analyze""")

st.file_uploader("Choose a file", type=["csv", "xls", "xlsx","pdf", "docx", "txt"])

st.button("Upload and Analyze")

st.title("Ask Questions about your data")   
st.text_input("Ask your questions here ")

st.button("Ask AI")

st.button("Generate Sql query")

