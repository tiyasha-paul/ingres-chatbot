import streamlit as st
from main import rag_with_fallback, load_data, plot_extraction_trend, plot_category_bar
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Constants
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Initialize Streamlit app
st.title("INGRES Groundwater Chatbot")
st.markdown("Enter a query about rainfall or groundwater (e.g., 'rainfall for Raiganj in 2024' or 'groundwater for Raiganj in 2024').")

# Load FAISS index
@st.cache_resource
def load_vectorstore():
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists("faiss_index"):
        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
            st.write("Loaded FAISS index from 'faiss_index'.")
            return vectorstore
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
    df = load_data()
    vectorstore = FAISS.from_documents([], embeddings_model)  # Placeholder if no index
    st.write("No FAISS index found. Building new one.")
    return vectorstore

vectorstore = load_vectorstore()

# Query input
query = st.text_input("Enter your query:", "")

# Visualization inputs
st.subheader("Generate Visualizations")
state = st.selectbox("Select State:", ["Delhi", "Odisha", "Haryana", "West Bengal"])
location = st.text_input("Location (optional):")
year_str = st.text_input("Year Pattern (e.g., '2024'):")

if st.button("Generate Trend Plot"):
    if state:
        fig = plot_extraction_trend(state, location if location else None)
        st.pyplot(fig)
    else:
        st.warning("Please select a state.")

if st.button("Generate Category Bar"):
    if state and year_str:
        fig = plot_category_bar(state, year_str)
        st.pyplot(fig)
    else:
        st.warning("Please select a state and year pattern.")

# Submit query
language = st.selectbox("Select language:", ["en", "hi", "bn"])  # Add more languages as needed

if st.button("Submit Query"):
    if query:
        with st.spinner("Processing query..."):
            answer, sources = rag_with_fallback(query, vectorstore, lang=language)
            st.subheader("Answer:")
            st.write(answer)
            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(f"- {source['location']}, {source['state']}, {source['year']}")
    else:
        st.warning("Please enter a query.")