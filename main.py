import streamlit as st
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


model = SentenceTransformer('all-MiniLM-L6-v2')


def fetch_pdf_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            reader = PdfReader("temp.pdf")
            content = ""
            for page in reader.pages:
                content += page.extract_text()
            print(f"Fetched content from {url}: {content[:100]}")  
            return content
        else:
            st.error(f"Failed to fetch PDF: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching PDF content: {str(e)}")
        return None


def create_embeddings(text):
    try:
        embedding = model.encode(text)
        print(f"Created embedding for text: {text[:50]}") 
        return embedding
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def create_faiss_index(embeddings, documents, file_path="faiss_store_local.pkl"):
    dimension = len(embeddings[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype(np.float32))
    
   
    with open(file_path, "wb") as f:
        pickle.dump((faiss_index, documents), f)


    print(f"FAISS index saved as {file_path}")


def load_faiss_index(file_path="faiss_store_local.pkl"):
    with open(file_path, "rb") as f:
        faiss_index, documents = pickle.load(f)
    return faiss_index, documents


def query_faiss_index(faiss_index, query_embedding, documents, k=3):
    D, I = faiss_index.search(np.array([query_embedding]).astype(np.float32), k)
    results = [(documents[i], D[0][j]) for j, i in enumerate(I[0])]
    return results


st.title("Automated Scheme Research Tool")
st.sidebar.header("Input URL(s)")


url_input = st.sidebar.text_area("Enter the PDF URLs (one per line):")


if st.sidebar.button("Process URLs"):
    urls = url_input.split("\n")
    st.write("Processing URLs...")

   
    documents = []
    embeddings = []
    
    for url in urls:
        content = fetch_pdf_content(url)
        if content:
            documents.append(content)
            embedding = create_embeddings(content)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                st.error("Failed to create embeddings.")
        else:
            st.error("Failed to fetch valid content from the URL.")


    if embeddings:
        create_faiss_index(embeddings, documents)
        st.success("Documents processed and FAISS index created!")


query = st.text_input("Ask a question about the scheme(s):")
if query and st.button("Submit Query"):
    query_embedding = create_embeddings(query)
    

    try:
        faiss_index, documents = load_faiss_index()
        results = query_faiss_index(faiss_index, query_embedding, documents)

        st.write("Top Results:")
        for i, (doc, score) in enumerate(results):
            st.write(f"Result {i+1} (Score: {score}):")
            st.write(doc[:500]) 
            st.write("-" * 50)

    except FileNotFoundError:
        st.error("FAISS index not found. Please process URLs first.")
