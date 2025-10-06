import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import concurrent.futures

# --- Streamlit UI Setup ---

st.set_page_config(layout="wide", page_title="Merge Suggestions Tool")

st.title("Candour URL Merge Suggestion Tool")
st.markdown("This tool helps you find the best pages to merge/redirect a list of source URLs into. It works by analysing page content similarity.")

# --- API Key Input ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API Key", type="password")
    st.markdown("You can get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")

# Use the embeddings model
EMBEDDING_MODEL = "text-embedding-004"


# --- Core Functions ---

@st.cache_data(show_spinner=False)
def fetch_page_content(url):
    """Fetches and extracts the main text content from a single URL."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        # Extract text from common content tags, join with space, and clean up
        text_parts = [tag.get_text(separator=' ', strip=True) for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'main'])]
        content = ' '.join(text_parts)
        
        return url, content[:15000] # Return URL and truncated content to manage token limits
    except requests.RequestException as e:
        return url, f"Error fetching URL: {e}"
    except Exception as e:
        return url, f"An unexpected error occurred: {e}"


def batch_fetch_content(urls):
    """Fetches content for a list of URLs concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_page_content, urls))
    return dict(results)


@st.cache_data(show_spinner=False)
def get_embeddings(texts, model_name):
    """Generates embeddings for a list of texts using the Gemini API."""
    if not texts:
        return []
    try:
        # The API can handle a batch of texts directly
        result = genai.embed_content(
            model=model_name,
            content=texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"An error occurred while generating embeddings: {e}")
        return None


def find_best_matches(source_urls, dest_urls, source_embeddings, dest_embeddings):
    """Finds the best destination URL for each source URL based on cosine similarity."""
    similarity_matrix = cosine_similarity(source_embeddings, dest_embeddings)
    
    matches = []
    for i, source_url in enumerate(source_urls):
        # Find the index of the destination URL with the highest similarity score
        best_match_index = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_match_index]
        best_match_url = dest_urls[best_match_index]
        
        matches.append({
            "Source URL": source_url,
            "Suggested Destination": best_match_url,
            "Similarity Score": f"{best_score:.2%}"
        })
    return matches


# --- Streamlit Page Layout ---

col1, col2 = st.columns(2)

with col1:
    st.header("1. Source URLs")
    st.markdown("👇 Paste the URLs you want to **redirect from** (one per line).")
    source_urls_text = st.text_area("Source URLs", height=250, label_visibility="collapsed", placeholder="https://example.com/old-page-1\nhttps://example.com/old-page-2")

with col2:
    st.header("2. Destination URLs")
    st.markdown("👇 Paste the URLs you want to **redirect to** (one per line). I think these should all be unique from the first list. Idk what will happen if some URLs here are present in the 1st list. Probably something bad.")
    dest_urls_text = st.text_area("Destination URLs", height=250, label_visibility="collapsed", placeholder="https://example.com/new-topic-a\nhttps://example.com/new-topic-b")

if st.button("Find Merge Suggestions", type="primary"):
    # --- Input Validation and API Configuration ---
    if not api_key:
        st.error("Please enter your Google API Key in the sidebar to continue.")
        st.stop()
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to configure API. Please check your key. Error: {e}")
        st.stop()
        
    source_urls = [url.strip() for url in source_urls_text.splitlines() if url.strip()]
    dest_urls = [url.strip() for url in dest_urls_text.splitlines() if url.strip()]

    if not source_urls or not dest_urls:
        st.warning("Please provide at least one source and one destination URL.")
    else:
        # --- Processing Logic ---
        with st.spinner("Step 1/3: Crawling content from all URLs... 🕸️"):
            all_urls = list(set(source_urls + dest_urls))
            content_map = batch_fetch_content(all_urls)
            
            # Separate content back into source and destination lists
            source_content = [content_map[url] for url in source_urls]
            dest_content = [content_map[url] for url in dest_urls]

        with st.spinner("Step 2/3: Generating embeddings with Gemini API... ✨"):
            source_embeddings = get_embeddings(source_content, EMBEDDING_MODEL)
            dest_embeddings = get_embeddings(dest_content, EMBEDDING_MODEL)
        
        if source_embeddings is not None and dest_embeddings is not None:
            with st.spinner("Step 3/3: Calculating similarity and finding matches... 📊"):
                results = find_best_matches(source_urls, dest_urls, source_embeddings, dest_embeddings)
            
            st.success("Suggestions Generated")
            
            st.header("3. Suggested Merges")
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, height=len(df) * 36 + 38)
            
            # Provide a download button for the results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
               label="📥 Download results as CSV",
               data=csv,
               file_name="merge_suggestions.csv",
               mime="text/csv",
            )
        else:
            st.error("Failed to generate embeddings. Please check the URLs or your API key permissions.")
