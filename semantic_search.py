import os
import pickle
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

EMBEDDING_CACHE_FILE = "embeddings_cache.pkl"
MODEL_NAME = "all-MiniLM-L6-v2" # Small, fast, effective

class SemanticSearch:
    def __init__(self):
        # Lazy load the model only when needed
        self.model = None
        self.library_data = []
        self.embeddings = None
        
    def _load_model(self):
        if not self.model:
            with st.spinner("Loading embedding model (first time only)..."):
                self.model = SentenceTransformer(MODEL_NAME)

    def index_library(self, library_data: List[Dict[str, Any]], force_refresh: bool = False):
        """
        Creates embeddings for the library. 
        Checks if cache exists and matches the current library size/content.
        """
        self.library_data = library_data
        
        # Check cache
        if not force_refresh and os.path.exists(EMBEDDING_CACHE_FILE):
            try:
                with open(EMBEDDING_CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)
                    # Simple validation: check if length matches
                    if len(cache["embeddings"]) == len(library_data):
                        self.embeddings = cache["embeddings"]
                        # print("Loaded embeddings from cache.")
                        return
            except Exception as e:
                print(f"Error loading embedding cache: {e}")

        # If no cache or invalid, generate new embeddings
        self._load_model()
        
        # Prepare text to embed: "Title - Artist - Album" gives best context
        texts = [f"{t['title']} - {t['artist']} - {t['album']}" for t in library_data]
        
        with st.spinner(f"Generating embeddings for {len(texts)} tracks..."):
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
        # Save cache
        with open(EMBEDDING_CACHE_FILE, "wb") as f:
            pickle.dump({"embeddings": self.embeddings}, f)

    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Searches the library for the query and returns the top_k matching tracks.
        """
        if self.embeddings is None or len(self.library_data) == 0:
            return []
            
        self._load_model()
        
        # Embed query
        query_embedding = self.model.encode([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(self.library_data[idx])
            
        return results
