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
        Checks if cache exists and matches the current library keys.
        """
        self.library_data = library_data
        current_keys = [t['key'] for t in library_data]
        
        # Check cache
        if not force_refresh and os.path.exists(EMBEDDING_CACHE_FILE):
            try:
                with open(EMBEDDING_CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)
                    
                    # Validate: check if keys match exactly
                    if "library_keys" in cache and cache["library_keys"] == current_keys:
                        self.embeddings = cache["embeddings"]
                        print(f"✓ Loaded {len(self.embeddings)} embeddings from cache (keys validated).")
                        return
                    elif "library_keys" in cache:
                        print(f"⚠ Cache key mismatch: regenerating embeddings...")
                        print(f"  Cache has {len(cache['library_keys'])} keys, library has {len(current_keys)} keys")
                    else:
                        print(f"⚠ Old cache format (no keys stored): regenerating embeddings...")
            except Exception as e:
                print(f"Error loading embedding cache: {e}")

        # If no cache or invalid, generate new embeddings
        print(f"Generating fresh embeddings for {len(library_data)} tracks...")
        self._load_model()
        
        # Prepare text to embed: "Title - Artist - Album - Tags" gives best context
        texts = []
        for t in library_data:
            text = f"{t['title']} - {t['artist']} - {t['album']}"
            if 'tags' in t and t['tags']:
                text += f" - {', '.join(t['tags'])}"
            texts.append(text)
        
        with st.spinner(f"Generating embeddings for {len(texts)} tracks..."):
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
        # Save cache with keys for validation
        with open(EMBEDDING_CACHE_FILE, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "library_keys": current_keys
            }, f)
        print(f"✓ Generated and cached {len(self.embeddings)} embeddings with key validation.")

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

    def generate_journey(self, start_track_key, end_track_key, num_steps: int = 10) -> List[Dict[str, Any]]:
        """
        Generates a playlist that transitions from start_track to end_track.
        Uses vector interpolation.
        
        Args:
            start_track_key: Plex ratingKey (string or int) for the starting track
            end_track_key: Plex ratingKey (string or int) for the ending track
            num_steps: Number of tracks in the journey
        """
        # Normalize keys to strings (Plex returns strings, but handle ints just in case)
        start_key_str = str(start_track_key)
        end_key_str = str(end_track_key)
        
        # Debug: Check state
        print(f"\n=== Journey Debug ===")
        print(f"Library data loaded: {len(self.library_data)} tracks")
        print(f"Embeddings loaded: {self.embeddings is not None}")
        if self.embeddings is not None:
            print(f"Embeddings shape: {self.embeddings.shape}")
        print(f"Start track key: {start_key_str} (original: {start_track_key}, type: {type(start_track_key).__name__})")
        print(f"End track key: {end_key_str} (original: {end_track_key}, type: {type(end_track_key).__name__})")
        
        if self.embeddings is None or len(self.library_data) == 0:
            print("ERROR: Embeddings or library data not loaded!")
            return []
            
        import numpy as np
        
        # Find indices of start and end tracks (comparing strings)
        start_idx = next((i for i, t in enumerate(self.library_data) if str(t['key']) == start_key_str), None)
        end_idx = next((i for i, t in enumerate(self.library_data) if str(t['key']) == end_key_str), None)
        
        if start_idx is None or end_idx is None:
            print(f"ERROR: Start or End track not found in library data.")
            print(f"Start track found: {start_idx is not None} (index: {start_idx})")
            print(f"End track found: {end_idx is not None} (index: {end_idx})")
            # Debug: Show a few track keys for comparison
            print(f"Sample track keys from library: {[t['key'] for t in self.library_data[:5]]}")
            print(f"Sample track key types: {[type(t['key']).__name__ for t in self.library_data[:5]]}")
            return []
            
        start_vec = self.embeddings[start_idx]
        end_vec = self.embeddings[end_idx]
        
        journey_tracks = [self.library_data[start_idx]]
        used_indices = {start_idx, end_idx}
        used_artists = {self.library_data[start_idx]['artist']}
        
        # Generate intermediate steps
        # We want (num_steps - 2) intermediates
        steps = num_steps - 1
        for i in range(1, steps):
            alpha = i / steps
            # Linear interpolation: V = (1-a)*Start + a*End
            target_vec = (1 - alpha) * start_vec + alpha * end_vec
            
            # Find nearest neighbor to this target vector
            # We reshape target_vec to (1, -1) because cosine_similarity expects 2D array
            similarities = cosine_similarity(target_vec.reshape(1, -1), self.embeddings)[0]
            
            # Get top matches, excluding ones we've already used
            # We look deeper (top 100) to find unique artists
            top_indices = similarities.argsort()[::-1]
            
            found_next = False
            
            # First pass: Try to find a unique artist with compatible audio features
            best_candidate = None
            best_score = -1
            
            # Look at top 50 candidates
            for idx in top_indices[:50]:
                candidate = self.library_data[idx]
                
                # Skip if already used or artist used
                if idx in used_indices or candidate['artist'] in used_artists:
                    continue
                    
                # Calculate Audio Feature Score
                audio_score = 0
                
                # BPM Continuity (if available)
                if 'bpm' in candidate and candidate['bpm'] and 'bpm' in journey_tracks[-1] and journey_tracks[-1]['bpm']:
                    prev_bpm = journey_tracks[-1]['bpm']
                    curr_bpm = candidate['bpm']
                    # Penalty for large BPM jumps
                    bpm_diff = abs(prev_bpm - curr_bpm)
                    if bpm_diff < 10:
                        audio_score += 0.5
                    elif bpm_diff < 20:
                        audio_score += 0.2
                    else:
                        audio_score -= 0.2
                
                # Key Compatibility (Camelot Wheel logic simplified)
                # This is complex to implement fully without a library, but we can check for same key/mode
                if 'musical_key' in candidate and 'musical_key' in journey_tracks[-1]:
                    if candidate['musical_key'] == journey_tracks[-1]['musical_key'] and candidate['mode'] == journey_tracks[-1]['mode']:
                         audio_score += 0.3
                
                # Semantic Score (implied by rank, but let's normalize rank)
                # Rank 0 is best.
                rank_score = 1.0 - (list(top_indices).index(idx) / 50.0)
                
                total_score = rank_score + audio_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                journey_tracks.append(best_candidate)
                used_indices.add(best_idx)
                used_artists.add(best_candidate['artist'])
                found_next = True
            
            # Fallback: Just pick the best unique artist if audio scoring failed
            if not found_next:
                 for idx in top_indices[:100]:
                    candidate = self.library_data[idx]
                    if idx not in used_indices and candidate['artist'] not in used_artists:
                        journey_tracks.append(candidate)
                        used_indices.add(idx)
                        used_artists.add(candidate['artist'])
                        found_next = True
                        break
            
            # Second pass: If no unique artist found, just pick the best unique track
            if not found_next:
                for idx in top_indices[:100]:
                    if idx not in used_indices:
                        journey_tracks.append(self.library_data[idx])
                        used_indices.add(idx)
                        # Don't add to used_artists here to allow further repeats if necessary
                        found_next = True
                        break
            
            if not found_next:
                # Fallback if we ran out of unique tracks (unlikely)
                pass
                
        journey_tracks.append(self.library_data[end_idx])
        return journey_tracks
