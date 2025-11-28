import json
import os
import time
from typing import List, Dict, Any
from thefuzz import process

CACHE_FILE = "library_cache.json"
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # 24 hours

def save_cache(data: List[Dict[str, Any]]):
    """Saves the library data to a local JSON cache file."""
    cache_data = {
        "timestamp": time.time(),
        "data": data
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f)

def load_cache() -> List[Dict[str, Any]]:
    """Loads library data from cache if it exists and is not expired."""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)
        
        if time.time() - cache_data.get("timestamp", 0) > CACHE_EXPIRY_SECONDS:
            return None
            
        return cache_data.get("data")
    except (json.JSONDecodeError, KeyError):
        return None

def clear_cache():
    """Removes the cache file."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

def match_tracks(suggested_titles: List[str], library_data: List[Dict[str, Any]], threshold: int = 85) -> List[Dict[str, Any]]:
    """
    Matches a list of suggested titles (strings) against the library data using fuzzy matching.
    Returns a list of matched track objects from the library.
    """
    matched_tracks = []
    
    # Create a dictionary for faster lookup if needed, but for fuzzy matching we need the list of choices.
    # We'll construct a list of "Title - Artist" strings from the library to match against.
    library_strings = [f"{track['title']} - {track['artist']}" for track in library_data]
    
    for suggestion in suggested_titles:
        # Find the best match
        result = process.extractOne(suggestion, library_strings)
        
        if result:
            match_string, score = result
            if score >= threshold:
                # Find the original track object. 
                # Note: This simple lookup assumes uniqueness of "Title - Artist". 
                # If there are duplicates, it picks the first one found.
                index = library_strings.index(match_string)
                track = library_data[index]
                matched_tracks.append({
                    "track": track,
                    "score": score,
                    "suggestion": suggestion
                })
    
    
    return matched_tracks

def normalize_key(title: str, artist: str) -> str:
    """
    Creates a normalized key for caching/matching.
    Rules:
    - Lowercase
    - Remove text in () or []
    - Remove punctuation
    - Remove leading 'the ' from artist
    """
    import re
    
    def clean(s):
        if not s: return ""
        s = s.lower()
        s = re.sub(r'\([^)]*\)', '', s) # Remove (text)
        s = re.sub(r'\[[^]]*\]', '', s) # Remove [text]
        s = re.sub(r'[^\w\s]', '', s)   # Remove punctuation
        return s.strip()
        
    c_title = clean(title)
    c_artist = clean(artist)
    
    if c_artist.startswith("the "):
        c_artist = c_artist[4:]
        
    return f"{c_title} - {c_artist}"

def fill_missing_features(library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Estimates missing Valence and Energy features using Danceability and BPM.
    This is a fallback heuristic when Last.fm data is missing.
    """
    enriched_library = []
    for track in library:
        new_track = track.copy()
        
        # Check if we need to fill missing features
        needs_valence = 'valence' not in new_track or new_track['valence'] is None
        needs_energy = 'energy' not in new_track or new_track['energy'] is None
        
        if needs_valence or needs_energy:
            # We need at least danceability or bpm
            danceability = new_track.get('danceability')
            bpm = new_track.get('bpm')
            
            if danceability is not None:
                # Heuristics
                
                # Energy: Heavily correlated with BPM and Danceability
                if needs_energy:
                    # Normalize BPM roughly to 0-1 (assuming 60-180 range)
                    norm_bpm = 0.5
                    if bpm:
                        norm_bpm = min(max((bpm - 60) / 120, 0.0), 1.0)
                    
                    # Energy formula
                    new_track['energy'] = (float(danceability) * 0.6) + (norm_bpm * 0.4)
                    new_track['features_source'] = 'heuristic_estimated'
                
                # Valence: Hard to estimate, but often correlated with Danceability
                if needs_valence:
                    # Very rough proxy
                    new_track['valence'] = float(danceability)
                    new_track['features_source'] = 'heuristic_estimated'
                    
        enriched_library.append(new_track)
        
    return enriched_library
