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
