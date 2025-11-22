import os
import requests
import time
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = "lastfm_cache.json"

class LastFMClient:
    def __init__(self):
        self.api_key = os.getenv("LASTFM_API_KEY")
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        self.cache = self._load_cache()
        
        if not self.api_key:
            print("Warning: LASTFM_API_KEY not found. Enrichment will be skipped.")

    def _load_cache(self) -> Dict[str, List[str]]:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f)

    def get_artist_tags(self, artist: str) -> List[str]:
        """
        Fetches top tags for an artist.
        Uses caching to avoid hitting API limits.
        """
        if not self.api_key:
            return []
            
        # Check cache first
        if artist in self.cache:
            return self.cache[artist]
            
        params = {
            "method": "artist.gettoptags",
            "artist": artist,
            "api_key": self.api_key,
            "format": "json"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                tags = []
                for tag in data.get("toptags", {}).get("tag", [])[:5]: # Top 5 tags
                    tags.append(tag["name"])
                
                # Cache the result
                self.cache[artist] = tags
                self._save_cache()
                
                # Be polite to the API
                time.sleep(0.2)
                
                return tags
            else:
                print(f"Last.fm Error for {artist}: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching tags for {artist}: {e}")
            return []

    def fetch_missing_tags(self, library: List[Dict[str, Any]]):
        """
        Scans the library and fetches tags for any artists not currently in the cache.
        This is the slow operation.
        """
        if not self.api_key:
            return

        unique_artists = list(set(track['artist'] for track in library))
        artists_to_fetch = [a for a in unique_artists if a not in self.cache]
        
        if not artists_to_fetch:
            print("All artists are already cached.")
            return

        print(f"Fetching tags for {len(artists_to_fetch)} new artists...")
        
        # Create a progress bar in Streamlit if we are running there
        import streamlit as st
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, artist in enumerate(artists_to_fetch):
            self.get_artist_tags(artist)
            
            # Update progress
            progress = (i + 1) / len(artists_to_fetch)
            progress_bar.progress(progress)
            status_text.text(f"Fetching tags for: {artist}")
            
        status_text.text("Tag fetching complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    def get_similar_artists(self, artist: str, limit: int = 20) -> List[str]:
        """
        Fetches similar artists from Last.fm.
        """
        if not self.api_key:
            return []
            
        params = {
            "method": "artist.getsimilar",
            "artist": artist,
            "api_key": self.api_key,
            "format": "json",
            "limit": limit
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                artists = []
                for a in data.get("similarartists", {}).get("artist", []):
                    artists.append(a["name"])
                return artists
            else:
                print(f"Last.fm Error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching similar artists: {e}")
            return []

    def apply_tags(self, library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies currently cached tags to the library. 
        This is a fast in-memory operation.
        """
        if not self.api_key:
            return library

        enriched_library = []
        for track in library:
            new_track = track.copy()
            new_track['tags'] = self.cache.get(track['artist'], [])
            enriched_library.append(new_track)
            
        return enriched_library
