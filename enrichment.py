import os
import requests
import time
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from utils import normalize_key

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
        Scans library and fetches tags for artists not in cache.
        """
        if not self.api_key:
            return

        # Identify artists needing tags
        # We normalize artist names for caching
        artists_to_fetch = set()
        for track in library:
            # For artist tags, we just normalize the artist name
            # We can reuse normalize_key but pass empty title
            # Or just implement simple artist normalization here?
            # Let's use normalize_key("", artist) to be consistent with the cleaning logic (lowercase, punctuation, etc)
            # Actually, normalize_key returns "- artist", so let's just use the clean logic inside utils if we could, 
            # but for now let's just strip "the" and lowercase.
            
            # Better: Let's just use the artist name from the track, but clean it.
            # We'll rely on the fact that apply_tags will use the same logic.
            
            artist = track['artist']
            # Simple normalization for artist-only cache
            norm_artist = normalize_key("", artist).replace(" - ", "") 
            
            if norm_artist not in self.cache:
                # We store the ORIGINAL artist name to query API, but key by normalized
                artists_to_fetch.add((norm_artist, artist))
        
        if not artists_to_fetch:
            print("All artists have tags cached.")
            return

        print(f"Fetching tags for {len(artists_to_fetch)} artists...")
        
        import streamlit as st
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (norm_artist, original_artist) in enumerate(artists_to_fetch):
            tags = self.get_artist_tags(original_artist)
            
            # Cache even if empty
            self.cache[norm_artist] = tags
            
            if i % 10 == 0:
                self._save_cache()
            
            progress = (i + 1) / len(artists_to_fetch)
            progress_bar.progress(progress)
            status_text.text(f"Tagging: {original_artist}")
            
            time.sleep(0.2) # Rate limit
            
        self._save_cache()
        status_text.text("Tag enrichment complete!")
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
            # Try exact match first, then normalized
            tags = self.cache.get(track['artist'])
            if not tags:
                norm_artist = normalize_key("", track['artist']).replace(" - ", "")
                tags = self.cache.get(norm_artist, [])
            
            new_track['tags'] = tags
            enriched_library.append(new_track)
            
        return enriched_library

    def estimate_features_from_tags(self, tags: List[str]) -> Dict[str, float]:
        """
        Estimates valence and energy based on tags.
        """
        if not tags:
            return {"valence": 0.5, "energy": 0.5}
            
        valence_score = 0
        energy_score = 0
        count = 0
        
        # Keywords and their impact (valence, energy)
        # Range -1 to 1
        keywords = {
            "happy": (0.8, 0.2), "upbeat": (0.7, 0.8), "cheerful": (0.9, 0.5),
            "sad": (-0.9, -0.4), "melancholic": (-0.7, -0.6), "depressing": (-0.9, -0.7),
            "energetic": (0.4, 0.9), "dance": (0.6, 0.9), "party": (0.7, 0.9),
            "chill": (0.2, -0.7), "relaxing": (0.3, -0.8), "mellow": (0.1, -0.6),
            "calm": (0.2, -0.9), "ambient": (0.0, -0.8), "acoustic": (0.1, -0.5),
            "rock": (0.0, 0.6), "metal": (-0.2, 0.9), "punk": (-0.1, 0.8),
            "pop": (0.5, 0.6), "electronic": (0.1, 0.7), "jazz": (0.2, -0.2),
            "blues": (-0.3, -0.1), "classical": (0.1, -0.3), "folk": (0.1, -0.4),
            "soul": (0.3, -0.2), "funk": (0.6, 0.7), "disco": (0.7, 0.8)
        }
        
        found_match = False
        
        for tag in tags:
            tag_lower = tag.lower()
            for key, (v, e) in keywords.items():
                if key in tag_lower:
                    valence_score += v
                    energy_score += e
                    count += 1
                    found_match = True
        
        if not found_match:
            return {"valence": 0.5, "energy": 0.5}
            
        # Normalize to 0-1 range
        # Average the scores first
        avg_v = valence_score / count
        avg_e = energy_score / count
        
        # Map from [-1, 1] to [0, 1]
        final_valence = (avg_v + 1) / 2
        final_energy = (avg_e + 1) / 2
        
        return {"valence": final_valence, "energy": final_energy}

    def apply_estimated_features(self, library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies estimated features to tracks that are missing them.
        """
        enriched = []
        for track in library:
            new_track = track.copy()
            
            # Only estimate if missing
            if 'valence' not in new_track or new_track['valence'] is None:
                tags = new_track.get('tags', [])
                if not tags:
                    # Try to get tags if not present
                    tags = self.cache.get(track['artist'])
                    if not tags:
                        norm_artist = normalize_key("", track['artist']).replace(" - ", "")
                        tags = self.cache.get(norm_artist, [])
                
                if tags:
                    features = self.estimate_features_from_tags(tags)
                    new_track['valence'] = features['valence']
                    new_track['energy'] = features['energy']
                    new_track['features_source'] = 'lastfm_estimated'
            
            enriched.append(new_track)
        return enriched
