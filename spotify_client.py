import os
import requests
import base64
import time
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from utils import normalize_key

load_dotenv()

CACHE_FILE = "spotify_cache.json"

class SpotifyClient:
    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        if self.client_id: self.client_id = self.client_id.strip()
        
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if self.client_secret: self.client_secret = self.client_secret.strip()
        
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501")
        if self.redirect_uri: self.redirect_uri = self.redirect_uri.strip()
        
        self.token_info = self._load_token()
        self.cache = self._load_cache()
        
        if not self.client_id or not self.client_secret:
            print("Warning: Spotify credentials not found.")

        if not self.client_id or not self.client_secret:
            print("Warning: Spotify credentials not found.")

    def _load_token(self):
        if os.path.exists("spotify_token.json"):
            try:
                with open("spotify_token.json", "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_token(self, token_info):
        with open("spotify_token.json", "w") as f:
            json.dump(token_info, f)
        self.token_info = token_info

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
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

    def get_auth_url(self):
        import urllib.parse
        scope = "user-read-private" # Minimal scope needed? Or maybe none for public features? 
        # Actually for audio-features we might not need specific scope, just user context.
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": scope
        }
        return f"https://accounts.spotify.com/authorize?{urllib.parse.urlencode(params)}"

    def get_token_from_code(self, code):
        auth_str = f"{self.client_id}:{self.client_secret}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {b64_auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri
        }
        
        response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
        if response.status_code != 200:
            error_msg = f"Token exchange failed: {response.text}"
            print(error_msg)
            raise Exception(error_msg)
        
        self._save_token(response.json())
        return True

    def _get_token(self):
        # No token at all
        if not self.token_info:
            return None

        now = time.time()
        expires_at = self.token_info.get("expires_at")

        # If we don't know expiration — assume expired and refresh
        if not expires_at or now > expires_at:
            return self._refresh_token()

        return self.token_info["access_token"]


    def _refresh_token(self):
        refresh = self.token_info.get("refresh_token")
        if not refresh:
            print("No refresh token available!")
            return None

        auth_str = f"{self.client_id}:{self.client_secret}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()

        headers = {
            "Authorization": f"Basic {b64_auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh
        }

        response = requests.post("https://accounts.spotify.com/api/token",
                                headers=headers, data=data)

        if response.status_code != 200:
            print("Token refresh failed:", response.text)
            return None

        new_data = response.json()

        # Spotify WILL NOT resend "refresh_token" every time
        # Keep your old one if it’s missing
        if "refresh_token" not in new_data:
            new_data["refresh_token"] = refresh

        # Compute and store expires_at
        new_data["expires_at"] = time.time() + new_data.get("expires_in", 3600)

        self._save_token(new_data)
        return new_data["access_token"]


    def search_track(self, title: str, artist: str) -> Optional[str]:
        """
        Searches for a track and returns its Spotify ID.
        """
        token = self._get_token()
        if not token:
            return None
            
        headers = {"Authorization": f"Bearer {token}"}
        query = f"track:{title} artist:{artist}"
        params = {"q": query, "type": "track", "limit": 1}
        
        try:
            response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("tracks", {}).get("items", [])
                if items:
                    return items[0]["id"]
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2))
                print(f"Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                return self.search_track(title, artist) # Recursive retry
            else:
                print(f"Search failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error searching Spotify for {title} - {artist}: {e}")
            
        return None

    def get_audio_features(self, spotify_id: str) -> Optional[Dict[str, float]]:
        """
        Fetches audio features for a track ID.
        """
        token = self._get_token()
        if not token:
            return None
            
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            response = requests.get(f"https://api.spotify.com/v1/audio-features/{spotify_id}", headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2))
                print(f"Rate limited (features). Waiting {retry_after}s...")
                time.sleep(retry_after)
                return self.get_audio_features(spotify_id)
            else:
                print(f"Error fetching features: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error fetching audio features for {spotify_id}: {e}")
            
        return None

    def fetch_missing_features(self, library: List[Dict[str, Any]]):
        """
        Scans library and fetches audio features for tracks not in cache.
        """
        if not self.client_id:
            return

        # Identify tracks needing features
        tracks_to_process = []
        for track in library:
            # Use normalized key
            cache_key = normalize_key(track['title'], track['artist'])
            cached_val = self.cache.get(cache_key)
            
            # Retry if not in cache or if previously not found
            if not cached_val or cached_val.get("not_found"):
                tracks_to_process.append((cache_key, track))
        
        if not tracks_to_process:
            print("All tracks have audio features cached.")
            return

        print(f"Fetching audio features for {len(tracks_to_process)} tracks...")
        
        import streamlit as st
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (cache_key, track) in enumerate(tracks_to_process):
            # We search using the raw title/artist for better Spotify search results,
            # but store under the normalized key.
            spotify_id = self.search_track(track['title'], track['artist'])
            
            features = {}
            if spotify_id:
                features = self.get_audio_features(spotify_id)
            
            self.cache[cache_key] = features if features else {"not_found": True}
            
            if i % 10 == 0:
                self._save_cache()
            
            progress = (i + 1) / len(tracks_to_process)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing: {track['title']}")
            
            time.sleep(0.1)
            
        self._save_cache()
        status_text.text("Audio analysis complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    def apply_features(self, library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies cached audio features to the library.
        """
        enriched_library = []
        for track in library:
            new_track = track.copy()
            cache_key = normalize_key(track['title'], track['artist'])
            features = self.cache.get(cache_key)
            
            if features and not features.get("not_found"):
                new_track['bpm'] = features.get('tempo')
                new_track['key'] = features.get('key')
                new_track['mode'] = features.get('mode')
                new_track['energy'] = features.get('energy')
                new_track['valence'] = features.get('valence')
                new_track['danceability'] = features.get('danceability')
            
            enriched_library.append(new_track)
            
        return enriched_library
