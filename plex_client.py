import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from utils import save_cache, load_cache

load_dotenv()

class PlexClient:
    def __init__(self):
        self.base_url = os.getenv("PLEX_URL")
        self.token = os.getenv("PLEX_TOKEN")
        self.section_id = os.getenv("PLEX_MUSIC_SECTION_ID")
        
        if not all([self.base_url, self.token, self.section_id]):
            raise ValueError("Missing Plex configuration in .env file.")
            
        self.headers = {
            "X-Plex-Token": self.token,
            "Accept": "application/json"
        }
        
        # Fetch Machine Identifier
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(self.base_url, headers=self.headers, timeout=5, verify=False)
            resp.raise_for_status()
            self.machine_id = resp.json().get("MediaContainer", {}).get("machineIdentifier")
            if not self.machine_id:
                print("Warning: Could not fetch machineIdentifier. Playlist creation might fail.")
        except Exception as e:
            print(f"Error fetching machineIdentifier: {e}")
            self.machine_id = None

    def fetch_library(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetches all tracks from the configured Plex music section.
        Uses caching to avoid unnecessary network calls.
        """
        if not force_refresh:
            cached_data = load_cache()
            if cached_data:
                return cached_data

        url = f"{self.base_url}/library/sections/{self.section_id}/all"
        params = {
            "type": 10  # Type 10 is for Tracks
        }
        
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()
            
            tracks = []
            # Parse the Plex response
            metadata = data.get("MediaContainer", {}).get("Metadata", [])
            
            for item in metadata:
                # Extract Media Part Key for streaming
                media_items = item.get("Media", [])
                part_key = None
                if media_items:
                    parts = media_items[0].get("Part", [])
                    if parts:
                        part_key = parts[0].get("key")

                tracks.append({
                    "title": item.get("title"),
                    "artist": item.get("grandparentTitle", "Unknown Artist"), # grandparentTitle is usually Album Artist
                    "album": item.get("parentTitle", "Unknown Album"),
                    "key": item.get("ratingKey"),
                    "part_key": part_key,
                    "duration": item.get("duration")
                })
            
            save_cache(tracks)
            return tracks
            
        except requests.exceptions.RequestException as e:
            # Mask token for logging
            safe_url = url.replace(self.token, "REDACTED") if self.token else url
            print(f"Error fetching Plex library from {safe_url}: {e}")
            raise e

    def get_stream_url(self, part_key: str) -> str:
        """
        Constructs a direct stream URL for a media part.
        """
        if not part_key:
            return None
        return f"{self.base_url}{part_key}?X-Plex-Token={self.token}"

    def create_playlist(self, name: str, track_keys: List[str]) -> str:
        """
        Creates a playlist in Plex with the given name and track keys.
        """
        if not track_keys:
            raise ValueError("No tracks provided for playlist creation.")
            
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # We need to provide 'uri' during creation to avoid 400 Bad Request.
        # We'll use the first batch for creation.
        batch_size = 50
        first_batch = track_keys[:batch_size]
        remaining_keys = track_keys[batch_size:]
        
        # Construct URI
        # Robust format: server://{machine_id}/com.plexapp.plugins.library/library/metadata/{key}
        if self.machine_id:
            uri_prefix = f"server://{self.machine_id}/com.plexapp.plugins.library/library/metadata"
            keys_str = ",".join(first_batch)
            uri = f"{uri_prefix}/{keys_str}"
        else:
            # Fallback to simple library URI
            keys_str = ",".join(first_batch)
            uri = f"library:///library/metadata/{keys_str}"

        create_url = f"{self.base_url}/playlists"
        create_params = {
            "type": "audio",
            "title": name,
            "smart": 0,
            "uri": uri
        }
        
        try:
            print(f"Creating playlist '{name}' with {len(first_batch)} initial tracks...")
            print(f"POST {create_url} with params: {create_params}")
            response = requests.post(create_url, headers=self.headers, params=create_params, timeout=10, verify=False)
            print(f"Create Response: {response.status_code} - {response.text}")
            response.raise_for_status()
            
            playlist_data = response.json().get("MediaContainer", {}).get("Metadata", [{}])[0]
            playlist_id = playlist_data.get("ratingKey")
            playlist_title = playlist_data.get("title")
            
            if not playlist_id:
                raise Exception("Failed to retrieve new playlist ID.")
                
            print(f"Playlist created with ID: {playlist_id}")
            
            # Step 2: Add remaining items if any
            if remaining_keys:
                for i in range(0, len(remaining_keys), batch_size):
                    batch = remaining_keys[i:i+batch_size]
                    
                    if self.machine_id:
                        uri_prefix = f"server://{self.machine_id}/com.plexapp.plugins.library/library/metadata"
                        keys_str = ",".join(batch)
                        uri = f"{uri_prefix}/{keys_str}"
                    else:
                        keys_str = ",".join(batch)
                        uri = f"library:///library/metadata/{keys_str}"
                    
                    add_url = f"{self.base_url}/playlists/{playlist_id}/items"
                    add_params = {
                        "uri": uri
                    }
                    
                    print(f"Adding remaining batch {i//batch_size + 1} to playlist...")
                    print(f"PUT {add_url} with params: {add_params}")
                    add_response = requests.put(add_url, headers=self.headers, params=add_params, timeout=10, verify=False)
                    print(f"Add Response: {add_response.status_code} - {add_response.text}")
                    add_response.raise_for_status()
            
            return playlist_title
            
        except requests.exceptions.RequestException as e:
            # Mask token for logging
            safe_url = create_url.replace(self.token, "REDACTED") if self.token else create_url
            print(f"Error creating/updating playlist at {safe_url}: {e}")
            raise e
