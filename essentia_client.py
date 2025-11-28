import os
import json
import time
import streamlit as st
from typing import Dict, List, Any, Optional
from utils import normalize_key

CACHE_FILE = "essentia_cache.json"

class EssentiaClient:
    def __init__(self):
        self.available = False
        try:
            import essentia
            import essentia.standard as es
            self.es = es
            self.available = True
        except ImportError:
            print("Essentia not found. Please install it using 'pip install essentia' or 'pip install essentia-tensorflow'.")
            
        self.cache = self._load_cache()

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

    def analyze_track(self, file_path: str) -> Dict[str, Any]:
        """
        Analyzes a track using Essentia to extract audio features.
        """
        if not self.available:
            return {}
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {"not_found": True}

        try:
            # Load audio
            # We use MonoLoader for downmixing
            loader = self.es.MonoLoader(filename=file_path)
            audio = loader()
            
            features = {}
            
            # 1. Rhythm (BPM)
            rhythm_extractor = self.es.RhythmExtractor2013(method="multifeature")
            bpm, _, _, _, _ = rhythm_extractor(audio)
            features['tempo'] = float(bpm)
            
            # 2. Key / Mode
            key_extractor = self.es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            
            # Map Key to Spotify integer format (0=C, 1=C#, etc.)
            key_map = {
                "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, 
                "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, 
                "A": 9, "A#": 10, "Bb": 10, "B": 11
            }
            features['key'] = key_map.get(key, -1)
            features['mode'] = 1 if scale == 'major' else 0
            
            # 3. Danceability
            # Essentia's Danceability algorithm returns a value usually 0-3ish?
            # It's not directly 0-1. But let's extract it.
            dance_extractor = self.es.Danceability()
            danceability, _ = dance_extractor(audio)
            # Normalize? Let's just clamp it 0-1 for now or leave it. 
            # Spotify uses 0-1. Essentia danceability can be > 1.
            # A sigmoid or simple scaling might be needed. 
            # For now, let's cap at 1.0.
            features['danceability'] = min(float(danceability) / 3.0, 1.0) # Rough heuristic
            
            # 4. Energy / Valence
            # Without high-level models, these are hard.
            # We will return None so Last.fm estimator can fill them in.
            features['energy'] = None
            features['valence'] = None
            
            # Optional: Use RMS for Energy proxy if we really want
            # rms = self.es.RMS()(audio)
            # features['energy'] = float(rms) * 5 # Arbitrary scaling
            
            return features
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {"error": str(e)}

    def fetch_missing_features(self, library: List[Dict[str, Any]]):
        """
        Scans library and analyzes tracks not in cache.
        """
        if not self.available:
            # Check if running in Streamlit context
            try:
                st.warning("Essentia is not installed. Skipping local analysis.")
            except:
                print("WARNING: Essentia is not installed. Skipping local analysis.")
            return

        tracks_to_process = []
        for track in library:
            cache_key = normalize_key(track['title'], track['artist'])
            cached_val = self.cache.get(cache_key)
            
            # Process if not in cache or if previously failed/not found
            if not cached_val or cached_val.get("not_found") or cached_val.get("error"):
                # We need the file path!
                if track.get('file_path'):
                    tracks_to_process.append((cache_key, track))
        
        if not tracks_to_process:
            print("All tracks have audio features cached.")
            return

        print(f"Analyzing {len(tracks_to_process)} tracks with Essentia...")
        
        # Check if running in Streamlit context
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            use_streamlit = True
        except:
            progress_bar = None
            status_text = None
            use_streamlit = False
        
        for i, (cache_key, track) in enumerate(tracks_to_process):
            file_path = track['file_path']
            
            if use_streamlit:
                status_text.text(f"Analyzing: {track['title']}...")
            else:
                # Print progress every 10 tracks in background mode
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(tracks_to_process)} - Analyzing: {track['title']}")
            
            features = self.analyze_track(file_path)
            self.cache[cache_key] = features
            
            if i % 5 == 0:
                self._save_cache()
            
            if use_streamlit:
                progress = (i + 1) / len(tracks_to_process)
                progress_bar.progress(progress)
            
        self._save_cache()
        
        if use_streamlit:
            status_text.text("Audio analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
        else:
            print(f"\n✅ Analysis complete! Processed {len(tracks_to_process)} tracks.")

    def apply_features(self, library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies cached audio features to the library.
        """
        enriched_library = []
        for track in library:
            new_track = track.copy()
            cache_key = normalize_key(track['title'], track['artist'])
            features = self.cache.get(cache_key)
            
            if features and not features.get("not_found") and not features.get("error"):
                new_track['bpm'] = features.get('tempo')
                new_track['musical_key'] = features.get('key')
                new_track['mode'] = features.get('mode')
                if features.get('energy') is not None:
                    new_track['energy'] = features.get('energy')
                if features.get('valence') is not None:
                    new_track['valence'] = features.get('valence')
                if features.get('danceability') is not None:
                    new_track['danceability'] = features.get('danceability')
            
            enriched_library.append(new_track)
            
        return enriched_library
