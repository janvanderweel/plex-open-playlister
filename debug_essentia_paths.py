import os
import sys
from dotenv import load_dotenv
from plex_client import PlexClient
from essentia_client import EssentiaClient

def test_essentia_setup():
    print("--- Testing Essentia Setup ---")
    
    # 1. Load Env
    load_dotenv()
    print("Environment loaded.")
    
    # 2. Initialize Plex Client
    try:
        plex = PlexClient()
        print("Plex Client initialized.")
    except Exception as e:
        print(f"Failed to init Plex Client: {e}")
        return

    # 3. Fetch Library (limit to 5 items for speed if possible, but fetch_library gets all)
    print("Fetching Plex Library...")
    try:
        library = plex.fetch_library()
        print(f"Fetched {len(library)} tracks.")
    except Exception as e:
        print(f"Failed to fetch library: {e}")
        return

    if not library:
        print("Library is empty.")
        return

    # 4. Check File Paths
    print("\n--- Checking File Paths ---")
    found_count = 0
    missing_count = 0
    
    # Check first 5 tracks
    for i, track in enumerate(library[:5]):
        path = track.get('file_path')
        print(f"Track {i+1}: {track['title']} - {path}")
        
        if path and os.path.exists(path):
            print(f"  [OK] File exists.")
            found_count += 1
        else:
            print(f"  [FAIL] File NOT found.")
            missing_count += 1
            
    print(f"\nSummary: {found_count} found, {missing_count} missing (in first 5).")
    
    if found_count == 0:
        print("CRITICAL: No files found. Path mapping might be needed.")
        return

    # 5. Test Essentia Analysis
    print("\n--- Testing Essentia Analysis ---")
    essentia = EssentiaClient()
    if not essentia.available:
        print("Essentia is NOT available.")
        return
    
    # Pick the first valid file
    valid_track = next((t for t in library if t.get('file_path') and os.path.exists(t['file_path'])), None)
    
    if valid_track:
        print(f"Analyzing: {valid_track['title']}...")
        features = essentia.analyze_track(valid_track['file_path'])
        print("Result:", features)
    else:
        print("No valid file to analyze.")

if __name__ == "__main__":
    test_essentia_setup()
