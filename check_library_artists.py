from plex_client import PlexClient
from utils import normalize_key
import os
from dotenv import load_dotenv

load_dotenv()

def check_artists():
    client = PlexClient()
    print("Fetching library...")
    library = client.fetch_library()
    
    all_artists = set(t['artist'] for t in library)
    print(f"Found {len(all_artists)} unique artists in library.")
    
    targets = ["Emma-Jean Thackray", "Kaidi Tatham", "TOKiMONSTA", "Jazzanova"]
    
    print("\n--- Checking Targets ---")
    for target in targets:
        # Exact match
        if target in all_artists:
            print(f"✅ Exact match found: '{target}'")
            continue
            
        # Case-insensitive match
        found = False
        for a in all_artists:
            if a.lower() == target.lower():
                print(f"✅ Case-insensitive match found: '{a}' (Target: '{target}')")
                found = True
                break
        if found: continue
        
        # Normalized match
        norm_target = normalize_key("", target).replace(" - ", "")
        for a in all_artists:
            norm_a = normalize_key("", a).replace(" - ", "")
            if norm_a == norm_target:
                print(f"✅ Normalized match found: '{a}' (Target: '{target}')")
                found = True
                break
        if found: continue
        
        print(f"❌ No match found for: '{target}'")
        
    print("\n--- Sample Artists ---")
    print(list(all_artists)[:20])

if __name__ == "__main__":
    check_artists()
