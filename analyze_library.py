#!/usr/bin/env python3
"""
Background script to analyze Plex library with Essentia.
Runs independently of the Streamlit app.
"""
import os
import sys
from dotenv import load_dotenv
from plex_client import PlexClient
from essentia_client import EssentiaClient

def main():
    print("=== Plex Library Audio Analysis ===")
    print("This will analyze all tracks in your Plex library using Essentia.")
    print("Progress is saved every 5 tracks, so you can stop/resume anytime.\n")
    
    # Load environment
    load_dotenv()
    
    # Initialize clients
    print("Initializing Plex client...")
    try:
        plex = PlexClient()
    except Exception as e:
        print(f"ERROR: Failed to initialize Plex client: {e}")
        sys.exit(1)
    
    print("Initializing Essentia client...")
    essentia = EssentiaClient()
    
    if not essentia.available:
        print("ERROR: Essentia is not available. Please check installation.")
        sys.exit(1)
    
    # Fetch library
    print("\nFetching Plex library...")
    try:
        library = plex.fetch_library()
        print(f"Found {len(library)} tracks in library.\n")
    except Exception as e:
        print(f"ERROR: Failed to fetch library: {e}")
        sys.exit(1)
    
    # Analyze
    print("Starting audio analysis...")
    print("(This will take several hours. You can safely Ctrl+C to stop and resume later.)\n")
    
    try:
        essentia.fetch_missing_features(library)
        print("\n✅ Analysis complete!")
        print(f"Results saved to: essentia_cache.json")
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        print("Progress has been saved. Run this script again to continue.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
