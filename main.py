import streamlit as st
import os
from dotenv import load_dotenv
from plex_client import PlexClient
from llm_client import LLMClient
from enrichment import LastFMClient
from utils import match_tracks, clear_cache

# Load config
load_dotenv()

st.set_page_config(page_title="Plex Open Playlister", page_icon="🎵", layout="wide")

# Initialize Clients
@st.cache_resource
def get_plex_client():
    try:
        return PlexClient()
    except Exception as e:
        st.error(f"Failed to initialize Plex Client: {e}")
        return None

@st.cache_resource
def get_llm_client():
    try:
        return LLMClient()
    except Exception as e:
        st.error(f"Failed to initialize LLM Client: {e}")
        return None

@st.cache_resource
def get_lastfm_client():
    return LastFMClient()

plex_client = get_plex_client()
llm_client = get_llm_client()
lastfm_client = get_lastfm_client()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Status Indicators
    if plex_client:
        st.success("✅ Plex Client Initialized")
    else:
        st.error("❌ Plex Client Failed")
        
    if llm_client:
        st.success(f"✅ LLM Client ({os.getenv('LLM_CHOICE')}) Initialized")
    else:
        st.error("❌ LLM Client Failed")

    if lastfm_client and lastfm_client.api_key:
        st.success("✅ Last.fm Enrichment Active")
        if st.button("Fetch Missing Tags"):
            try:
                with st.spinner("Fetching library to identify artists..."):
                    lib = plex_client.fetch_library()
                lastfm_client.fetch_missing_tags(lib)
                st.success("Tags updated successfully!")
            except Exception as e:
                st.error(f"Error fetching tags: {e}")
    else:
        st.warning("⚠️ Last.fm Enrichment Inactive")
        
    st.divider()
    
    if st.button("Clear Library Cache"):
        clear_cache()
        st.toast("Cache cleared!", icon="🗑️")

# Main Content
st.title("🎵 Plex Open Playlister")
st.markdown("Generate AI-curated playlists for your Plex server.")

# Input
col1, col2 = st.columns([3, 1])
with col1:
    mode = st.radio("Mode", ["AI Prompt", "Artist Radio", "Journey"], horizontal=True)
    
    if mode == "AI Prompt":
        user_request = st.text_area("What do you want to listen to?", placeholder="e.g., 'Upbeat 80s pop songs for a workout'")
    elif mode == "Artist Radio":
        user_request = st.text_input("Enter an Artist Name", placeholder="e.g., 'James Brown'")
    elif mode == "Journey":
        # We need the library loaded to populate the selectboxes.
        # This is a bit tricky in Streamlit flow. We'll try to load from cache or fetch if needed.
        # For UX, we might need a "Load Library" button if it's empty, but let's try to be seamless.
        
        # Try to get library from cache without triggering a full fetch if possible, 
        # but PlexClient.fetch_library handles caching, so it's fast.
        if 'library_cache' not in st.session_state:
             with st.spinner("Loading library for Journey mode..."):
                 try:
                     st.session_state.library_cache = plex_client.fetch_library()
                 except:
                     st.session_state.library_cache = []
        
        library_options = st.session_state.library_cache
        
        if not library_options:
             st.warning("Library not loaded. Please click 'Curate Playlist' once or check connection.")
             start_track = None
             end_track = None
        else:
            # Create a mapping for display
            # Format: "Title - Artist"
            # We need to map back to 'key' later
            
            # Optimization: Create list of strings once
            if 'track_options' not in st.session_state:
                st.session_state.track_options = [f"{t['title']} - {t['artist']}" for t in library_options]
                st.session_state.track_map = {f"{t['title']} - {t['artist']}": t['key'] for t in library_options}
            
            c1, c2 = st.columns(2)
            with c1:
                start_track_name = st.selectbox("Start Song", st.session_state.track_options, index=0 if st.session_state.track_options else None)
            with c2:
                end_track_name = st.selectbox("End Song", st.session_state.track_options, index=len(st.session_state.track_options)-1 if st.session_state.track_options else None)
            
            start_track = st.session_state.track_map.get(start_track_name)
            end_track = st.session_state.track_map.get(end_track_name)
            
            user_request = f"Journey from {start_track_name} to {end_track_name}" # For display/logging

with col2:
    num_songs = st.slider("Number of Songs", min_value=5, max_value=100, value=20, step=5)

# Initialize session state for results
if 'matched_items' not in st.session_state:
    st.session_state.matched_items = None
if 'generated_request' not in st.session_state:
    st.session_state.generated_request = ""

if st.button("Curate Playlist", type="primary", disabled=not (plex_client and (llm_client if mode == "AI Prompt" else True))):
    if mode == "AI Prompt" and not user_request:
        st.warning("Please enter a request.")
    elif mode == "Artist Radio" and not user_request:
        st.warning("Please enter an artist name.")
    elif mode == "Journey" and (not start_track or not end_track):
        st.warning("Please select start and end tracks.")
    else:
        st.session_state.generated_request = user_request
        try:
            # 1. Fetch Library (if not already in session for Journey)
            library = []
            try:
                if mode == "Journey" and 'library_cache' in st.session_state:
                    library = st.session_state.library_cache
                else:
                    with st.spinner("Fetching Plex Library..."):
                        library = plex_client.fetch_library()
                        st.session_state.library_cache = library # Update session cache
                    
                # 1b. Enrich Library (Fast Apply)
                if lastfm_client:
                    library = lastfm_client.apply_tags(library)
                        
                st.info(f"Library loaded: {len(library)} tracks.")
            except Exception as e:
                st.error(f"Failed to fetch/enrich library: {e}")
                st.stop()
            
            suggested_titles = []
            
            # BRANCH: Journey
            if mode == "Journey":
                from semantic_search import SemanticSearch
                search_engine = SemanticSearch()
                
                with st.spinner("Calculating musical path..."):
                    # Ensure index exists (might need to build it if first time)
                    search_engine.index_library(library)
                    
                    journey_tracks = search_engine.generate_journey(start_track, end_track, num_songs)
                
                if not journey_tracks:
                    st.error("Failed to generate journey. Are embeddings generated?")
                else:
                    st.session_state.matched_items = [{
                        'track': t,
                        'score': 100,
                        'suggestion': f"Step {i+1}"
                    } for i, t in enumerate(journey_tracks)]
            
            # BRANCH: Artist Radio
            elif mode == "Artist Radio":
                if not lastfm_client or not lastfm_client.api_key:
                    st.error("Last.fm is required for Artist Radio.")
                    st.stop()
                    
                with st.spinner(f"Finding artists similar to '{user_request}'..."):
                    similar_artists = lastfm_client.get_similar_artists(user_request, limit=50)
                    # Add the seed artist themselves
                    similar_artists.insert(0, user_request)
                    
                if not similar_artists:
                    st.error("No similar artists found.")
                    st.stop()
                    
                st.write(f"Found {len(similar_artists)-1} similar artists: {', '.join(similar_artists[1:6])}...")
                
                # Filter library
                lib_artists = list(set(t['artist'] for t in library))
                valid_artists = []
                
                # Normalize for comparison
                sim_artists_lower = [a.lower() for a in similar_artists]
                
                for lib_artist in lib_artists:
                    if lib_artist.lower() in sim_artists_lower:
                        valid_artists.append(lib_artist)
                        
                if not valid_artists:
                    st.warning(f"None of the similar artists found in your library.")
                    st.stop()
                    
                st.success(f"Found {len(valid_artists)} matching artists in your library.")
                
                # Collect tracks
                candidate_tracks = [t for t in library if t['artist'] in valid_artists]
                
                # Randomly select num_songs
                import random
                if len(candidate_tracks) > num_songs:
                    selected_tracks = random.sample(candidate_tracks, num_songs)
                else:
                    selected_tracks = candidate_tracks
                    
                st.session_state.matched_items = [{
                    'track': t,
                    'score': 100,
                    'suggestion': f"Similar to {user_request}"
                } for t in selected_tracks]
                
            # BRANCH: AI Prompt
            else:
                # 2. Consult LLM
                try:
                    with st.spinner(f"Consulting AI Curator (Target: {num_songs} songs)..."):
                        suggested_titles = llm_client.generate_playlist(user_request, library, num_songs)
                except Exception as e:
                    st.error(f"Failed to generate playlist with LLM: {e}")
                    st.stop()
                
                if not suggested_titles:
                    st.error("The AI could not generate a playlist. Please try again.")
                else:
                    # 3. Match Tracks
                    with st.spinner(f"Matching {len(suggested_titles)} suggestions to library..."):
                        matched_items = match_tracks(suggested_titles, library)
                    
                    if not matched_items:
                        st.warning("No matching tracks found in your library.")
                        st.session_state.matched_items = None
                    else:
                        st.session_state.matched_items = matched_items

                    
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Display Results (Persistent)
if st.session_state.matched_items:
    st.subheader("Proposed Playlist")
    
    matched_items = st.session_state.matched_items
    
    # Prepare data for display
    display_data = []
    track_keys = []
    for item in matched_items:
        track = item['track']
        display_data.append({
            "Title": track['title'],
            "Artist": track['artist'],
            "Match Score": f"{item['score']}%",
            "Suggestion": item['suggestion']
        })
        track_keys.append(track['key'])
        
    st.table(display_data)
    
    # Create Playlist Button
    # Use the request that generated these results
    req_preview = st.session_state.generated_request[:20]
    playlist_name = f"AI: {req_preview}..."
    
    if st.button(f"Save as '{playlist_name}' to Plex"):
        try:
            with st.spinner("Creating playlist on Plex..."):
                created_title = plex_client.create_playlist(playlist_name, track_keys)
                st.success(f"Playlist '{created_title}' created successfully!")
                st.balloons()
        except Exception as e:
            st.error(f"Failed to create playlist on Plex: {e}")
