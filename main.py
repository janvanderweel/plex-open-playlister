import streamlit as st
import os
from dotenv import load_dotenv
from plex_client import PlexClient
from llm_client import LLMClient
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

plex_client = get_plex_client()
llm_client = get_llm_client()

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
    user_request = st.text_area("What do you want to listen to?", placeholder="e.g., 'Upbeat 80s pop songs for a workout' or 'Melancholy jazz for a rainy night'")
with col2:
    num_songs = st.slider("Number of Songs", min_value=5, max_value=100, value=20, step=5)

# Initialize session state for results
if 'matched_items' not in st.session_state:
    st.session_state.matched_items = None
if 'generated_request' not in st.session_state:
    st.session_state.generated_request = ""

if st.button("Curate Playlist", type="primary", disabled=not (plex_client and llm_client)):
    if not user_request:
        st.warning("Please enter a request.")
    else:
        st.session_state.generated_request = user_request
        try:
            # 1. Fetch Library
            library = []
            try:
                with st.spinner("Fetching Plex Library..."):
                    library = plex_client.fetch_library()
                    st.info(f"Library loaded: {len(library)} tracks.")
            except Exception as e:
                st.error(f"Failed to fetch library from Plex: {e}")
                st.stop()
            
            # 2. Consult LLM
            suggested_titles = []
            try:
                with st.spinner(f"Consulting AI Curator (Target: {num_songs} songs)..."):
                    suggested_titles = llm_client.generate_playlist(user_request, library, num_songs)
            except Exception as e:
                st.error(f"Failed to generate playlist with LLM: {e}")
                st.stop()
            
            if not suggested_titles:
                st.error("The AI could not generate a playlist. Please try again.")
                st.session_state.matched_items = None
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
