import streamlit as st
import os
from dotenv import load_dotenv
from plex_client import PlexClient
from llm_client import LLMClient
from enrichment import LastFMClient
from essentia_client import EssentiaClient
from utils import match_tracks, clear_cache, fill_missing_features

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

@st.cache_resource
def get_essentia_client():
    return EssentiaClient()

plex_client = get_plex_client()
llm_client = get_llm_client()
lastfm_client = get_lastfm_client()
essentia_client = get_essentia_client()

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

    # Essentia Status
    if essentia_client and essentia_client.available:
        st.success("✅ Essentia Analysis Active")
        if st.button("Fetch Audio Features"):
            try:
                # 1. Ensure we have a library to work with
                if 'library_cache' not in st.session_state:
                        with st.spinner("Fetching library..."):
                            st.session_state.library_cache = plex_client.fetch_library()
                
                lib = st.session_state.library_cache
                
                # 2. Fetch missing features (updates JSON cache)
                essentia_client.fetch_missing_features(lib)
                
                # 3. Re-apply features to the in-memory library immediately
                st.session_state.library_cache = essentia_client.apply_features(lib)
                
                # 4. Fallback to Last.fm Estimates
                if lastfm_client:
                    with st.spinner("Estimating missing features from Last.fm tags..."):
                        # Ensure tags are applied
                        st.session_state.library_cache = lastfm_client.apply_tags(st.session_state.library_cache)
                        # Estimate
                        st.session_state.library_cache = lastfm_client.apply_estimated_features(st.session_state.library_cache)
                
                # 5. Fallback Heuristics (if still missing)
                st.session_state.library_cache = fill_missing_features(st.session_state.library_cache)
                
                st.success("Audio features updated successfully (including estimates)!")
                st.rerun() # Force UI update
            except Exception as e:
                st.error(f"Error fetching features: {e}")
    else:
        st.warning("⚠️ Essentia Analysis Inactive (Not Installed)")
        st.info("Install essentia with `pip install essentia` or `pip install essentia-tensorflow`")
        
    st.divider()
    
    # Statistics
    if 'library_cache' in st.session_state and st.session_state.library_cache:
        lib = st.session_state.library_cache
        total = len(lib)
        if total > 0:
            # Last.fm Stats
            tagged = sum(1 for t in lib if 'tags' in t and t['tags'])
            st.caption(f"🏷️ Last.fm Coverage: {tagged}/{total} ({int(tagged/total*100)}%)")
            
            # Essentia Stats
            analyzed = sum(1 for t in lib if 'bpm' in t and t['bpm'] is not None)
            st.caption(f"🎵 Essentia Coverage: {analyzed}/{total} ({int(analyzed/total*100)}%)")
    
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
    mode = st.radio("Mode", ["AI Prompt", "Artist Radio", "Journey", "Mood Map", "Genre Explorer"], horizontal=True)
    
    # Clear results if mode changes
    if 'last_mode' not in st.session_state:
        st.session_state.last_mode = mode
    
    if mode != st.session_state.last_mode:
        st.session_state.matched_items = None
        st.session_state.generated_request = ""
        st.session_state.last_mode = mode
        st.rerun()
    
    if mode == "AI Prompt":
        user_request = st.text_area("What do you want to listen to?", placeholder="e.g., 'Upbeat 80s pop songs for a workout'")
    elif mode == "Artist Radio":
        user_request = st.text_input("Enter an Artist Name", placeholder="e.g., 'James Brown'")
        c1, c2 = st.columns(2)
        with c1:
            depth = st.slider("Discovery Depth", 1, 3, 2, help="1=Directly similar, 2=Similar to similar, etc.")
        with c2:
            max_per_artist = st.number_input("Max per Artist", 1, 20, 3, help="Prevent one artist from dominating the playlist.")
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
            
            user_request = f"Journey from {start_track_name} to {end_track_name}"

    elif mode == "Mood Map":
        # Ensure library is loaded for visualization
        if 'library_cache' not in st.session_state:
             with st.spinner("Loading library for Mood Map..."):
                 try:
                     st.session_state.library_cache = plex_client.fetch_library()
                     # Apply features if available
                     if essentia_client:
                         st.session_state.library_cache = essentia_client.apply_features(st.session_state.library_cache)
                     if lastfm_client:
                         st.session_state.library_cache = lastfm_client.apply_tags(st.session_state.library_cache)
                         st.session_state.library_cache = lastfm_client.apply_estimated_features(st.session_state.library_cache)
                     
                     # Fallback Heuristics
                     st.session_state.library_cache = fill_missing_features(st.session_state.library_cache)
                 except:
                     st.session_state.library_cache = []
        
        library = st.session_state.library_cache
        
        # Filter tracks that have audio features
        mood_tracks = [t for t in library if 'valence' in t and t['valence'] is not None]
        
        if not mood_tracks:
            st.warning("No tracks with audio features found. Please use 'Fetch Audio Features' in the sidebar first.")
            target_valence = 0.5
            target_energy = 0.5
        else:
            # Sliders for Target
            mc1, mc2 = st.columns(2)
            with mc1:
                target_valence = st.slider("Valence (Sad ↔ Happy)", 0.0, 1.0, 0.5)
            with mc2:
                target_energy = st.slider("Energy (Chill ↔ Intense)", 0.0, 1.0, 0.5)
            
            # Diversity Settings
            with st.expander("Diversity & Matching Settings", expanded=True):
                div_col1, div_col2 = st.columns(2)
                with div_col1:
                    search_radius = st.slider("Match Radius", 0.0, 1.0, 0.2, help="Songs within this distance are considered 'good matches'. Increase for more variety.")
                with div_col2:
                    max_per_artist = st.number_input("Max per Artist", 1, 50, 3, help="Maximum number of songs from a single artist.")
                
                randomize_selection = st.checkbox("Randomize within Radius", value=True, help="Shuffle songs that fall within the match radius to avoid picking the same ones every time.")
            
            user_request = f"Mood: Valence {target_valence:.2f}, Energy {target_energy:.2f}"
            
            # Visualization
            import pandas as pd
            import altair as alt
            
            # Prepare data for plot
            # Sample down if too many points for performance
            plot_data = mood_tracks if len(mood_tracks) < 2000 else mood_tracks[:2000]
            df = pd.DataFrame([{
                'Title': t['title'],
                'Artist': t['artist'],
                'Valence': t.get('valence', 0),
                'Energy': t.get('energy', 0)
            } for t in plot_data])
            
            # Base chart
            base = alt.Chart(df).mark_circle(size=60).encode(
                x='Valence',
                y='Energy',
                color=alt.value('lightgray'),
                tooltip=['Title', 'Artist', 'Valence', 'Energy']
            ).properties(
                height=300
            )
            
            # Target point
            target_df = pd.DataFrame([{'Valence': target_valence, 'Energy': target_energy, 'Label': 'Target'}])
            target_point = alt.Chart(target_df).mark_point(shape='cross', size=200, filled=True).encode(
                x='Valence',
                y='Energy',
                color=alt.value('red')
            )
            
            # Radius circle (visualize the diversity radius)
            # We can approximate a circle using a mark_point with size? 
            # Or just leave it as the cross. Visualizing the radius might be complex in Altair without generating circle data.
            # Let's stick to the cross for now.
            
            st.altair_chart(base + target_point, use_container_width=True)

    elif mode == "Genre Explorer":
        # Ensure library is loaded
        if 'library_cache' not in st.session_state:
             with st.spinner("Loading library for Genre Explorer..."):
                 try:
                     st.session_state.library_cache = plex_client.fetch_library()
                     if lastfm_client:
                         st.session_state.library_cache = lastfm_client.apply_tags(st.session_state.library_cache)
                 except:
                     st.session_state.library_cache = []
        
        library = st.session_state.library_cache
        
        from genre_mapper import GenreMapper
        # Always reload to ensure fresh genres.json
        st.session_state.genre_mapper = GenreMapper()
            
        mapper = st.session_state.genre_mapper
        
        # Prepare data for Sunburst
        stats = mapper.get_tree_stats(library)
        
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Sunburst(
            ids=stats['ids'],
            labels=stats['labels'],
            parents=stats['parents'],
            values=stats['values'],
            marker=dict(colors=stats['colors']),
            insidetextorientation='radial'
        ))
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0), height=400)
        
        # Use streamlit-plotly-events to capture clicks
        from streamlit_plotly_events import plotly_events
        
        # We need to handle the selection state
        if 'selected_genre_node' not in st.session_state:
            st.session_state.selected_genre_node = "All Music"
            
        # Reset Button
        if st.button("Reset Chart"):
            st.session_state.selected_genre_node = "All Music"
            st.rerun()

        # Render chart and capture click
        # Note: plotly_events returns a list of dicts
        selected_points = plotly_events(fig, click_event=True, hover_event=False, override_height=400)
        
        if selected_points:
            # The return format for Sunburst click is usually:
            # [{'pointNumber': 4, 'pointNumbers': [4], 'root': 'All Music', 'entry': 'Rock', 'label': 'Rock', ...}]
            # We want 'label' or 'entry' or 'id'
            clicked_point = selected_points[0]
            if 'label' in clicked_point:
                st.session_state.selected_genre_node = clicked_point['label']
            elif 'pointNumber' in clicked_point:
                # Fallback if label is missing (sometimes happens), try to map index
                # But ids list matches point numbers usually?
                idx = clicked_point['pointNumber']
                if idx < len(stats['ids']):
                    st.session_state.selected_genre_node = stats['ids'][idx]
        
        # Flatten tree for selectbox
        all_genres = sorted([g for g in stats['ids'] if g != "All Music"])
        
        # Sync selectbox with chart click
        # If clicked node is in our list, use it as default
        index_to_use = 0
        if st.session_state.selected_genre_node in all_genres:
            index_to_use = all_genres.index(st.session_state.selected_genre_node)
            
        selected_genre = st.selectbox("Select Genre to Play", all_genres, index=index_to_use)
        
        # Update state if selectbox changed manually (Streamlit handles this via key usually, but here we are mixing)
        # Actually, if user changes selectbox, it updates selected_genre.
        # If user clicks chart, it updates session_state.selected_genre_node, which updates selectbox index on rerun.
        
        user_request = selected_genre


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
            # 1. Fetch Library (if not already in session for Journey/Mood)
            library = []
            try:
                if (mode == "Journey" or mode == "Mood Map" or mode == "Genre Explorer") and 'library_cache' in st.session_state:
                    library = st.session_state.library_cache
                else:
                    with st.spinner("Fetching Plex Library..."):
                        library = plex_client.fetch_library()
                        st.session_state.library_cache = library # Update session cache
                    
                # 1b. Enrich Library (Fast Apply)
                if lastfm_client:
                    library = lastfm_client.apply_tags(library)
                
                # 1c. Enrich Audio Features (Fast Apply)
                if essentia_client:
                    library = essentia_client.apply_features(library)

                # 1d. Fallback Estimates
                if lastfm_client:
                    library = lastfm_client.apply_estimated_features(library)
                
                # 1e. Fallback Heuristics
                library = fill_missing_features(library)
                     # Add More Feature
                if st.button("➕ Add More Songs"):
                    try:
                        # 1. Fetch Library (cached)
                        library = st.session_state.library_cache if 'library_cache' in st.session_state else plex_client.fetch_library()
                        if lastfm_client: library = lastfm_client.apply_tags(library)
                        if essentia_client: library = essentia_client.apply_features(library)
                        
                        new_items = []
                        
                        # Logic based on mode
                        if mode == "AI Prompt":
                            with st.spinner(f"Generating more songs for '{user_request}'..."):
                                # Ask for 5 more
                                suggested = llm_client.generate_playlist(user_request, library, 5) 
                                matched = match_tracks(suggested, library)
                                if matched: new_items = matched
                                
                        elif mode == "Artist Radio":
                             with st.spinner(f"Finding more artists similar to '{user_request}'..."):
                                similar_artists = lastfm_client.get_similar_artists(user_request, limit=50)
                                similar_artists.insert(0, user_request)
                                
                                # Filter library
                                valid_artists = [a for a in list(set(t['artist'] for t in library)) if a.lower() in [s.lower() for s in similar_artists]]
                                candidate_tracks = [t for t in library if t['artist'] in valid_artists]
                                
                                # Filter out tracks already in the playlist
                                current_keys = set(item['track']['key'] for item in st.session_state.matched_items)
                                unique_candidates = [t for t in candidate_tracks if t['key'] not in current_keys]
                                
                                import random
                                if unique_candidates:
                                    selected = random.sample(unique_candidates, min(5, len(unique_candidates)))
                                    new_items = [{
                                        'track': t,
                                        'score': 100,
                                        'suggestion': f"Similar to {user_request}"
                                    } for t in selected]
                                
                        elif mode == "Journey":
                            st.warning("Adding more songs to a finished Journey is not yet supported. Try increasing the song count and regenerating.")
                        
                        if new_items:
                            st.session_state.matched_items.extend(new_items)
                            st.rerun()
                        elif mode != "Journey":
                            st.warning("Could not find more unique songs to add.")
                            
                    except Exception as e:
                        st.error(f"Error adding songs: {e}")
                        
                st.info(f"Library loaded: {len(library)} tracks.")
            except Exception as e:
                st.error(f"Failed to fetch/enrich library: {e}")
                st.stop()
            
            suggested_titles = []
            
            # BRANCH: Genre Explorer
            if mode == "Genre Explorer":
                from genre_mapper import GenreMapper
                # Always reload mapper to pick up json changes
                mapper = GenreMapper()
                st.session_state.genre_mapper = mapper
                
                with st.spinner(f"Finding tracks for {user_request}..."):
                    genre_tracks = mapper.get_tracks_in_genre(user_request, library)
                    
                if not genre_tracks:
                    st.warning(f"No tracks found for genre: {user_request}")
                    st.session_state.matched_items = None
                else:
                    # Random sample
                    import random
                    if len(genre_tracks) > num_songs:
                        selected = random.sample(genre_tracks, num_songs)
                    else:
                        selected = genre_tracks
                        
                    st.session_state.matched_items = [{
                        'track': t,
                        'score': 100,
                        'suggestion': f"Genre: {user_request}"
                    } for t in selected]
            
            # BRANCH: Mood Map
            elif mode == "Mood Map":
                # Filter for tracks with features
                candidates = [t for t in library if 'valence' in t and t['valence'] is not None]
                
                if not candidates:
                    st.error("No tracks with audio features found.")
                    st.stop()
                    
                # Calculate Euclidean distance
                import math
                
                scored_candidates = []
                for t in candidates:
                    v_diff = t['valence'] - target_valence
                    e_diff = t['energy'] - target_energy
                    distance = math.sqrt(v_diff**2 + e_diff**2)
                    
                    # Convert distance to score (0 distance = 100 score)
                    # Max possible distance is sqrt(1+1) = 1.414
                    score = max(0, 100 - (distance * 100))
                    
                    scored_candidates.append({
                        'track': t,
                        'score': int(score),
                        'distance': distance,
                        'suggestion': f"V:{t['valence']:.2f} E:{t['energy']:.2f}"
                    })
                
                # --- Diversity Logic ---
                # 1. Separate into "Good Matches" (within radius) and "Rest"
                good_matches = [x for x in scored_candidates if x['distance'] <= search_radius]
                rest = [x for x in scored_candidates if x['distance'] > search_radius]
                
                # 2. Sort 'rest' by distance (so we fallback gracefully to closest)
                rest.sort(key=lambda x: x['distance'])
                
                # 3. Handle 'good matches'
                if randomize_selection:
                    import random
                    random.shuffle(good_matches)
                else:
                    good_matches.sort(key=lambda x: x['distance'])
                    
                # 4. Combined candidate pool
                final_pool = good_matches + rest
                
                # 5. Selection with Artist Limits
                selected_items = []
                artist_counts = {}
                
                # Pass 1: Strict Limit
                for item in final_pool:
                    if len(selected_items) >= num_songs:
                        break
                        
                    artist = item['track']['artist']
                    count = artist_counts.get(artist, 0)
                    
                    if count < max_per_artist:
                        selected_items.append(item)
                        artist_counts[artist] = count + 1
                        
                # Pass 2: Fill if needed (ignoring limit)
                if len(selected_items) < num_songs:
                    # Get items not yet selected
                    selected_ids = set(x['track']['key'] for x in selected_items)
                    
                    for item in final_pool:
                        if len(selected_items) >= num_songs:
                            break
                        if item['track']['key'] not in selected_ids:
                            selected_items.append(item)
                            
                st.session_state.matched_items = selected_items
            
            # BRANCH: Journey
            elif mode == "Journey":
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
                
                # Depth Control (Moved to Input Section)
                # depth = st.slider("Discovery Depth", 1, 3, 1, help="1=Directly similar, 2=Similar to similar, etc. Higher depth finds more obscure connections.")
                
                with st.spinner(f"Finding artists similar to '{user_request}' (Depth {depth})..."):
                    # Returns dict {artist: weight}
                    similar_artists_map = lastfm_client.get_similar_artists_recursive(user_request, depth=depth, limit_per_step=10)
                    
                if not similar_artists_map:
                    st.error("No similar artists found.")
                    st.stop()
                    
                # Sort by weight for display
                sorted_artists = sorted(similar_artists_map.items(), key=lambda x: x[1], reverse=True)
                top_names = [a[0] for a in sorted_artists[:5] if a[0].lower() != user_request.lower()]
                st.write(f"Found {len(similar_artists_map)} related artists. Top matches: {', '.join(top_names)}...")
                
                # Filter library
                lib_artists = list(set(t['artist'] for t in library))
                valid_artists = []
                
                # Normalize for comparison
                sim_artists_lower = {a.lower(): w for a, w in similar_artists_map.items()}
                
                # Map library artist -> weight
                artist_weights = {}
                
                for lib_artist in lib_artists:
                    if lib_artist.lower() in sim_artists_lower:
                        weight = sim_artists_lower[lib_artist.lower()]
                        artist_weights[lib_artist] = weight
                        valid_artists.append(lib_artist)
                        
                if not valid_artists:
                    st.warning(f"None of the similar artists found in your library.")
                    st.stop()
                    
                st.success(f"Found {len(valid_artists)} matching artists in your library.")
                
                # Collect tracks with weights
                candidate_tracks = []
                track_weights = []
                
                for t in library:
                    if t['artist'] in artist_weights:
                        candidate_tracks.append(t)
                        track_weights.append(artist_weights[t['artist']])
                
                # Weighted Selection with Coverage Guarantee
                import random
                selected_tracks = []
                selected_indices = set()
                artist_counts = {}
                
                # Group tracks by artist
                artist_tracks = {}
                for t in candidate_tracks:
                    a = t['artist']
                    if a not in artist_tracks:
                        artist_tracks[a] = []
                    artist_tracks[a].append(t)
                    
                # Phase 1: Coverage (One per artist)
                # Sort artists by weight descending to prioritize best matches if we can't fit all
                # But shuffle within same weight to avoid alphabetical bias
                unique_artists = list(artist_tracks.keys())
                # We need artist weights. We have `artist_weights` dict.
                # Sort by (weight desc, random)
                random.shuffle(unique_artists) # Shuffle first for tie-breaking
                unique_artists.sort(key=lambda a: artist_weights[a], reverse=True)
                
                for artist in unique_artists:
                    if len(selected_tracks) >= num_songs:
                        break
                        
                    # Pick a random track from this artist
                    tracks = artist_tracks[artist]
                    choice = random.choice(tracks)
                    
                    selected_tracks.append(choice)
                    selected_indices.add(choice['key'])
                    artist_counts[artist] = 1
                    
                # Phase 2: Fill (Weighted Random)
                # Only if we still have space
                if len(selected_tracks) < num_songs:
                    attempts = 0
                    max_attempts = num_songs * 20
                    
                    while len(selected_tracks) < num_songs and attempts < max_attempts:
                        attempts += 1
                        # Pick one from full candidate pool
                        choice = random.choices(candidate_tracks, weights=track_weights, k=1)[0]
                        
                        # Check uniqueness and artist limit
                        if choice['key'] not in selected_indices:
                            artist = choice['artist']
                            if artist_counts.get(artist, 0) < max_per_artist:
                                selected_tracks.append(choice)
                                selected_indices.add(choice['key'])
                                artist_counts[artist] = artist_counts.get(artist, 0) + 1
                                
                    # Fallback: If strict limits prevented filling, try to fill with anything unique
                    if len(selected_tracks) < num_songs:
                         remaining = [t for t in candidate_tracks if t['key'] not in selected_indices]
                         random.shuffle(remaining)
                         for t in remaining:
                             if len(selected_tracks) >= num_songs:
                                 break
                             selected_tracks.append(t)
                             selected_indices.add(t['key'])
                            
                st.session_state.matched_items = [{
                    'track': t,
                    'score': int(artist_weights[t['artist']] * 100),
                    'suggestion': f"Similar to {user_request} (Depth match)"
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
                # Smart shuffle to avoid same artists back-to-back
                import random
                
                # Create a list of (key, artist) tuples
                # IMPORTANT: Convert all keys to strings to avoid mixed type issues
                tracks_with_artists = [(str(item['track']['key']), item['track']['artist']) for item in matched_items]
                
                # DEBUG: Print what we're starting with
                print(f"\n=== DEBUG: Starting with {len(tracks_with_artists)} tracks ===")
                for i, (key, artist) in enumerate(tracks_with_artists[:5]):
                    # Find the track title
                    track_title = matched_items[i]['track']['title']
                    print(f"  {i}: {track_title} - {artist} (key: {key})")
                
                # DEBUG: Check for duplicates
                all_keys = [key for key, artist in tracks_with_artists]
                unique_keys = set(all_keys)
                if len(all_keys) != len(unique_keys):
                    print(f"\n⚠️  WARNING: Found {len(all_keys) - len(unique_keys)} duplicate keys!")
                    from collections import Counter
                    key_counts = Counter(all_keys)
                    duplicates = {k: v for k, v in key_counts.items() if v > 1}
                    print(f"  Duplicate keys: {duplicates}")
                
                # Smart shuffle algorithm: spread out artists
                shuffled = []
                remaining = tracks_with_artists.copy()
                random.shuffle(remaining)  # Initial randomization
                
                while remaining:
                    # Try to find a track with a different artist than the last one
                    if shuffled:
                        last_artist = shuffled[-1][1]
                        # Look for a track with a different artist
                        different_artist_tracks = [t for t in remaining if t[1] != last_artist]
                        if different_artist_tracks:
                            next_track = random.choice(different_artist_tracks)
                        else:
                            # All remaining are same artist, just pick one
                            next_track = remaining[0]
                    else:
                        # First track, pick randomly
                        next_track = remaining[0]
                    
                    shuffled.append(next_track)
                    remaining.remove(next_track)
                
                # Extract just the keys (already strings)
                shuffled_keys = [key for key, artist in shuffled]
                
                # DEBUG: Print what we're sending
                print(f"\n=== DEBUG: Sending {len(shuffled_keys)} shuffled keys ===")
                print(f"First 5 keys: {shuffled_keys[:5]}")
                
                created_title = plex_client.create_playlist(playlist_name, shuffled_keys)
                st.success(f"Playlist '{created_title}' created successfully!")
                st.balloons()
        except Exception as e:
            st.error(f"Failed to create playlist on Plex: {e}")
