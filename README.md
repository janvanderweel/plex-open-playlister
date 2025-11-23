# 🎵 Plex Open Playlister (Alpha)

> **⚠️ Alpha Status**: This project is currently in early Alpha.
> *   **Gemini Integration**: Working ✅
> *   **Ollama Integration**: Working (with `qwen2.5-coder:7b` or similar) ✅
> *   **Journey Mode**: Working ✅

**Plex Open Playlister** is a standalone Python application that acts as an intelligent DJ for your Plex Media Server. It goes beyond simple shuffling by using AI and vector mathematics to curate playlists that flow perfectly.

## ✨ Features

### 1. 🧠 AI Prompt Mode
Describe what you want to hear in natural language.
*   *"Funky basslines for a summer BBQ"*
*   *"Melancholic jazz for a rainy night"*
*   **How it works**: Uses **Hybrid Search**. First, it uses semantic vector search (embeddings) to find the top 500 relevant tracks, then uses an LLM (Gemini or Ollama) to pick the best 20.

### 2. 📻 Artist Radio Mode
Create a playlist based on a specific artist.
*   **How it works**: Fetches "Similar Artists" from **Last.fm**, scans your library for matches, and creates a mix of the seed artist and their peers. No AI required—just pure musical metadata.

### 3. 🚀 Journey Mode
Create a seamless transition between two songs.
*   **Start**: "Morning Mood - Edvard Grieg"
*   **End**: "Enter Sandman - Metallica"
*   **How it works**: Uses **Vector Interpolation** and **Audio Analysis**. It calculates a path through the embedding space while ensuring smooth transitions in BPM and Key (via Spotify data).

### 4. 🎭 Mood Map Mode
Visualize your library on a 2D emotional map.
*   **Axes**: Valence (Happy/Sad) vs. Energy (Chill/Intense).
*   **How it works**: Plots your tracks using Spotify audio features. You simply drag the sliders to pick a target mood (e.g., "High Energy + High Happiness" for a party), and it finds the closest songs mathematically.

### 5. 🧭 Genre Explorer
Visualize your music collection as an interactive taxonomy tree.
*   **Visualization**: A Sunburst Chart allows you to see the breakdown of genres (Rock -> Metal -> Thrash).
*   **How it works**: Maps unstructured Last.fm tags to a structured genre hierarchy. You can select any node (e.g., "Jazz") to play a mix of that genre and all its sub-genres.

---

## 🛠️ Installation

### 1. Prerequisites
*   Python 3.10 or higher.
*   A running Plex Media Server.
*   A **Last.fm API Key** (Required for enrichment and Artist Radio).
*   (Optional) A Google Gemini API Key or local Ollama setup.

### 2. Setup
Clone the repository and enter the directory:
```bash
git clone https://github.com/yourusername/plex-open-playlister.git
cd plex-open-playlister
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create your configuration file:
```bash
cp .env.example .env
```

Open `.env` and fill in your details:

```ini
PLEX_URL=http://192.168.1.50:32400
PLEX_TOKEN=your_token
PLEX_MUSIC_SECTION_ID=1

# AI Choice: GEMINI or OLLAMA
LLM_CHOICE=GEMINI
GEMINI_API_KEY=your_key
# Or for Ollama:
# OLLAMA_URL=http://localhost:11434/api/generate
# LLM_MODEL=qwen2.5-coder:7b

# Last.fm (Required)
LASTFM_API_KEY=your_lastfm_key
LASTFM_SHARED_SECRET=your_secret
```

### 4. First Run & Enrichment
Start the app:
```bash
streamlit run main.py
```

1.  Go to the Sidebar.
2.  Click **"Fetch Missing Tags"**.
    *   This will scan your library and fetch genre tags from Last.fm for every artist.
    *   *Note*: This takes time on the first run (approx. 0.5s per artist).
    *   **Why?** This data powers the "Semantic Search" and "Journey Mode", making them much smarter.

## 🔑 Finding Your Config Values
*   **Plex Token**: Open an item in Plex Web > "Get Info" > "View XML". Look for `X-Plex-Token` in the URL.
*   **Last.fm Key**: Get one [here](https://www.last.fm/api/account/create).

## 🚧 Roadmap
*   [ ] **Frontend Settings**: Move configuration from `.env` to UI.
*   [ ] **Advanced Filtering**: Exclude specific genres manually.
*   [ ] **Mood Analysis**: Use audio analysis (BPM, key) for even better transitions.
