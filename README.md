# 🎵 Plex Open Playlister (Alpha)

> **⚠️ Alpha Status**: This project is currently in early Alpha.
> *   **Gemini Integration**: Working ✅
> *   **Ollama Integration**: Working (with `qwen2.5-coder:7b` or similar) ✅
> *   **Journey Mode**: Working ✅
> *   **Local Audio Analysis (Essentia)**: Working ✅

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
*   **How it works**: Uses **Vector Interpolation** and **Audio Analysis**. It calculates a path through the embedding space while ensuring smooth transitions in BPM and Key (via Essentia local audio analysis).

### 4. 🎭 Mood Map Mode
Visualize your library on a 2D emotional map.
*   **Axes**: Valence (Happy/Sad) vs. Energy (Chill/Intense).
*   **How it works**: Plots your tracks using Essentia audio features. You simply drag the sliders to pick a target mood (e.g., "High Energy + High Happiness" for a party), and it finds the closest songs mathematically.

### 5. 🧭 Genre Explorer
Visualize your music collection as an interactive taxonomy tree.
*   **Visualization**: A Sunburst Chart allows you to see the breakdown of genres (Rock -> Metal -> Thrash).
*   **How it works**: Maps unstructured Last.fm tags to a structured genre hierarchy. You can select any node (e.g., "Jazz") to play a mix of that genre and all its sub-genres.

---

## 🛠️ Installation

### Standard Installation (Linux/Mac/Windows)

#### 1. Prerequisites
*   Python 3.10 or higher.
*   A running Plex Media Server.
*   A **Last.fm API Key** (Required for enrichment and Artist Radio).
*   (Optional) A Google Gemini API Key or local Ollama setup.

#### 2. Setup
Clone the repository and enter the directory:
```bash
git clone https://github.com/yourusername/plex-open-playlister.git
cd plex-open-playlister
```

Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Note**: On most platforms, `essentia` will install from pre-built wheels. If compilation is required, see the Raspberry Pi instructions below.

#### 3. Configuration
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

#### 4. First Run & Enrichment
Start the app:
```bash
streamlit run main.py
```

1.  Go to the Sidebar.
2.  Click **"Fetch Missing Tags"**.
    *   This will scan your library and fetch genre tags from Last.fm for every artist.
    *   *Note*: This takes time on the first run (approx. 0.5s per artist).
    *   **Why?** This data powers the "Semantic Search" and "Journey Mode", making them much smarter.
3.  Click **"Fetch Audio Features"** (optional but recommended).
    *   This will analyze your audio files locally using Essentia to extract tempo, key, mode, and danceability.
    *   *Note*: This takes several hours for large libraries. You can run it in the background (see below).

---

### 🥧 Raspberry Pi 5 Installation

Essentia requires compilation from source on ARM64 platforms (Raspberry Pi). Follow these additional steps:

#### 1. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y build-essential libyaml-dev libfftw3-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    python3-dev libsamplerate0-dev libtag1-dev libeigen3-dev git
```

#### 2. Clone and Setup
```bash
git clone https://github.com/yourusername/plex-open-playlister.git
cd plex-open-playlister
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Compile Essentia
```bash
git clone https://github.com/MTG/essentia.git
cd essentia
./waf configure --mode=release --with-python --python=../venv/bin/python3 --no-msse
./waf
./waf install --destdir=../.venv
cd ..
```

**Note**: Compilation takes ~10 minutes on Raspberry Pi 5.

#### 4. Install Python Dependencies
```bash
# Copy Essentia to correct location
cp -r .venv/usr/local/lib/python3.11/site-packages/essentia .venv/lib/python3.11/site-packages/

# Install other dependencies
.venv/bin/pip install streamlit requests python-dotenv google-generativeai \
    thefuzz python-Levenshtein sentence-transformers scikit-learn pandas \
    altair plotly streamlit-plotly-events gitpython pydeck pydantic

# Downgrade NumPy for Essentia compatibility
.venv/bin/pip install 'numpy<2'
```

#### 5. Run the App
Use the wrapper script that sets the correct library path:
```bash
./run_streamlit.sh run main.py --server.address=0.0.0.0
```

Access from another device on your network at `http://raspberry-pi-ip:8501`

#### 6. Background Audio Analysis (Recommended for Pi)
Analyzing your library can take several hours. Run it in the background:
```bash
nohup ./run_analysis.sh > analysis.log 2>&1 &
```

Monitor progress:
```bash
# Check how many tracks are analyzed
python3 -c "import json; data=json.load(open('essentia_cache.json')); print(f'Tracks analyzed: {len(data)}')"

# View log (filtered)
tail -f analysis.log | grep -v "ScriptRunContext"
```

#### 7. Running as a Service (Optional)
To keep the app running in the background and start automatically on boot:

1.  **Install the Service**:
    ```bash
    sudo cp plex-open-playlister.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable plex-open-playlister.service
    sudo systemctl start plex-open-playlister.service
    ```

2.  **Manage the Service**:
    *   **Status**: `sudo systemctl status plex-open-playlister.service`
    *   **Stop**: `sudo systemctl stop plex-open-playlister.service`
    *   **Restart**: `sudo systemctl restart plex-open-playlister.service`
    *   **Logs**: `journalctl -u plex-open-playlister.service -f`

3.  **Uninstall the Service**:
    ```bash
    sudo systemctl stop plex-open-playlister.service
    sudo systemctl disable plex-open-playlister.service
    sudo rm /etc/systemd/system/plex-open-playlister.service
    sudo systemctl daemon-reload
    ```

---

## 🔑 Finding Your Config Values
*   **Plex Token**: Open an item in Plex Web > "Get Info" > "View XML". Look for `X-Plex-Token` in the URL.
*   **Last.fm Key**: Get one [here](https://www.last.fm/api/account/create).
*   **Gemini API Key**: Get one [here](https://aistudio.google.com/app/apikey).

## 🚧 Roadmap
*   [x] **Local Audio Analysis**: Use Essentia instead of Spotify API.
*   [ ] **Frontend Settings**: Move configuration from `.env` to UI.
*   [ ] **Advanced Filtering**: Exclude specific genres manually.
*   [ ] **Playlist Export**: Save playlists back to Plex.

