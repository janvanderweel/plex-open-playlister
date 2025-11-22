# 🎵 Plex Open Playlister (Alpha)

> **⚠️ Alpha Status**: This project is currently in early Alpha. The core functionality works, but expect some rough edges.
> *   **Gemini Integration**: Working ✅
> *   **Ollama Integration**: Experimental / Unstable 🚧
> *   **Settings**: Currently managed via `.env` file (UI planned).

**Plex Open Playlister** is a standalone Python application that acts as an intelligent DJ for your Plex Media Server. It uses Large Language Models (LLMs) to understand natural language requests (e.g., *"Funky basslines for a summer BBQ"*) and builds a playlist from your local music library.

## ✨ Features
*   **AI-Powered Curation**: Uses Google Gemini to understand mood, genre, and era.
*   **Fuzzy Matching**: Smartly matches AI suggestions to your actual files, handling slight spelling differences.
*   **Batch Processing**: Capable of processing libraries with thousands of tracks by splitting them into manageable chunks.
*   **Direct Plex Integration**: Creates the playlist directly on your server using the robust `server://` URI format.
*   **Customizable**: Choose how many songs you want in your playlist.

## 🛠️ Installation

### 1. Prerequisites
*   Python 3.10 or higher.
*   A running Plex Media Server.
*   A Google Gemini API Key (Free tier available).

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
Create your configuration file by copying the example:
```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in the required values.

## 🔑 Finding Your Config Values

### Plex Configuration
*   **`PLEX_URL`**: Your server's local IP address and port.
    *   *Example*: `http://192.168.1.50:32400` (Ensure you use `http` or `https` correctly).
*   **`PLEX_TOKEN`**: Your authentication token.
    *   **How to find it**: Open Plex Web, go to any media item, click the **Three Dots (⋮)** > **Get Info** > **View XML**. The token is in the URL bar at the very end: `X-Plex-Token=...`
*   **`PLEX_MUSIC_SECTION_ID`**: The ID of your music library.
    *   **How to find it**: Open Plex Web and navigate to your Music library. Look at the URL in your browser. It will look like `.../section/2/...`. The number (e.g., `2`) is your ID.

### AI Configuration
*   **`LLM_CHOICE`**: Set this to `GEMINI`. (Ollama is currently unstable).
*   **`GEMINI_API_KEY`**: Required for Gemini.
    *   **Get it here**: [Google AI Studio](https://aistudio.google.com/app/apikey).
*   **`LLM_MODEL`**: Recommended: `gemini-1.5-flash` or `gemini-2.0-flash-lite-preview-02-05`.

## ▶️ Usage

Start the application:
```bash
streamlit run main.py
```

The web interface will open automatically (usually at `http://localhost:8501`).
1.  Enter your prompt (e.g., *"90s Hip Hop with great flow"*).
2.  Select the number of songs.
3.  Click **Curate Playlist**.
4.  Review the matches and click **Save to Plex**.

## 🚧 Roadmap
*   [ ] **Frontend Settings**: Move configuration from `.env` to a settings page in the UI.
*   [ ] **Fix Ollama**: Stabilize local LLM support.
*   [ ] **Advanced Filtering**: Allow excluding specific artists or genres manually.
