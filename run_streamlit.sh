#!/bin/bash
# Wrapper script to run Streamlit with Essentia library path
export LD_LIBRARY_PATH=/home/pi/plex-open-playlister/.venv/usr/local/lib:$LD_LIBRARY_PATH
exec /home/pi/plex-open-playlister/.venv/bin/streamlit "$@"
