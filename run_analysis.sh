#!/bin/bash
# Wrapper to run the analysis script with Essentia library path
export LD_LIBRARY_PATH=/home/pi/plex-open-playlister/.venv/usr/local/lib:$LD_LIBRARY_PATH
exec /home/pi/plex-open-playlister/.venv/bin/python3 -u /home/pi/plex-open-playlister/analyze_library.py "$@"
