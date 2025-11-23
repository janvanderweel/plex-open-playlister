import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

base_url = os.getenv("PLEX_URL")
token = os.getenv("PLEX_TOKEN")
section_id = os.getenv("PLEX_MUSIC_SECTION_ID")

headers = {
    "X-Plex-Token": token,
    "Accept": "application/json"
}

# Fetch one track to inspect metadata
url = f"{base_url}/library/sections/{section_id}/all"
params = {
    "type": 10,
    "X-Plex-Container-Start": 0,
    "X-Plex-Container-Size": 1
}

try:
    response = requests.get(url, headers=headers, params=params, verify=False)
    data = response.json()
    track = data.get("MediaContainer", {}).get("Metadata", [])[0]
    print(json.dumps(track, indent=2))
except Exception as e:
    print(e)
