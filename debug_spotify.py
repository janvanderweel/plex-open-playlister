from spotify_client import SpotifyClient

client = SpotifyClient()
title = "Wake Me Up Before You Go-Go"
artist = "Wham!"

print(f"Searching for {title} - {artist}...")
sid = client.search_track(title, artist)
print(f"Spotify ID: {sid}")

if sid:
    # Try Audio Analysis
    print("Trying Audio Analysis...")
    import requests
    token = client._get_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(f"https://api.spotify.com/v1/audio-analysis/{sid}", headers=headers)
        print(f"Analysis Status: {r.status_code}")
        if r.status_code != 200:
            print(r.text)
    except Exception as e:
        print(e)
else:
    print("Track not found.")
