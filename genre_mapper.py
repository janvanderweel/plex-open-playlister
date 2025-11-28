import json
import os
from typing import List, Dict, Any, Set

GENRE_FILE = "genres.json"

class GenreMapper:
    def __init__(self):
        self.tree = self._load_tree()
        self.flat_map = {} # Map alias/name -> Node Name
        self.parent_map = {} # Map Node Name -> Parent Name
        self.node_children_map = {} # Map Node Name -> Set of all recursive children names
        
        if self.tree:
            self._build_maps(self.tree)

    def _load_tree(self):
        if os.path.exists(GENRE_FILE):
            try:
                with open(GENRE_FILE, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    def _build_maps(self, node, parent_name=None):
        name = node["name"]
        # Map self
        self.flat_map[name.lower()] = name
        if parent_name:
            self.parent_map[name] = parent_name
            
        # Map aliases
        for alias in node.get("aliases", []):
            self.flat_map[alias.lower()] = name
            
        # Recurse
        children_names = set()
        if "children" in node:
            for child in node["children"]:
                self._build_maps(child, name)
                children_names.add(child["name"])
                # Add child's children
                children_names.update(self.node_children_map.get(child["name"], set()))
        
        self.node_children_map[name] = children_names

    def map_track(self, track: Dict[str, Any]) -> List[str]:
        """
        Returns a list of 'Canonical Genres' for a track based on its tags.
        """
        if 'tags' not in track or not track['tags']:
            return []
            
        found_genres = set()
        for tag in track['tags']:
            tag_lower = tag.lower()
            if tag_lower in self.flat_map:
                found_genres.add(self.flat_map[tag_lower])
        
        return list(found_genres)

    def get_tracks_in_genre(self, genre_name: str, library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns all tracks that belong to genre_name OR any of its sub-genres.
        """
        target_genres = {genre_name}
        target_genres.update(self.node_children_map.get(genre_name, set()))
        
        matched_tracks = []
        for track in library:
            track_genres = self.map_track(track)
            # Check intersection
            if not set(track_genres).isdisjoint(target_genres):
                matched_tracks.append(track)
                
        return matched_tracks

    def get_tree_stats(self, library: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns the tree structure enriched with track counts for visualization.
        """
        # 1. Count tracks per canonical genre
        counts = {}
        for track in library:
            genres = self.map_track(track)
            for g in genres:
                counts[g] = counts.get(g, 0) + 1
                
        # 2. Recursive build for Plotly
        # Plotly Sunburst needs: ids, labels, parents, values
        ids = []
        labels = []
        parents = []
        values = []
        colors = []
        
        import colorsys
        
        # Define base hues for top-level genres to ensure distinct colors
        # We have about 10 top level genres
        base_hues = {
            "Rock": 0.0,      # Red
            "Electronic": 0.1, # Orange/Yellow
            "Pop": 0.18,      # Yellow
            "Hip Hop": 0.28,  # Green
            "Jazz": 0.45,     # Cyan
            "R&B": 0.55,      # Light Blue
            "Classical": 0.65,# Blue
            "Reggae": 0.75,   # Purple
            "Folk": 0.85,     # Magenta
            "All Music": 0.0  # White/Gray
        }

        def get_color(name, parent_name, depth, sibling_index, total_siblings):
            if name == "All Music":
                return "#ffffff"
            
            # Find the top-level ancestor to determine hue
            # For now, let's just pass hue down or look it up if we are at level 1
            
            hue = 0.0
            saturation = 0.7
            value = 0.9
            
            if parent_name == "All Music":
                # Level 1
                hue = base_hues.get(name, (sibling_index / max(1, total_siblings)))
            else:
                # Inherit hue from parent? 
                # Since we are traversing depth-first, we might need to pass hue down.
                # But our traverse function doesn't easily support passing state without changing signature significantly.
                # Let's use a simpler approach: 
                # If we are deep, we need to know our "root" genre.
                pass

            return "#cccccc" # Placeholder if logic is complex in single pass

        # Better approach: Pre-assign hues to all nodes
        node_hues = {}
        
        def assign_hues(node, parent_hue=None, level=0, sibling_idx=0, total_siblings=1):
            name = node["name"]
            
            if level == 0:
                hue = 0.0 # Unused for root
            elif level == 1:
                # Use predefined or distribute
                if name in base_hues:
                    hue = base_hues[name]
                else:
                    hue = sibling_idx / max(1, total_siblings)
            else:
                # Slight variation from parent
                # e.g. shift by small amount or keep same
                hue = parent_hue
                
            node_hues[name] = hue
            
            if "children" in node:
                children = node["children"]
                for i, child in enumerate(children):
                    assign_hues(child, hue, level + 1, i, len(children))
                    
        assign_hues(self.tree)

        def traverse(node, parent_label="", level=0):
            name = node["name"]
            
            count = counts.get(name, 0)
            
            ids.append(name)
            labels.append(name)
            parents.append(parent_label)
            values.append(count)
            
            # Generate Color
            if name == "All Music":
                colors.append("#eeeeee")
            else:
                h = node_hues.get(name, 0.0)
                # Vary saturation/lightness by level
                # Level 1: S=0.9, V=0.9
                # Level 2: S=0.75, V=0.9
                # Level 3: S=0.6, V=0.9
                s = max(0.3, 0.9 - (level * 0.15))
                v = 0.9
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
                colors.append(hex_color)
            
            if "children" in node:
                for child in node["children"]:
                    traverse(child, name, level + 1)
                    
        traverse(self.tree)
        
        return {
            "ids": ids,
            "labels": labels,
            "parents": parents,
            "values": values,
            "colors": colors
        }
