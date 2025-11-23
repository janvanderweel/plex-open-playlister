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
        
        def traverse(node, parent_label=""):
            name = node["name"]
            
            # Value is count of this specific node + sum of children?
            # For sunburst, leaf nodes have values, parents sum them up automatically usually.
            # But here a track might be tagged "Rock" but not "Alt Rock".
            # So we assign the direct count to this node.
            count = counts.get(name, 0)
            
            ids.append(name)
            labels.append(name)
            parents.append(parent_label)
            values.append(count)
            
            if "children" in node:
                for child in node["children"]:
                    traverse(child, name)
                    
        traverse(self.tree)
        
        return {
            "ids": ids,
            "labels": labels,
            "parents": parents,
            "values": values
        }
