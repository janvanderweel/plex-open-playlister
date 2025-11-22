import os
import json
import requests
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.choice = os.getenv("LLM_CHOICE", "GEMINI").upper()
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        
        if self.choice == "GEMINI":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for Gemini.")
            genai.configure(api_key=api_key)
            
            # Disable safety settings to prevent blocking on song titles/lyrics
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            self.model = genai.GenerativeModel(self.model_name, safety_settings=self.safety_settings)
            
        elif self.choice == "OLLAMA":
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
            
        else:
            raise ValueError(f"Unsupported LLM_CHOICE: {self.choice}")

    def generate_playlist(self, user_prompt: str, library_context: List[Dict[str, Any]], num_songs: int = 20) -> List[str]:
        """
        Generates a list of song titles based on the user prompt and library context.
        Uses Semantic Search to pre-filter, then LLM for final curation.
        """
        import time
        import random
        from semantic_search import SemanticSearch
        
        # Initialize Semantic Search
        search_engine = SemanticSearch()
        # This will load/create embeddings. It might take a moment on first run.
        search_engine.index_library(library_context)
        
        # Step 1: Semantic Search Pre-filtering
        # We get a large pool of candidates (e.g., 500) that are semantically relevant.
        # This filters out completely irrelevant genres/artists efficiently.
        print("Running semantic search...")
        semantic_candidates = search_engine.search(user_prompt, top_k=500)
        
        if not semantic_candidates:
            print("Semantic search found no results. Falling back to random sample.")
            semantic_candidates = random.sample(library_context, min(500, len(library_context)))
            
        print(f"Semantic search narrowed library to {len(semantic_candidates)} candidates.")
        
        # Step 2: LLM Curation
        # Now we use the LLM to pick the best songs from this refined list.
        
        # Shuffle to ensure variety within the top matches
        random.shuffle(semantic_candidates) 
        
        CHUNK_SIZE = 200 # Smaller chunks for Ollama since we have fewer total items
        all_candidates = []
        
        simplified_library = [f"{t['title']} - {t['artist']}" for t in semantic_candidates]
        total_tracks = len(simplified_library)
        
        # Split into chunks
        chunks = [simplified_library[i:i + CHUNK_SIZE] for i in range(0, total_tracks, CHUNK_SIZE)]
        
        print(f"Processing {len(chunks)} chunks for {total_tracks} tracks. Target: {num_songs} songs.")
        
        # We want to find enough candidates to fulfill the request, plus some buffer
        target_candidates = int(num_songs * 1.5) 
        
        for i, chunk in enumerate(chunks):
            # Construct prompt for this chunk
            system_instruction = (
                "You are a music curator. You have access to the following list of tracks:\n"
                f"{json.dumps(chunk)}\n\n"
                f"User Request: '{user_prompt}'\n\n"
                "Task: Select tracks from the provided list that STRICTLY fit the User Request.\n"
                "Rules:\n"
                "1. Return ONLY a JSON object with a single key 'tracks'.\n"
                "2. 'tracks' must be a list of strings copied EXACTLY from the provided list.\n"
                "3. Use your internal knowledge to verify the genre. If the song is by 'Elvis Presley' and the request is 'Funk', DO NOT include it. Elvis is Rock/Rockabilly, not Funk.\n"
                "4. If the request is specific (e.g., '80s Pop'), check the release year if you know it, or the artist's era.\n"
                "5. Better to return an empty list than a wrong song.\n"
            )
            
            try:
                if self.choice == "GEMINI":
                    batch_matches = self._call_gemini(system_instruction)
                elif self.choice == "OLLAMA":
                    batch_matches = self._call_ollama(system_instruction)
                else:
                    batch_matches = []
                
                if batch_matches:
                    print(f"Chunk {i+1}/{len(chunks)}: Found {len(batch_matches)} matches.")
                    all_candidates.extend(batch_matches)
                else:
                    print(f"Chunk {i+1}/{len(chunks)}: No matches.")
                    
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                # Continue to next chunk instead of failing completely
                continue
                
            # Rate limiting delay (less critical for local Ollama but good practice)
            time.sleep(1)
            
            # Optimization: If we have enough tracks, we could stop early.
            if len(all_candidates) >= target_candidates:
                print(f"Found enough candidates ({len(all_candidates)}), stopping early.")
                break
        
        # Return the requested number of songs
        return all_candidates[:num_songs]

    def _call_gemini(self, prompt: str) -> List[str]:
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            return self._parse_json_response(text)
        except Exception as e:
            # Re-raise to be caught by main.py
            raise Exception(f"Gemini API Error: {e}")

    def _call_ollama(self, prompt: str) -> List[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            text = response.json().get("response", "")
            return self._parse_json_response(text)
        except Exception as e:
            raise Exception(f"Ollama API Error: {e}")

    def _parse_json_response(self, text: str) -> List[str]:
        """Parses the LLM response to extract the list of tracks."""
        # Clean up markdown code blocks if present
        clean_text = text.strip()
        if clean_text.startswith("```"):
            # Handle ```json and ```
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                clean_text = "\n".join(lines[1:-1])
        
        try:
            data = json.loads(clean_text)
            return data.get("tracks", [])
        except json.JSONDecodeError:
            raise Exception(f"Failed to parse JSON. Raw output: {text[:500]}...")
