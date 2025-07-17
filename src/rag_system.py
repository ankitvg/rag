import chromadb
import requests
import json
import re
import os
from typing import List, Dict
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from config import Config

console = Console()

class SimpleRAGSystem:
    def __init__(self, collection_name: str = "documents"):
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Config.CHROMA_SETTINGS
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            console.print(f"✓ Loaded existing collection: {collection_name}", style="green")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            console.print(f"✓ Created new collection: {collection_name}", style="blue")
    
    def get_jina_embedding(self, text: str) -> List[float]:
        """Get embedding from Jina model via Ollama"""
        try:
            response = requests.post(
                Config.get_ollama_embeddings_url(),
                json={
                    'model': Config.EMBEDDING_MODEL,
                    'prompt': text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['embedding']
        except Exception as e:
            console.print(f"❌ Error getting embedding: {e}", style="red")
            return []
    
    def generate_chunks_from_file(self, file_path: Path, chunk_size: int = None, overlap: int = None):
        """
        Read a file and yield chunks of text using a recursive splitting strategy.
        """
        chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        overlap = overlap or Config.DEFAULT_OVERLAP
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            console.print(f"❌ Error reading file: {e}", style="red")
            return

        # Define the separators to use for splitting, in order of preference
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_text(text_to_split: str, current_separators: List[str]) -> List[str]:
            """Recursively split text to find chunks of the appropriate size."""
            if len(text_to_split) <= chunk_size:
                return [text_to_split]
            
            # Get the next separator to try
            if not current_separators:
                # If no more separators, just split by character length
                return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size)]

            separator = current_separators[0]
            remaining_separators = current_separators[1:]
            
            # Split the text by the current separator
            splits = text_to_split.split(separator)
            
            # Process the splits
            final_chunks = []
            current_chunk = ""
            for s in splits:
                # If a split is too large, recurse
                if len(s) > chunk_size:
                    final_chunks.extend(split_text(s, remaining_separators))
                    continue

                # If adding the next split doesn't exceed chunk_size, append it
                if len(current_chunk) + len(s) + len(separator) <= chunk_size:
                    current_chunk += s + separator
                else:
                    # Otherwise, finalize the current_chunk and start a new one
                    final_chunks.append(current_chunk.strip())
                    current_chunk = s + separator
            
            # Add the last remaining chunk
            if current_chunk:
                final_chunks.append(current_chunk.strip())
                
            return final_chunks

        # Start the recursive splitting process
        initial_chunks = split_text(text, separators)
        
        # Handle overlap
        if overlap > 0 and len(initial_chunks) > 1:
            overlapped_chunks = [initial_chunks[0]]
            for i in range(1, len(initial_chunks)):
                # Get the last `overlap` characters from the previous chunk
                prev_chunk_overlap = overlapped_chunks[-1][-overlap:]
                # Prepend it to the current chunk
                overlapped_chunks.append(prev_chunk_overlap + initial_chunks[i])
            
            for chunk in overlapped_chunks:
                yield re.sub(r'\s+', ' ', chunk.strip())
        else:
            for chunk in initial_chunks:
                yield re.sub(r'\s+', ' ', chunk.strip())

    def add_document(self, file_path: str, document_id: str = None):
        """Add a document to the vector database from a file path"""
        path = Path(file_path)
        if not path.exists():
            console.print(f"❌ File not found: {path}", style="red")
            return

        doc_id = document_id or path.stem

        # --- New Feature: Display Document Info ---
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            word_count = len(content.split())
            
            console.print(f"📄 [bold]Processing document:[/] [cyan]{path.name}[/]")
            console.print(f"   - [bold]Word Count:[/] {word_count}")
            console.print(f"   - [bold]Chunk Size:[/] {Config.DEFAULT_CHUNK_SIZE} characters")
            console.print(f"   - [bold]Overlap:[/]    {Config.DEFAULT_OVERLAP} characters")
            
            # Estimate expected chunks
            effective_chunk_size = Config.DEFAULT_CHUNK_SIZE - Config.DEFAULT_OVERLAP
            if effective_chunk_size > 0:
                expected_chunks = (len(content) + effective_chunk_size - 1) // effective_chunk_size
                console.print(f"   - [bold]Est. Chunks:[/]  ~{expected_chunks}")
            console.print("-" * 30)

        except Exception as e:
            console.print(f"⚠️ Could not read file to count words: {e}", style="yellow")
        # --- End New Feature ---
        
        chunk_generator = self.generate_chunks_from_file(path)
        
        batch_size = 10
        batch_chunks = []
        total_chunks = 0

        with tqdm(desc=f"Processing {doc_id}", unit="chunk") as pbar:
            for chunk in chunk_generator:
                batch_chunks.append(chunk)
                if len(batch_chunks) >= batch_size:
                    self._process_batch(batch_chunks, doc_id, total_chunks, path)
                    total_chunks += len(batch_chunks)
                    pbar.update(len(batch_chunks))
                    batch_chunks = []
            
            if batch_chunks:
                self._process_batch(batch_chunks, doc_id, total_chunks, path)
                total_chunks += len(batch_chunks)
                pbar.update(len(batch_chunks))

        console.print(f"✅ Successfully added {total_chunks} chunks from {doc_id}", style="green")

    def _process_batch(self, batch_chunks: List[str], document_id: str, start_index: int, file_path: Path):
        """Helper to process a batch of chunks"""
        embeddings = []
        ids = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(batch_chunks):
            embedding = self.get_jina_embedding(chunk)
            if embedding:
                chunk_id = f"{document_id}_chunk_{start_index + i}"
                embeddings.append(embedding)
                ids.append(chunk_id)
                metadatas.append({
                    'document_id': document_id,
                    'chunk_index': start_index + i,
                    'chunk_size': len(chunk),
                    'source_file': str(file_path)
                })
                documents.append(chunk)
        
        if embeddings:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        # Get query embedding
        query_embedding = self.get_jina_embedding(query)
        
        if not query_embedding:
            return []
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]
            })
        
        return formatted_results
    
    def display_results(self, results: List[Dict], query: str):
        """Display search results in a nice format"""
        if not results:
            console.print("❌ No results found.", style="red")
            return
        
        console.print(f"\n🔍 Results for: '{query}'", style="bold blue")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Relevance", justify="right", width=10)
        table.add_column("Source", width=15)
        table.add_column("Content Preview", width=60)
        
        for i, result in enumerate(results, 1):
            relevance = f"{result['relevance_score']:.3f}"
            source = result['metadata']['document_id']
            preview = result['document']
            
            table.add_row(
                str(i),
                relevance,
                source,
                preview
            )
        
        console.print(table)
    
    def get_collection_info(self):
        """Get information about the collection"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }

    def get_db_size(self):
        """Calculates the total size of the ChromaDB directory."""
        total_size = 0
        start_path = Config.CHROMA_DB_PATH
        try:
            for dirpath, dirnames, filenames in os.walk(start_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    # skip if it is symbolic link
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
        except FileNotFoundError:
            return "Database directory not found."
        except Exception as e:
            return f"Error calculating size: {e}"

        # Format the size
        if total_size == 0:
            return "0 Bytes"
        size_name = ("Bytes", "KB", "MB", "GB", "TB")
        i = int(np.floor(np.log(total_size) / np.log(1024)))
        p = np.power(1024, i)
        s = round(total_size / p, 2)
        return f"{s} {size_name[i]}"

    def reset_collection(self):
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.create_collection(name=self.collection.name)
            console.print(f"✓ Collection '{self.collection.name}' has been reset.", style="yellow")
        except Exception as e:
            console.print(f"❌ Error resetting collection: {e}", style="red")
            # Ensure the collection still exists
            self.collection = self.client.get_or_create_collection(name=self.collection.name)