import argparse
import shutil
import os
from pathlib import Path
from rag_system import SimpleRAGSystem
from config import Config
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description='RAG Document Retrieval System')
    parser.add_argument('--collection', default='documents', help='Collection name for all operations')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)
    
    # Add document command
    add_parser = subparsers.add_parser('add', help='Add document to collection')
    add_parser.add_argument('file_path', help='Path to the document file')
    add_parser.add_argument('--id', help='Document ID (optional)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search in collection')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--results', type=int, default=5, help='Number of results')
    
    # Info command
    subparsers.add_parser('info', help='Show collection information')

    # Clean command
    subparsers.add_parser('clean', help='Clean (delete) the entire database directory')
    
    args = parser.parse_args()

    # Handle the clean command as a special case before initializing the system
    if args.command == 'clean':
        db_path = Config.CHROMA_DB_PATH
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
                console.print(f"✓ Successfully deleted database directory: {db_path}", style="bold green")
            except Exception as e:
                console.print(f"❌ Error deleting directory {db_path}: {e}", style="bold red")
        else:
            console.print(f"Directory {db_path} not found. Nothing to clean.", style="yellow")
        return  # Exit after cleaning

    # For all other commands, initialize the RAG system
    rag = SimpleRAGSystem(args.collection)
    
    if args.command == 'add':
        rag.add_document(args.file_path, args.id)
        db_size = rag.get_db_size()
        console.print(f"📊 Database size: {db_size}", style="bold yellow")
    
    elif args.command == 'search':
        results = rag.search(args.query, args.results)
        rag.display_results(results, args.query)
    
    elif args.command == 'info':
        info = rag.get_collection_info()
        console.print(f"📊 Collection: {info['collection_name']}")
        console.print(f"📄 Total chunks: {info['total_chunks']}")



if __name__ == "__main__":
    main()