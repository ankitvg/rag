import argparse
from pathlib import Path
from rag_system import SimpleRAGSystem
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description='RAG Document Retrieval System')
    parser.add_argument('--collection', default='documents', help='Collection name')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add document command
    add_parser = subparsers.add_parser('add', help='Add document to collection')
    add_parser.add_argument('file_path', help='Path to the document file')
    add_parser.add_argument('--id', help='Document ID (optional)')
    add_parser.add_argument('--clean', action='store_true', help='Delete all existing data before adding')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search in collection')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--results', type=int, default=5, help='Number of results')
    
    # Info command
    subparsers.add_parser('info', help='Show collection information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize RAG system
    rag = SimpleRAGSystem(args.collection)
    
    if args.command == 'add':
        if args.clean:
            rag.reset_collection()
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