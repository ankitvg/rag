import requests
from pathlib import Path

def download_book(url, filename):
    """Download a book from Project Gutenberg"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(data_dir / filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    print(f"Downloaded: {filename}")

# Sample books from Project Gutenberg
books = [
    ("https://www.gutenberg.org/files/11/11-0.txt", "alice_wonderland.txt"),
    ("https://www.gutenberg.org/files/1342/1342-0.txt", "pride_prejudice.txt"),
    ("https://www.gutenberg.org/files/74/74-0.txt", "tom_sawyer.txt"),
]

if __name__ == "__main__":
    for url, filename in books:
        download_book(url, filename)