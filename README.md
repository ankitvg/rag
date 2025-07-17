# RAG Document Retrieval System

This project is a simple command-line interface (CLI) for a Retrieval-Augmented Generation (RAG) system. It allows you to add documents to a vector database, search them, and view collection information.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python 3:** Make sure you have a modern version of Python 3 installed.
- **Ollama:** This project relies on a running Ollama instance for language model and embedding services. You can download it from [ollama.com](https://ollama.com/).

Once Ollama is running, you must pull the required embedding model:
```bash
ollama pull nomic-embed-text
```

## Getting Started

### 1. Activate Environment

First, activate your Python virtual environment.

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 2. Install Dependencies

Install the required Python packages using pip.

```bash
pip install -r requirements.txt
```

### 3. Download Sample Data

Download the sample books to populate the `data/` directory.

```bash
python scripts/download_sample_books.py
```

### 4. Using the CLI

The main entry point for interacting with the RAG system is `src/cli.py`.

**Add a document:**
```bash
python src/cli.py add data/alice_wonderland.txt --id alice
```

**Search for information:**
```bash
python src/cli.py search "What happens to Alice?" --results 3
```

**Get collection information:**
```bash
python src/cli.py info
```

### 5. Customize and Play

Explore the codebase to see how the RAG system works and tweak its parameters.

- **Hyperparameters:** Open `src/config.py` to modify key values like `EMBEDDING_MODEL`, `DEFAULT_CHUNK_SIZE`, and `DEFAULT_OVERLAP`.
- **Batch Processing:** In `src/rag_system.py`, you can adjust the `batch_size` within the `add_document` method to control how many chunks are processed at once.

#### Use collections:
```bash
python src/cli.py --collection fiction add data/tom_sawyer.txt --id sawyer  
python src/cli.py --collection fiction search "Who is accused of murdering Dr. Robinson?" --results 1
python src/cli.py --collection fiction info
```

Experiment with these settings to see how they impact retrieval performance and resource usage.
