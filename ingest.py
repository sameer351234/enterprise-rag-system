import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    JSONLoader
)

def load_document(file_path):
    """Automatically detects file type and loads the appropriate loader."""
    if file_path.endswith(".pdf"):
        print("Detected PDF file...")
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        print("Detected Text file...")
        return TextLoader(file_path).load()
    elif file_path.endswith(".csv"):
        print("Detected CSV file...")
        return CSVLoader(file_path).load()
    elif file_path.endswith(".docx"):
        print("Detected Word file...")
        return UnstructuredWordDocumentLoader(file_path).load()
    elif file_path.endswith(".md"):
        print("Detected Markdown file...")
        return UnstructuredMarkdownLoader(file_path).load()
    elif file_path.endswith(".html") or file_path.endswith(".htm"):
        print("Detected HTML file...")
        return UnstructuredHTMLLoader(file_path).load()
    elif file_path.endswith(".json"):
        print("Detected JSON file...")
        return JSONLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file format for: {file_path}")

# 1. Define your data folder
data_folder = "data"
all_documents = []

# 2. Loop through all files in the 'data' folder
if os.path.exists(data_folder):
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        if os.path.isfile(file_path):
            try:
                docs = load_document(file_path)
                all_documents.extend(docs)
                print(f"Successfully loaded: {filename}")
            except Exception as e:
                print(f"Could not load {filename}. Error: {e}")
else:
    print(f"The '{data_folder}' folder does not exist.")

# 3. Split documents into small chunks
if all_documents:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"\nTotal chunks created from all files: {len(chunks)}")
    print("\n--- Example of Chunk #1 ---")
    print(chunks[0].page_content)
else:
    print("No documents found to process.")
