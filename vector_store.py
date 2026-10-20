import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load and split documents
pdf_path = os.path.join("data", "sample.pdf")
if os.path.exists(pdf_path):
  loader = PyPDFLoader(pdf_path)
  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  )
  chunks = text_splitter.split_documents(docs)

  # 2. Initialize HuggingFace Embeddings model
  print("Creating vector embeddings (this might take a moment)...")
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )

  # 3. Store chunks into FAISS vector database
  vector_db = FAISS.from_documents(chunks, embeddings)

  # 4. Save the vector database locally
  vector_db.save_local("faiss_index")
  print("Vector database successfully created and saved to 'faiss_index'!")
else:
  print("Please ensure 'sample.pdf' exists inside the 'data' folder.")
