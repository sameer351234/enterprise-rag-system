from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="Enterprise RAG API")

# Define request body model
class Query(BaseModel):
    query: str

# Load embeddings and vector database on startup
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

@app.post("/ask")
def ask_question(q: Query):
    # Retrieve relevant document chunks based on user query
    relevant_docs = retriever.invoke(q.query)

    # Extract text content from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    return {
        "query": q.query,
        "retrieved_context": context,
        "message": "Successfully retrieved context for your query!",
    }
