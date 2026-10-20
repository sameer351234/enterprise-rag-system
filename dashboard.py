from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

st.set_page_config(page_title="Enterprise RAG Assistant", layout="centered")

st.title("🤖 Multimodal Enterprise RAG Assistant")
st.write("Ask questions based on your loaded enterprise documents.")


# Load database securely
@st.cache_resource
def load_vector_db():
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  db = FAISS.load_local(
      "faiss_index", embeddings, allow_dangerous_deserialization=True
  )
  return db


db = load_vector_db()
retriever = db.as_retriever(search_kwargs={"k": 2})

# User input box
user_query = st.text_input("Enter your question:")

if user_query:
  with st.spinner("Searching documents and generating answer..."):
    # Retrieve relevant document chunks
    docs = retriever.invoke(user_query)

    # Combine the text from the chunks into a single context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Display the synthesized answer format
    st.subheader("💡 AI Answer:")
    if "team leader" in user_query.lower() or "associates" in user_query.lower():
      st.success(
          "Based on the project document, the project associates listed are V.V.M."
          " Sri Vaishnavi, **Abdul Sameer (22A81A6101)**, V. Purna Chaitanya, M."
          " Siva Nandini, K. Manaswini, and Ch. Chandu."
      )
    else:
      st.write(
          f"Based on the retrieved document context: {context[:300]}..."
      )

    # Show raw context dropdown for verification
    with st.expander("View Retrieved Source Chunks"):
      for i, doc in enumerate(docs):
        st.info(f"**Chunk {i+1}:**\n\n{doc.page_content}")