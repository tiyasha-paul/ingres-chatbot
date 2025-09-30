from fastapi import FastAPI
from main import rag_with_fallback, FAISS, HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import uvicorn

# Load your API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# Load the vectorstore when the API starts
@app.on_event("startup")
async def startup_event():
    global vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create a chat endpoint
@app.get("/chat")
async def chat(query: str, lang: str = "en"):
    try:
        answer, _ = rag_with_fallback(query, vectorstore, lang)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)