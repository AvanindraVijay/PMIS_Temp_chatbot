from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import torch
import logging
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot with Rasa Intent Detection")

try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Sentence Transformer model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}")
    model = None


BASE_DIR = r"E:\PMIS"
DB_FAISS_PATH = os.path.abspath("vectorstore/db_faiss")  
MODEL_PATH = os.path.join(BASE_DIR, "model1", "llama-2-7b-chat.ggmlv3.q4_0.bin")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cpu":
    logger.warning("No GPU found! Using CPU mode which may be slower.")

custom_prompt_template = """You are an expert in answering questions related to PMIS internships.
    Given the context and the question, please provide a clear, concise, and accurate answer.

    Context: {context}
    Question: {question}

    Answer:"""

@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def load_llm():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Ensure it is downloaded correctly.")
    
    logger.info(f"Loading LLaMA model from {MODEL_PATH}")
    try:
        llm = CTransformers(
            model=MODEL_PATH,
            model_type="llama",
            max_new_tokens=1024,
            temperature=0.1,
            device=device
        )
        logger.info("LLaMA model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLaMA model: {e}")
        raise

FAISS_DB = None 

def load_faiss_db():
    global FAISS_DB
    if FAISS_DB is not None:
        return FAISS_DB
        
    logger.info("Initializing HuggingFace embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(DB_FAISS_PATH):
        logger.error(f"FAISS database not found at: {DB_FAISS_PATH}")
        return None

    try:
        logger.info(f"Loading FAISS database from: {DB_FAISS_PATH}")
        FAISS_DB = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS database loaded successfully!")
        return FAISS_DB
    except Exception as e:
        logger.error(f"Error loading FAISS: {e}")
        return None

def search_faiss(query):
    if FAISS_DB is None:
        db = load_faiss_db()
        if db is None:
            return None
    results = FAISS_DB.similarity_search(query, k=1)
    return results if results else None

def send_to_rasa(query):
    """Send query to Rasa server with proper format"""
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    try:

        payload = {"sender": "test_user", "message": query}
        logger.info(f"Sending to Rasa: {query}")
        response = requests.post(rasa_url, json=payload, timeout=10)
        
        if response.ok and response.json():
            return response.json()[0].get('text', "No response from Rasa.")
        else:
            logger.warning("No usable response from Rasa, falling back to LLaMA")
            return None  
    except requests.exceptions.RequestException as e:
        logger.error(f"Rasa connection error: {e}")
        return None

qa_chain = None

def qa_bot():
    global qa_chain
    if qa_chain is not None:
        return qa_chain
        
    db = load_faiss_db()
    if db is None:
        raise RuntimeError("FAISS DB loading failed.")
    
    llm = load_llm()
    prompt = set_custom_prompt()

    logger.info("Initializing QA chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    logger.info("QA chain initialized successfully")
    return qa_chain


@app.on_event("startup")
async def startup_event():
    try:
        qa_bot()
        logger.info("QA bot loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load QA bot: {e}")


class QueryRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    query = request.message
    logger.info(f"Received query: {query}")
    

    import time
    start_time = time.time()
    

    if qa_chain is None:
        try:
            qa_bot()
        except Exception as e:
            logger.error(f"Failed to initialize QA bot on demand: {e}")
            return {"response": "I'm sorry, I'm having technical difficulties at the moment."}
    

    try:
        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "I don't have enough information to answer that question.")
        logger.info(f"LLaMA processing took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with LLaMA: {e}")
        answer = "Sorry, I'm having trouble processing your question right now."
    
    return {"response": answer}