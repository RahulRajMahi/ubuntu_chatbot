from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import create_chatbot
import os
import logging

from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure logging for FastAPI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Ubuntu Docs Q&A Chatbot", version="0.1.0")

# Define the input and output models for the API
class QueryRequest(BaseModel):
    query: str

class ChatbotResponse(BaseModel):
    response: str

# Initialize the chatbot when the application starts
try:
    chatbot_instance = create_chatbot(faiss_index_path="ubuntu_docs.faiss")
    if chatbot_instance is None:
        raise ValueError("Chatbot initialization failed.")
    logger.info("Chatbot instance created successfully.")
except Exception as e:
    logger.error(f"Error initializing chatbot: {e}")
    chatbot_instance = None

@app.post("/query/", response_model=ChatbotResponse)
async def query_chatbot(request: QueryRequest):
    """
    Endpoint to query the Ubuntu documentation chatbot.
    """
    if chatbot_instance is None:
        raise HTTPException(status_code=500, detail="Chatbot service is unavailable.")

    try:
        response = chatbot_instance(request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error querying chatbot: {e}")
        raise HTTPException(status_code=500, detail="Error processing your query.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)