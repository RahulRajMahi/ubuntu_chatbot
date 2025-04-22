import os
import logging
from dotenv import load_dotenv
import faiss
import numpy as np
from langchain_community.docstore import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logging.error("GOOGLE_API_KEY not found in environment variables. Please set it.")
    exit()

# Gemini 1.5 Flash Model Name
GEMINI_MODEL_NAME = "gemini-1.5-flash"

def load_faiss_index(index_path: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads the FAISS index and associated metadata.
    """
    try:
        index = faiss.read_index(index_path)
        metadata_path = index_path.replace(".faiss", ".npz")
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True)
            filenames = metadata.get("filenames").tolist()
            logging.info(f"FAISS index loaded from '{index_path}' with {index.ntotal} embeddings and metadata.")
            return index, filenames
        else:
            logging.warning(f"Metadata file '{metadata_path}' not found.")
            return index, None
    except FileNotFoundError:
        logging.error(f"FAISS index file not found at '{index_path}'.")
        return None, None
    except Exception as e:
        logging.error(f"Error loading FAISS index from '{index_path}': {e}")
        return None, None

def create_chatbot(faiss_index_path: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
    """
    Creates a chatbot that uses the FAISS index and Google Gemini model to answer user queries.
    """
    index, filenames = load_faiss_index(faiss_index_path, embedding_model_name)
    if index is None:
        logging.error("Chatbot initialization failed due to FAISS index loading error.")
        return None

    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    documents = []
    ubuntu_docs_folder = "ubuntu-docs"

    if filenames:
        for i, filename in enumerate(filenames):
            file_path = os.path.join(ubuntu_docs_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(page_content=content, metadata={"source": filename, "id": str(i)}))
            except FileNotFoundError:
                logging.error(f"File not found while loading content: {file_path}")
                return None
            except Exception as e:
                logging.error(f"Error reading content from {file_path}: {e}")
                return None
    else:
        logging.warning("No filenames found in metadata. Cannot load document content.")
        # Handle the case where no filenames are found to prevent errors later
        if not index:
            logging.error("FAISS index also not loaded. Cannot initialize vectorstore.")
            return None
        documents = [Document(page_content="dummy", metadata={"source": "none", "id": "0"})] # Create a dummy document
        filenames = ["none"] # Create a dummy filename

    docstore = InMemoryDocstore({doc.metadata['id']: doc for doc in documents})
    index_to_docstore_id = {i: doc.metadata['id'] for i, doc in enumerate(documents)}

    # Initialize FAISS with the loaded index, docstore, and mapping
    embedding_dimension = index.d if index else embedding_function.embedding_size
    vectorstore = FAISS(embedding_function, index, docstore, index_to_docstore_id)

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=google_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    qa_chain = RetrievalQA.from_llm(llm, retriever=retriever)

    def chatbot(query: str):
        """
        Processes a user query and returns a response from the chatbot.
        """
        logging.info(f"User query: {query}")

        relevant_docs = retriever.invoke(query)
        logging.info("Top 5 relevant documents:")
        intermediate_output = "Here are the top 5 most relevant documents found:\n"
        for doc in relevant_docs:
            similarity_score = vectorstore.similarity_search_with_score(query, k=5) # Get scores here
            for d, score in similarity_score:
                if d.metadata['source'] == doc.metadata['source']:
                    similarity_percentage = f"{score * 100:.2f}%"
                    intermediate_output += f"- {doc.metadata['source']} (Similarity: {similarity_percentage})\n"
                    logging.info(f"- {doc.metadata['source']} (Similarity: {similarity_percentage})")
                    break
        print(intermediate_output) # Show intermediate output to the user

        try:
            response = qa_chain.invoke({"query": query})
            logging.info(f"Chatbot response: {response['result']}")
            return response['result']
        except Exception as e:
            logging.error(f"Error querying the LLM: {e}")
            logging.error(f"Detailed error: {e}")
            return "Sorry, I encountered an error while generating the response."

    return chatbot

if __name__ == "__main__":
    # Define the path to your FAISS index file
    faiss_index_file = "ubuntu_docs.faiss"  # Assuming you used the default name

    # Create the chatbot
    ubuntu_chatbot = create_chatbot(faiss_index_file)

    if ubuntu_chatbot:
        while True:
            user_query = input("Ask a question about Ubuntu (or Start new, continue or stop? (start/continue/stop)): ")
            if user_query.lower() == 'stop':
                break
            answer = ubuntu_chatbot(user_query)
            print(f"Chatbot: {answer}\n")
    else:
        print("Chatbot could not be initialized.")