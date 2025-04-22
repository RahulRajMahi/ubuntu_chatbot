import os
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_ubuntu_docs_vector_store(
    docs_folder: str = "ubuntu-docs",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    faiss_index_name: str = "ubuntu_docs.faiss",
):
    """
    Creates a FAISS vector store from .md files within the specified folder and its subfolders
    using a Hugging Face Sentence Transformer model.

    Args:
        docs_folder (str, optional): Path to the folder containing the .md files.
                                     Defaults to "ubuntu-docs".
        embedding_model_name (str, optional): Name of the Hugging Face Sentence Transformer
                                               model to use. Defaults to "all-MiniLM-L6-v2".
        faiss_index_name (str, optional): Name of the FAISS index file to save.
                                           Defaults to "ubuntu_docs.faiss".

    Returns:
        bool: True if the vector store creation is successful, False otherwise.
    """
    logging.info(f"Starting vector store creation from '{docs_folder}'.")
    documents = []
    filenames = []

    try:
        if not os.path.isdir(docs_folder):
            logging.error(f"Folder '{docs_folder}' not found in the current directory.")
            return False

        md_files_paths = []
        for root, _, files in os.walk(docs_folder):
            for file in files:
                if file.endswith(".md"):
                    md_files_paths.append(os.path.join(root, file))

        if not md_files_paths:
            logging.warning(f"No .md files found in '{docs_folder}' or its subfolders.")
            return True

        logging.info(f"Found {len(md_files_paths)} .md files to process.")

        for file_path in tqdm(md_files_paths, desc="Reading documents"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    filenames.append(os.path.relpath(file_path, docs_folder))
            except FileNotFoundError:
                logging.error(f"File not found: {file_path}")
            except UnicodeDecodeError as e:
                logging.error(f"Error decoding file '{file_path}': {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while reading file '{file_path}': {e}")

        if not documents:
            logging.warning("No document content found after reading files.")
            return True

        logging.info(f"Loading embedding model: '{embedding_model_name}'.")
        try:
            model = SentenceTransformer(embedding_model_name)
            embeddings = model.encode(documents)
            embedding_dimension = embeddings.shape[1]
            logging.info(f"Embeddings generated with dimension: {embedding_dimension}.")
        except Exception as e:
            logging.error(f"Error loading or using the embedding model '{embedding_model_name}': {e}")
            return False

        logging.info("Creating FAISS index.")
        try:
            index = faiss.IndexFlatL2(embedding_dimension)
            index.add(np.array(embeddings).astype('float32'))
            faiss.write_index(index, faiss_index_name)
            logging.info(f"FAISS index saved to '{faiss_index_name}'.")

            metadata = {"filenames": filenames}
            np.savez(faiss_index_name.replace(".faiss", ".npz"), **metadata)
            logging.info(f"Metadata (filenames) saved to '{faiss_index_name.replace('.faiss', '.npz')}'.")

            return True

        except Exception as e:
            logging.error(f"Error creating or saving FAISS index: {e}")
            return False

    except Exception as overall_error:
        logging.critical(f"An unhandled error occurred during vector store creation: {overall_error}")
        return False

if __name__ == "__main__":
    success = create_ubuntu_docs_vector_store()
    if success:
        logging.info("Successfully created the Ubuntu documentation vector store.")
    else:
        logging.error("Failed to create the Ubuntu documentation vector store.")