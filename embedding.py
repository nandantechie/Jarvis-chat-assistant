from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from src.config import Config
import os


class CustomSentenceTransformerEmbeddings(Embeddings):
    """Custom wrapper for SentenceTransformer to avoid device issues."""

    def __init__(self, model_name: str):
        self.device = 'cpu'  # Force CPU to avoid meta tensor issues

        print(f"[Embedding] Loading SentenceTransformer model: {model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=os.path.join(os.getcwd(), "model_cache")  # Optional: local cache folder
            )
            print(f"[Embedding] Model loaded successfully: {model_name}")
        except Exception as e:
            print(f"[Embedding] Failed to load model '{model_name}': {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings
        except Exception as e:
            print(f"[Embedding] Error embedding documents: {str(e)}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)[0]
            return embedding
        except Exception as e:
            print(f"[Embedding] Error embedding query: {str(e)}")
            return []


class EmbeddingManager:
    """Manages embeddings and retrieval using custom SentenceTransformer wrapper."""

    def __init__(self):
        model_name = Config.EMBEDDING_MODEL
        self.embedding_model = None

        # Attempt to load the primary model
        try:
            print(f"[EmbeddingManager] Attempting to load primary model: {model_name}")
            self.embedding_model = CustomSentenceTransformerEmbeddings(model_name)
        except Exception as e:
            print(f"[EmbeddingManager] Error loading primary model '{model_name}': {e}")

            # Attempt fallback model
            fallback_model = "all-MiniLM-L6-v2"
            try:
                print(f"[EmbeddingManager] Attempting fallback model: {fallback_model}")
                self.embedding_model = CustomSentenceTransformerEmbeddings(fallback_model)
            except Exception as e2:
                print(f"[EmbeddingManager] Error loading fallback model '{fallback_model}': {e2}")
                raise RuntimeError("Failed to initialize any embedding model")

        self.vectorstore = None
        self.retriever = None

    def create_embeddings(self, documents: List[Document]) -> bool:
        """
        Creates embeddings for documents and stores them in FAISS.
        """
        if not self.embedding_model:
            print("[EmbeddingManager] Embedding model is not initialized.")
            return False

        try:
            print("[EmbeddingManager] Creating FAISS index from documents...")
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embedding_model
            )

            print("[EmbeddingManager] Creating retriever...")
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": Config.TOP_K}
            )

            print("[EmbeddingManager] Embedding and retriever setup successful.")
            return True
        except Exception as e:
            print(f"[EmbeddingManager] Error creating embeddings or retriever: {str(e)}")
            return False

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Searches for relevant documents based on the query.
        """
        if k is None:
            k = Config.TOP_K

        if not self.vectorstore or not self.retriever:
            print("[EmbeddingManager] No vectorstore or retriever found.")
            return []

        try:
            print(f"[EmbeddingManager] Searching for query: '{query}' (top {k})")
            relevant_docs = self.retriever.get_relevant_documents(query)
            return relevant_docs
        except Exception as e:
            print(f"[EmbeddingManager] Error during search: {str(e)}")
            return []
