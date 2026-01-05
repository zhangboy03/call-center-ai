import json
import logging
import os
import pickle
import time
import copy
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import dashscope
import faiss
import numpy as np
from dashscope import TextEmbedding

logger = logging.getLogger(__name__)


class RAGService:
    FILLERS = [
        "请问",
        "想问",
        "我想问",
        "麻烦",
        "不好意思",
        "那个",
        "就是",
        "嗯",
        "啊",
        "呃",
        "呢",
        "吧",
        "您好",
        "你好",
    ]

    def __init__(
        self,
        knowledge_base_path: str = "app/resources/knowledge_base.json",
        index_path: str = "app/resources/faiss_index.bin",
        metadata_path: str = "app/resources/metadata.pkl",
    ):
        self.kb_path = Path(knowledge_base_path)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = None
        self.documents = []  # List of dicts (the knowledge base)
        self.embeddings = None  # Numpy array (optional, if we want to rebuild)
        self._cache = OrderedDict()
        self._cache_size = 128

        # Ensure DashScope API Key is set
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = api_key

    def load_or_build_index(self):
        """Loads index from disk or builds it if missing."""
        if not self.kb_path.exists():
            logger.warning(f"Knowledge base not found at {self.kb_path}")
            return

        # Load Raw Data
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Try Loading Index
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                logger.info("Loading FAISS index from disk...")
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "rb") as f:
                    cached_docs = pickle.load(f)
                    # Simple validation: Count matches
                    if len(cached_docs) == len(self.documents):
                        logger.info("Index loaded successfully.")
                        return
                    else:
                        logger.warning("Index size mismatch with KB. Rebuilding...")
            except Exception as e:
                logger.error(f"Failed to load index: {e}. Rebuilding...")

        # Build Index
        self.build_index()

    def clean_query(self, text: str) -> str:
        """
        Remove common fillers/politeness and normalize spacing to improve retrieval recall.
        """
        if not text:
            return ""

        cleaned = text.strip()
        for filler in self.FILLERS:
            cleaned = cleaned.replace(filler, " ")

        # Collapse whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _get_cached_result(self, key):
        if key in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            return copy.deepcopy(self._cache[key])
        return None

    def _set_cached_result(self, key, value):
        self._cache[key] = copy.deepcopy(value)
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Batch generate embeddings using Aliyun text-embedding-v3.
        Note: DashScope batch size limit is usually 25.
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = []
        batch_size = 5

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v3,
                    input=batch,
                    dimension=1024,  # Explicit dimension
                )
                if resp.status_code == 200:
                    # DashScope SDK response might be an object or dict depending on version
                    output = resp.output
                    if isinstance(output, dict):
                        items = output.get("embeddings", [])
                        for item in items:
                            # Item can also be dict or object
                            if isinstance(item, dict):
                                embeddings.append(item.get("embedding"))
                            else:
                                embeddings.append(item.embedding)
                    else:
                        for item in output.embeddings:
                            embeddings.append(item.embedding)
                else:
                    logger.error(f"Embedding API error: {resp.message}")
            except Exception as e:
                logger.error(f"Embedding Exception: {e}")
                # Fallback: append zeros or handle error
                # For critical build process, strictly raising might be better,
                # but here we skip to avoid crashing
                pass

            # Rate limit politeness
            time.sleep(0.1)

        return np.array(embeddings, dtype="float32")

    def build_index(self):
        """Generates embeddings and builds FAISS index."""
        logger.info("Building new Vector Index...")

        # Prepare text to embed: "Topic: Question" usually works well
        texts_to_embed = [
            f"{doc['topic']}: {doc['question']}" for doc in self.documents
        ]

        if not texts_to_embed:
            logger.warning("No texts to embed.")
            return

        emb_matrix = self.get_embeddings(texts_to_embed)

        if len(emb_matrix) == 0:
            logger.error("Failed to generate embeddings. Index not built.")
            return

        # Normalize for Cosine Similarity
        faiss.normalize_L2(emb_matrix)

        # Build Index (FlatIP = Inner Product, acts as Cosine Sim with normalized vectors)
        dimension = emb_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(emb_matrix)

        # Save to disk
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Index built and saved with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 3, threshold: float = 0.4) -> List[Dict]:
        """
        Search for query.
        Returns list of results with 'score' and 'doc'.
        Threshold: Min similarity score (0-1).
        """
        query = self.clean_query(query)
        if not query:
            return []

        cache_key = (query, top_k, round(threshold, 3))
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            logger.info("[RAG] Cache hit for query: %s", query)
            return cached

        if not self.index:
            self.load_or_build_index()
            if not self.index:
                return []

        # Embed query
        query_vec = self.get_embeddings([query])
        if len(query_vec) == 0:
            return []

        faiss.normalize_L2(query_vec)

        # Search
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            if score < threshold:
                continue

            doc = self.documents[idx]
            results.append(
                {
                    "score": float(score),
                    "answer": doc["answer"],
                    "question": doc["question"],
                    "topic": doc["topic"],
                    "id": doc["id"],
                }
            )

        if results:
            self._set_cached_result(cache_key, results)
        else:
            logger.info("[RAG] No results above threshold for query: %s", query)

        return results


# Singleton instance
rag_service = RAGService()
