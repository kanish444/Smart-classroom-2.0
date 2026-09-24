import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from loguru import logger
from config.settings import get_settings


class FaissVectorStore:
    """
    FAISS-based High-Performance Vector Storage and Nearest Neighbor Search Engine.
    Uses IndexFlatIP wrapped in IndexIDMap2 for exact cosine similarity (Inner Product on L2-normalized vectors).
    Maintains synchronized mapping between FAISS vector IDs and SQLite student/embedding records.
    """

    def __init__(self, embedding_dim: int = 512, index_path: Optional[str] = None):
        self.embedding_dim = embedding_dim
        self.settings = get_settings().recognition
        self.index_path = index_path or self.settings.index_path
        self.meta_path = f"{self.index_path}.meta.json"

        # Initialize IndexIDMap2 wrapping IndexFlatIP
        self._init_empty_index()

        # In-memory mapping: vector_id (int) -> metadata dict (student_id, student_name, etc.)
        self.id_to_metadata: Dict[int, Dict[str, Any]] = {}

        # Load existing index and metadata if present
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.load_index(self.index_path)
            except Exception as e:
                logger.warning(f"Could not load existing FAISS index: {e}. Starting fresh.")
                self._init_empty_index()
                self.id_to_metadata = {}

    def _init_empty_index(self):
        """Initializes a new empty FAISS IndexIDMap2."""
        sub_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIDMap2(sub_index)

    @property
    def total_vectors(self) -> int:
        """Returns the number of indexed vectors."""
        return int(self.index.ntotal)

    def add_vector(
        self,
        embedding_id: Optional[int] = None,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[int] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Adds a single L2-normalized embedding vector with its corresponding database embedding_id.
        Supports both positional and keyword forms (embedding_id/vector_id, vector/embedding).
        """
        emb_id = embedding_id if embedding_id is not None else vector_id
        if emb_id is None:
            raise ValueError("embedding_id or vector_id must be provided.")
        vec = vector if vector is not None else embedding
        if vec is None or not isinstance(vec, np.ndarray):
            raise ValueError("Vector must be a non-null numpy.ndarray.")

        if vec.size != self.embedding_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.embedding_dim}, got {vec.size}.")

        # Ensure float32 2D array (1, dim)
        vec_2d = np.ascontiguousarray(vec.reshape(1, self.embedding_dim), dtype=np.float32)
        
        # Verify L2 normalization
        norm = float(np.linalg.norm(vec_2d))
        if abs(norm - 1.0) > 1e-3:
            # Normalize if not perfectly unit norm
            vec_2d = vec_2d / max(norm, 1e-12)

        ids = np.array([emb_id], dtype=np.int64)
        self.index.add_with_ids(vec_2d, ids)
        self.id_to_metadata[int(emb_id)] = metadata or {}
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Searches top_k nearest neighbors using Cosine Similarity (Inner Product).
        
        Returns:
            List of (similarity_score, metadata) sorted descending by similarity.
            If the index is empty, returns an empty list safely without crashing.
        """
        if self.total_vectors == 0:
            return []

        if query_vector is None or not isinstance(query_vector, np.ndarray):
            raise ValueError("Query vector must be a non-null numpy.ndarray.")

        if query_vector.size != self.embedding_dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.embedding_dim}, got {query_vector.size}.")

        # Ensure 2D float32
        q_vec = np.ascontiguousarray(query_vector.reshape(1, self.embedding_dim), dtype=np.float32)

        # Ensure query is L2-normalized
        norm = float(np.linalg.norm(q_vec))
        if abs(norm - 1.0) > 1e-3:
            q_vec = q_vec / max(norm, 1e-12)

        k = min(top_k, self.total_vectors)
        distances, indices = self.index.search(q_vec, k)

        results = []
        for sim, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.id_to_metadata.get(int(idx), {"embedding_id": int(idx)})
            results.append((float(sim), meta))

        return results

    def save_index(self, path: Optional[str] = None):
        """Persists the FAISS index and associated metadata mapping to disk."""
        target_path = path or self.index_path
        target_meta = f"{target_path}.meta.json"

        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        faiss.write_index(self.index, target_path)

        with open(target_meta, "w", encoding="utf-8") as f:
            json.dump(self.id_to_metadata, f, indent=2)

        logger.info(f"Saved FAISS index ({self.total_vectors} vectors) to {target_path}")

    def load_index(self, path: Optional[str] = None):
        """Loads the FAISS index and associated metadata mapping from disk."""
        target_path = path or self.index_path
        target_meta = f"{target_path}.meta.json"

        if not os.path.exists(target_path) or not os.path.exists(target_meta):
            raise FileNotFoundError(f"FAISS index or metadata file not found at '{target_path}'.")

        self.index = faiss.read_index(target_path)
        with open(target_meta, "r", encoding="utf-8") as f:
            # JSON keys are strings, convert back to int keys
            raw_meta = json.load(f)
            self.id_to_metadata = {int(k): v for k, v in raw_meta.items()}

        logger.info(f"Loaded FAISS index ({self.total_vectors} vectors) from {target_path}")

    def clear(self):
        """Resets the FAISS index and clears all metadata mappings."""
        self._init_empty_index()
        self.id_to_metadata.clear()

    def sync_with_database(self, all_embeddings: List[Dict[str, Any]]):
        """
        Rebuilds the FAISS index from the provided SQLite embedding records to guarantee 100% synchronization.
        """
        self.clear()
        for rec in all_embeddings:
            emb_id = rec["embedding_id"]
            vec = rec["vector"]
            meta = {
                "embedding_id": emb_id,
                "student_id": rec["student_id"],
                "student_name": rec.get("student_name", "Unknown"),
                "sample_label": rec.get("sample_label", "frontal"),
                "quality_score": rec.get("quality_score", 1.0)
            }
            self.add_vector(emb_id, vec, meta)

        logger.info(f"FAISS vector store synchronized with SQLite: {self.total_vectors} vectors indexed.")

    def validate_consistency(self, expected_db_count: int) -> Tuple[bool, str]:
        """
        Verifies that FAISS vector count matches database embedding count
        and every indexed vector has corresponding identity metadata.
        """
        if self.total_vectors != expected_db_count:
            msg = f"FAISS / Database Mismatch: FAISS has {self.total_vectors} vectors, but SQLite has {expected_db_count} records."
            return False, msg

        if len(self.id_to_metadata) != self.total_vectors:
            msg = f"Metadata Mismatch: FAISS has {self.total_vectors} vectors, but metadata store has {len(self.id_to_metadata)} records."
            return False, msg

        return True, "Consistent"
