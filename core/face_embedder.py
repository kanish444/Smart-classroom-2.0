import os
import time
from typing import Optional, Tuple
import numpy as np
import onnxruntime as ort
from loguru import logger
from config.settings import get_settings


class ArcFaceEmbedder:
    """
    ArcFace (MobileFaceNet) Face Embedding Generator.
    Uses ONNX Runtime CPU inference on 112x112 aligned, normalized face tensors.
    Produces 512-dimensional L2-normalized embedding vectors.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.settings = get_settings().recognition
        
        # Resolve model file path
        if model_path:
            self.model_path = model_path
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "models", "weights", self.settings.model_name)
            if not os.path.exists(self.model_path):
                # Fallback to root or models directory if present
                alt_path = os.path.join(base_dir, self.settings.model_name)
                if os.path.exists(alt_path):
                    self.model_path = alt_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ArcFace model file not found at '{self.model_path}'. "
                f"Ensure 'w600k_mbf.onnx' is present in models/weights/."
            )

        logger.info(f"Loading ArcFace MobileFaceNet from: {self.model_path}")
        t0 = time.perf_counter()

        # Initialize ONNX Runtime Session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0

        # Inspect Input and Output Signatures
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        self.embedding_dim = self.output_shape[1] if len(self.output_shape) > 1 else self.settings.embedding_dim

        logger.info(
            f"ArcFace model loaded in {self.load_time_ms:.2f} ms. "
            f"Input: '{self.input_name}' {self.input_shape}, Output: '{self.output_name}' {self.output_shape}"
        )

        # Warmup inference
        dummy_tensor = np.zeros((1, 3, 112, 112), dtype=np.float32)
        _ = self.session.run([self.output_name], {self.input_name: dummy_tensor})
        logger.info("ArcFace model warmup complete.")

    def validate_input_tensor(self, face_tensor: np.ndarray) -> np.ndarray:
        """
        Validates the incoming tensor shape, data type, and values.
        Accepts (1, 3, 112, 112) or (3, 112, 112).
        """
        if face_tensor is None or not isinstance(face_tensor, np.ndarray):
            raise ValueError("Input face tensor must be a non-null numpy.ndarray.")

        if face_tensor.size == 0:
            raise ValueError("Input face tensor cannot be empty.")

        # Ensure 4D shape (1, 3, 112, 112)
        if face_tensor.ndim == 3:
            if face_tensor.shape != (3, 112, 112):
                raise ValueError(f"Expected 3D tensor shape (3, 112, 112), got {face_tensor.shape}.")
            face_tensor = np.expand_dims(face_tensor, axis=0)
        elif face_tensor.ndim == 4:
            if face_tensor.shape[1:] != (3, 112, 112):
                raise ValueError(f"Expected 4D tensor shape (B, 3, 112, 112), got {face_tensor.shape}.")
        else:
            raise ValueError(f"Invalid tensor dimensions: {face_tensor.ndim}D. Expected 3D or 4D.")

        if face_tensor.dtype != np.float32:
            face_tensor = face_tensor.astype(np.float32)

        if np.isnan(face_tensor).any() or np.isinf(face_tensor).any():
            raise ValueError("Input face tensor contains NaN or Inf values.")

        return face_tensor

    def normalize_embedding(self, raw_embedding: np.ndarray) -> np.ndarray:
        """
        Applies L2 normalization: v_norm = v / ||v||_2
        Guarantees unit vector representation for exact cosine similarity via dot product.
        """
        norm = np.linalg.norm(raw_embedding, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-12)  # Prevent division by zero
        normalized = raw_embedding / norm
        return normalized.astype(np.float32)

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validates numeric validity, dimension, and unit L2 norm of generated embedding.
        """
        if embedding is None or not isinstance(embedding, np.ndarray):
            return False

        if embedding.size != self.embedding_dim:
            return False

        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return False

        # Verify unit norm within tolerance
        norm = float(np.linalg.norm(embedding))
        if abs(norm - 1.0) > 1e-3:
            return False

        return True

    def generate_embedding(self, face_tensor: np.ndarray) -> np.ndarray:
        """
        Runs forward inference on a normalized (1, 3, 112, 112) face tensor
        and produces a 512-dim L2-normalized float32 embedding vector.
        
        Returns:
            np.ndarray of shape (1, 512), dtype float32, unit L2 norm.
        """
        validated_tensor = self.validate_input_tensor(face_tensor)

        try:
            raw_out = self.session.run([self.output_name], {self.input_name: validated_tensor})[0]
        except Exception as e:
            logger.error(f"ArcFace ONNX inference failed: {e}")
            raise RuntimeError(f"ArcFace model inference failure: {e}")

        if raw_out is None or raw_out.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Unexpected raw output shape from ArcFace: {raw_out.shape if raw_out is not None else None}. "
                f"Expected last dimension = {self.embedding_dim}."
            )

        # Apply L2 normalization
        normalized_embedding = self.normalize_embedding(raw_out)

        # Validate final embedding
        if not self.validate_embedding(normalized_embedding):
            raise ValueError("Generated face embedding failed post-inference numerical validation.")

        return normalized_embedding
