import torch
import numpy as np
from PIL import Image
import io
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
from loguru import logger

try:
    from config import JINA_MODEL_PATH, USE_CPU
except ImportError:
    JINA_MODEL_PATH = "jinaai/jina-embeddings-v4"
    USE_CPU = False
    logger.warning("config.py not found, using default model path and GPU mode")


class JinaRetriever():
    def __init__(self, bs=128, use_gpu=None, vector_type="dense"):
        self.bs = bs
        self.bs_query = 8
        self.vector_type = vector_type
        self.model_name = JINA_MODEL_PATH

        if use_gpu is None:
            use_gpu = not USE_CPU

        device = "cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu"
        self.device = torch.device(device)

        if device == "cpu":
            logger.info("CPU mode: loading model with float32 precision")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

        self.model = self.model.to(device)
        self.model.eval()
        
        # Note: Don't use DataParallel for Jina v4 to avoid method access issues
        # The official example doesn't use DataParallel, let's follow that pattern
        if torch.cuda.device_count() > 1 and use_gpu:
            logger.info(f"Multiple GPUs detected ({torch.cuda.device_count()}), using single GPU: {device}")

        logger.info(f"Jina v4 loaded with '{self.vector_type}' vector type")

    def embed_queries(self, queries, pad=False):
        """
        Embed queries using Jina v4
        
        Args:
            queries: string or list of strings
            pad: whether to pad embeddings (kept for compatibility)
            
        Returns:
            List of embeddings as numpy arrays
        """
        if isinstance(queries, str):
            queries = [queries]
            
        embeddings = []
        
        # Process queries in batches
        for i in range(0, len(queries), self.bs_query):
        
            batch_queries = queries[i:i+self.bs_query]
            
            with torch.no_grad():
                if self.vector_type == "dense":
                    # Dense single-vector embeddings
                    batch_embeddings = self.model.encode_text(
                        texts=batch_queries,
                        task="retrieval",
                        prompt_name="query",
                        return_multivector=False,
                        
                    )

                else:  # multi-vector
                    # Multi-vector embeddings
                    batch_embeddings = self.model.encode_text(
                        texts=batch_queries,
                        task="retrieval", 
                        prompt_name="query",
                        return_multivector=True,
                        
                    )

                # Convert to numpy arrays and append to embeddings list
                if isinstance(batch_embeddings, torch.Tensor):
                    # Move to CPU and convert to numpy
                    batch_embeddings_np = batch_embeddings.cpu().float().numpy()
                    if batch_embeddings_np.ndim == 2:  # [batch_size, dim]
                        for emb in batch_embeddings_np:
                            embeddings.append(emb.copy())  # Use copy() for safety
                    else:  # multi-vector case [batch_size, seq_len, dim]
                        for emb in batch_embeddings_np:
                            embeddings.append(emb.copy())
                elif isinstance(batch_embeddings, np.ndarray):
                    if batch_embeddings.ndim == 2:
                        for emb in batch_embeddings:
                            embeddings.append(emb.astype(np.float32))
                    else:
                        for emb in batch_embeddings:
                            embeddings.append(emb.astype(np.float32))
                else:  # list or other iterable
                    for emb in batch_embeddings:
                        if isinstance(emb, torch.Tensor):
                            embeddings.append(emb.cpu().float().numpy().astype(np.float32))
                        elif isinstance(emb, np.ndarray):
                            embeddings.append(emb.astype(np.float32))
                        else:
                            embeddings.append(np.array(emb, dtype=np.float32))
        
        return embeddings

    def embed_quotes(self, images, hybrid=False):
        """
        Embed quotes/documents using Jina v4 with OOM retry mechanism
        """
        import gc
        
        embeddings = []
        
        if not hybrid:
            # Image mode
            if isinstance(images, (Image.Image, bytes, bytearray, str)):
                images = [images]
                
            for i in range(0, len(images), self.bs):
                batch_images = images[i:i+self.bs]
                
                # Convert to proper format
                processed_images = []
                for img in batch_images:
                    if isinstance(img, Image.Image):
                        processed_images.append(img)
                    elif isinstance(img, (bytes, bytearray)):
                        pil_img = Image.open(io.BytesIO(img))
                        processed_images.append(pil_img.convert("RGB"))
                    elif isinstance(img, str):
                        processed_images.append(img)
                    else:
                        raise ValueError("Each image must be a PIL.Image.Image, bytes, or string path/URL.")
                
                batch_embeddings = self._encode_with_retry(processed_images, hybrid=False)
                embeddings.extend(batch_embeddings)
                        
        else:
            # Text mode
            if isinstance(images, str):
                images = [images]
                
            for i in range(0, len(images), self.bs):
                batch_texts = images[i:i+self.bs]
                
                # Truncate long passages
                def truncate_passages(passages, max_words=400):
                    truncated = []
                    for passage in passages:
                        words = passage.split()
                        if len(words) > max_words:
                            truncated_passage = ' '.join(words[:max_words])
                        else:
                            truncated_passage = passage
                        truncated.append(truncated_passage)
                    return truncated
                
                batch_texts = truncate_passages(batch_texts)
                
                batch_embeddings = self._encode_with_retry(batch_texts, hybrid=True)
                embeddings.extend(batch_embeddings)
                            
        return embeddings

    def _encode_with_retry(self, batch_data, hybrid=False, max_retries=3):
        """Encode with OOM retry: halves batch size on GPU memory errors."""
        import gc
        
        current_bs = len(batch_data)
        retry_count = 0
        
        while current_bs > 0 and retry_count < max_retries:
            try:
                current_batch = batch_data[:current_bs]
                
                with torch.no_grad():
                    if hybrid:  # Text
                        if self.vector_type == "dense":
                            batch_embeddings = self.model.encode_text(
                                texts=current_batch,
                                task="retrieval",
                                prompt_name="passage",
                                return_multivector=False,
                                batch_size=len(current_batch)
                            )
                        else:
                            batch_embeddings = self.model.encode_text(
                                texts=current_batch,
                                task="retrieval",
                                prompt_name="passage",
                                return_multivector=True,
                                batch_size=len(current_batch)
                            )
                    else:  # Image
                        if self.vector_type == "dense":
                            batch_embeddings = self.model.encode_image(
                                images=current_batch,
                                task="retrieval",
                                return_multivector=False,
                            )
                        else:
                            batch_embeddings = self.model.encode_image(
                                images=current_batch,
                                task="retrieval",
                                return_multivector=True,
                            )
                
                result_embeddings = []
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().float().numpy()
                
                if hasattr(batch_embeddings, 'ndim') and batch_embeddings.ndim == 2:
                    for emb in batch_embeddings:
                        result_embeddings.append(emb)
                else:
                    for emb in batch_embeddings:
                        result_embeddings.append(emb)
                
                if current_bs < len(batch_data):
                    remaining_data = batch_data[current_bs:]
                    remaining_embeddings = self._encode_with_retry(remaining_data, hybrid, max_retries)
                    result_embeddings.extend(remaining_embeddings)
                
                return result_embeddings
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    retry_count += 1
                    old_bs = current_bs
                    current_bs = max(1, current_bs // 4)
                    
                    logger.warning(f"OOM occurred with batch size {old_bs}, retrying with {current_bs} (retry {retry_count}/{max_retries})")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    if current_bs == 1 and retry_count >= max_retries:
                        logger.error("Even batch size 1 failed after max retries, skipping this batch")
                        return []
                else:
                    logger.error(f"Non-OOM error: {e}")
                    raise e
            except Exception as e:
                logger.error(f"Unexpected error in encoding: {e}")
                raise e
        
        return []

    def score(self, query_embs, image_embs):
        """
        Calculate similarity scores between query and document embeddings
        
        Args:
            query_embs: List of query embeddings
            image_embs: List of document/image embeddings
            
        Returns:
            2D numpy array of similarity scores
        """
        if self.vector_type == "dense":
            # Dense vector scoring - cosine similarity
            query_emb = np.asarray(query_embs)
            quote_emb = np.asarray(image_embs)
            
            # Normalize embeddings for cosine similarity
            query_emb_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            quote_emb_norm = quote_emb / np.linalg.norm(quote_emb, axis=1, keepdims=True)
            
            scores = np.dot(query_emb_norm, quote_emb_norm.T)
            return scores
            
        else:
            # Multi-vector scoring (late interaction like ColBERT)
            # Following the same pattern as ColPali's score method
            qs = [torch.from_numpy(e) if isinstance(e, np.ndarray) else e for e in query_embs]
            ds = [torch.from_numpy(e) if isinstance(e, np.ndarray) else e for e in image_embs]
            
            scores = np.zeros((len(qs), len(ds)), dtype=np.float32)
            for i, q in enumerate(qs):
                q = q.float()  # [Lq, d]
                for j, d in enumerate(ds):
                    d = d.float()  # [Ld, d]
                    sim = torch.matmul(q, d.T)    # [Lq, Ld]
                    maxsim = torch.max(sim, dim=1)[0].sum().item()  # colbert-style: sum-of-max over query tokens
                    scores[i, j] = maxsim
                    
            return scores

# Convenience classes for specific modes
class JinaDenseRetriever(JinaRetriever):
    """Jina v4 Dense Vector Retriever"""
    def __init__(self, bs=4, use_gpu=None):
        super().__init__(bs=bs, use_gpu=use_gpu, vector_type="dense")


class JinaMultiRetriever(JinaRetriever):
    """Jina v4 Multi-Vector Retriever (Late Interaction)"""
    def __init__(self, bs=4, use_gpu=None):
        super().__init__(bs=bs, use_gpu=use_gpu, vector_type="multi")
