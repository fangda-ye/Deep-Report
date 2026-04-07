import os
# Import GPU config
try:
    from config import CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
except ImportError:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import re
import hashlib
import base64
from loguru import logger
from pathlib import Path
import traceback
from tqdm import tqdm
import threading

from flask import Flask, request, jsonify
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import math

# ============== Configuration ==============
try:
    from config import (
        MILVUS_PATH,
        MAX_CHUNK_LENGTH, DEFAULT_TEXT_BATCH_SIZE, DEFAULT_IMAGE_BATCH_SIZE,
        FLASK_HOST, FLASK_PORT, FLASK_DEBUG, LOG_FILE, LOG_ROTATION, USE_CPU,
        IMAGE_BASE_DIR, ORIGINAL_IMAGE_PREFIXES
    )
except ImportError:
    logger.warning("config.py not found, using defaults")
    MILVUS_PATH = "data/sandbox/milvus_MMLF.db"
    MAX_CHUNK_LENGTH = 350
    DEFAULT_TEXT_BATCH_SIZE = 16
    DEFAULT_IMAGE_BATCH_SIZE = 16
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5555
    FLASK_DEBUG = False
    LOG_FILE = "logs/retriever.log"
    LOG_ROTATION = "10 MB"
    USE_CPU = False

# ============== Concurrency Control ==============
model_lock = threading.Lock()

# ============== Markdown Processing ==============

def clean_links(text):
    """Remove HTTP/HTTPS links from text."""
    try:
        text = re.sub(r'https?://[^\s\)]+', '', text)
        return text
    except Exception as e:
        logger.warning(f"Failed to clean links: {e}")
        return text

def split_by_headers(content):
    """Split markdown content by headers."""
    try:
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_header = ""
        
        for line in lines:
            try:
                if re.match(r'^#{1,6}\s', line):
                    if current_chunk:
                        chunks.append({
                            'header': current_header,
                            'lines': current_chunk
                        })
                    current_header = line
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            except Exception as e:
                logger.warning(f"Error processing line: {e}, skipping")
                continue
        
        if current_chunk:
            chunks.append({
                'header': current_header,
                'lines': current_chunk
            })
        
        return chunks
    except Exception as e:
        logger.error(f"Failed to split by headers: {e}")
        try:
            return [{'header': '', 'lines': content.split('\n')}]
        except:
            return [{'header': '', 'lines': [content]}]

def merge_lines_to_chunks(chunks, max_length=MAX_CHUNK_LENGTH):
    """Merge lines into chunks up to max_length without splitting single lines."""
    final_chunks = []
    
    if not chunks:
        logger.warning("Empty chunks, returning empty list")
        return final_chunks
    
    for chunk_group in chunks:
        try:
            header = chunk_group.get('header', '')
            lines = chunk_group.get('lines', [])
            
            current_text = ""
            for line in lines:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if current_text and len(current_text) + len(line) + 1 > max_length:
                        final_chunks.append(current_text)
                        current_text = line
                    else:
                        if current_text:
                            current_text += "\n" + line
                        else:
                            current_text = line
                except Exception as e:
                    logger.warning(f"Error merging line: {e}, skipping")
                    continue
            
            if current_text:
                final_chunks.append(current_text)
        except Exception as e:
            logger.warning(f"Error processing chunk group: {e}, skipping")
            continue
    
    return final_chunks

def process_md_file(md_path):
    """Process a single markdown file into text chunks."""
    try:
        if not os.path.exists(md_path):
            logger.warning(f"File not found: {md_path}")
            return []
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"File is empty: {md_path}")
            return []
        
        content = clean_links(content)
        header_chunks = split_by_headers(content)
        text_chunks = merge_lines_to_chunks(header_chunks, MAX_CHUNK_LENGTH)
        
        return text_chunks
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error {md_path}: {e}")
        try:
            with open(md_path, 'r', encoding='gbk') as f:
                content = f.read()
            content = clean_links(content)
            header_chunks = split_by_headers(content)
            text_chunks = merge_lines_to_chunks(header_chunks, MAX_CHUNK_LENGTH)
            return text_chunks
        except Exception as e2:
            logger.error(f"GBK fallback also failed {md_path}: {e2}")
            return []
    except PermissionError as e:
        logger.error(f"Permission error {md_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to process file {md_path}: {e}")
        logger.error(traceback.format_exc())
        return []

# ============== Milvus Operations ==============

class MilvusDB:
    @staticmethod
    def create_collection(collection_name, dim):
        """Create a Milvus collection with the given vector dimension."""
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="img_path", dtype=DataType.VARCHAR, max_length=1024),
            ]
            
            schema = CollectionSchema(fields=fields, description="MD document collection")
            client = MilvusClient(MILVUS_PATH)
            client.create_collection(collection_name=collection_name, schema=schema)
            
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="IP",
                index_type="IVF_FLAT",
                index_name="vector_index",
                params={"nlist": 128}
            )
            client.create_index(collection_name=collection_name, index_params=index_params, sync=False)
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    @staticmethod
    def insert_data(collection_name, entries, dim, batch_size=2000):
        """Insert data into Milvus in batches."""
        if not entries:
            logger.warning("No data to insert")
            return 0
        
        try:
            client = MilvusClient(MILVUS_PATH)
            
            if not client.has_collection(collection_name):
                MilvusDB.create_collection(collection_name, dim)
            
            total_inserted = 0
            total_entries = len(entries)
            
            for i in range(0, total_entries, batch_size):
                try:
                    batch_entries = entries[i:i + batch_size]
                    
                    to_insert = []
                    duplicates = 0
                    
                    batch_ids = [entry['id'] for entry in batch_entries]
                    
                    existing_ids = set()
                    query_batch_size = 100
                    
                    for j in range(0, len(batch_ids), query_batch_size):
                        try:
                            query_ids = batch_ids[j:j + query_batch_size]
                            id_list_str = str(query_ids).replace("'", '"')
                            id_filter = f'id in {id_list_str}'
                            
                            existing = client.query(
                                collection_name=collection_name,
                                filter=id_filter,
                                output_fields=["id"],
                                limit=len(query_ids)
                            )
                            existing_ids.update([item['id'] for item in existing])

                        except Exception as e:
                            logger.warning(f"Duplicate check query failed, skipping: {e}")
                            break
                    
                    for entry in batch_entries:
                        try:
                            if entry['id'] in existing_ids:
                                duplicates += 1
                                continue
                            to_insert.append(entry)
                        except Exception as e:
                            logger.warning(f"Error processing entry: {e}, skipping")
                            continue
                    
                    if to_insert:
                        try:
                            client.insert(collection_name=collection_name, data=to_insert)
                            batch_inserted = len(to_insert)
                            total_inserted += batch_inserted
                        except Exception as e:
                            logger.error(f"Batch {i//batch_size + 1} insert failed: {e}")
                            single_inserted = 0
                            for single_entry in to_insert:
                                try:
                                    client.insert(collection_name=collection_name, data=[single_entry])
                                    single_inserted += 1
                                except Exception as se:
                                    continue
                            if single_inserted > 0:
                                total_inserted += single_inserted
                    
                    if duplicates > 0:
                        logger.info(f"Batch {i//batch_size + 1}: skipped {duplicates} duplicates")
                        continue
                
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            return total_inserted
            
        except Exception as e:
            logger.error(f"Insert data failed: {e}")
            logger.error(traceback.format_exc())
            return 0

    @staticmethod
    def search(collection_name, embeddings, topk, data_type):
        """Search vectors in the collection."""
        try:
            client = MilvusClient(MILVUS_PATH)
            
            if not client.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            filter_expr = f'type == "{data_type}"'
            
            results = client.search(
                collection_name=collection_name,
                data=embeddings,
                anns_field="vector",
                limit=topk,
                output_fields=["id", "type", "text", "img_path"],
                filter=filter_expr
            )
            
            all_hits = []
            for result_set in results:
                for hit in result_set:
                    try:
                        entity = hit.get('entity', {})
                        all_hits.append({
                            "id": entity.get("id"),
                            "type": entity.get("type"),
                            "text": entity.get("text"),
                            "img_path": entity.get("img_path"),
                            "score": hit.get('distance', 0)
                        })
                    except Exception as e:
                        logger.warning(f"Error processing search result: {e}")
                        continue
            
            return all_hits
        except Exception as e:
            logger.error(f"Search failed: {e}")
            logger.error(traceback.format_exc())
            return []

# ============== Data Processing ==============

def get_content_hash(content):
    """Generate an MD5 hash of content as a unique ID."""
    try:
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Hash generation failed: {e}")
        try:
            return hashlib.md5(str(content).encode('utf-8', errors='ignore')).hexdigest()
        except:
            return hashlib.md5(f"fallback_{id(content)}".encode()).hexdigest()

def process_and_insert_batch(texts, entries, jina_model, collection_name):
    """Embed a batch of texts and insert into the collection."""
    try:
        embeddings = jina_model.embed_quotes(texts, hybrid=True)
        
        for i, emb in enumerate(embeddings):
            entries[i]["vector"] = emb.tolist()
        
        if entries:
            dim = len(entries[0]['vector'])
            inserted = MilvusDB.insert_data(collection_name, entries, dim, batch_size=len(entries))
            return inserted
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 0

def process_text_folder_optimized(folder_path, jina_model, batch_size=DEFAULT_TEXT_BATCH_SIZE):
    """Process text files in a folder: chunk, embed, and insert in batches."""

    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return 0
    
    md_files = list(folder.rglob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files")
    
    logger.info("Preprocessing files, counting chunks...")
    all_chunks = []
    for md_file in tqdm(md_files, desc="Preprocessing files"):
        try:
            chunks = process_md_file(md_file)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {md_file}: {e}")
            continue
    
    total_chunks = len(all_chunks)
    logger.info(f"{folder_path}: {total_chunks} text chunks to process")
    
    if total_chunks == 0:
        logger.warning("No valid text chunks found")
        return 0
    
    total_inserted = 0
    text_batch = []
    entry_batch = []
    
    with tqdm(total=total_chunks, desc="Processing text chunks") as pbar:
        for chunk in all_chunks:
            try:
                chunk_id = get_content_hash(chunk)
                entry = {
                    "id": chunk_id,
                    "type": "text",
                    "text": chunk,
                    "img_path": ""
                }
                
                text_batch.append(chunk)
                entry_batch.append(entry)
                
                if len(text_batch) >= batch_size:
                    inserted = process_and_insert_batch(text_batch, entry_batch, jina_model, "text_collection")
                    total_inserted += inserted

                    pbar.update(len(text_batch))
                    pbar.set_postfix({
                        'inserted': total_inserted,
                        'batch_size': len(text_batch)
                    })
                    
                    text_batch = []
                    entry_batch = []
                    
            except Exception as e:
                logger.error(f"Failed to process text chunk: {e}")
                pbar.update(1)
                continue
        
        if text_batch:
            inserted = process_and_insert_batch(text_batch, entry_batch, jina_model, "text_collection")
            total_inserted += inserted
            pbar.update(len(text_batch))
            pbar.set_postfix({
                'inserted': total_inserted,
                'batch_size': len(text_batch)
            })
    
    return total_inserted

def process_image_jsonl_optimized(jsonl_path, jina_model, batch_size=DEFAULT_IMAGE_BATCH_SIZE):
    """Process images from JSONL: embed and insert in batches."""

    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL file not found: {jsonl_path}")
        return 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    logger.info(f"Found {total_lines} image records")
    
    total_inserted = 0
    image_batch = []
    entry_batch = []
    MAX_PIXELS = 4_000_000
    processed_count = 0
    MIN_SIZE = 28
    
    with tqdm(total=total_lines, desc="Processing images") as pbar:
        for line_num, line in enumerate(lines):
            try:
                data = json.loads(line)
                img_path = data.get('source_image_path', '')
                description = data.get('description', '')
                
                pbar.update(1)
                pbar.set_postfix({
                    'processed': processed_count,
                    'inserted': total_inserted,
                    'batch_size': len(image_batch)
                })
                
                if not os.path.exists(img_path):
                    continue
                
                try:
                    if os.path.getsize(img_path) == 0:
                        continue
                    
                    file_size = os.path.getsize(img_path)
                    if file_size < 10 * 1024:
                        logger.warning(f"Image too small {img_path}: {file_size} bytes (< 10KB)")
                        continue

                    img = Image.open(img_path)
                    w, h = img.size
                    total_pixels = w * h

                    if w < MIN_SIZE or h < MIN_SIZE:
                        logger.warning(f"Image dimensions too small {img_path}: {w}x{h}, min {MIN_SIZE}x{MIN_SIZE}")
                        continue

                    if total_pixels > MAX_PIXELS:
                        scale = math.sqrt(MAX_PIXELS / total_pixels)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    img_id = get_content_hash(description + img_path)
                    entry = {
                        "id": img_id,
                        "type": "image", 
                        "text": description,
                        "img_path": img_path
                    }
                    
                    image_batch.append(img)
                    entry_batch.append(entry)
                    processed_count += 1
                    
                    if len(image_batch) >= batch_size:
                        inserted = process_and_insert_image_batch_pil(image_batch, entry_batch, jina_model, "image_collection")
                        total_inserted += inserted
                        
                        pbar.set_postfix({
                            'processed': processed_count,
                            'inserted': total_inserted,
                            'batch_size': len(image_batch)
                        })
                        
                        image_batch = []
                        entry_batch = []
                        
                except Exception as e:
                    logger.warning(f"Failed to process image {img_path}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to process line {line_num + 1}: {e}")
                continue
        
        if image_batch:
            inserted = process_and_insert_image_batch_pil(image_batch, entry_batch, jina_model, "image_collection")
            total_inserted += inserted
            pbar.set_postfix({
                'processed': processed_count,
                'inserted': total_inserted,
                'final_batch': len(image_batch)
            })
    
    return total_inserted

def process_and_insert_image_batch_pil(image_batch, entries, jina_model, collection_name):
    """Embed PIL Image objects and insert into the vector database."""
    try:
        embeddings = jina_model.embed_quotes(image_batch)

        for i, emb in enumerate(embeddings):
            entries[i]["vector"] = emb.tolist()

        if entries:
            dim = len(entries[0]['vector'])
            inserted = MilvusDB.insert_data(collection_name, entries, dim, batch_size=len(entries))
            return inserted
        
    except Exception as e:
        logger.error(f"Batch image processing failed: {e}")
        return 0

def rerank_results(hits, query_list, jina_model):
    """Rerank search results by score (descending)."""
    try:
        return sorted(hits, key=lambda x: x.get('score', 0), reverse=True)
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return hits

def remap_image_path(original_path):
    """Remap image paths stored in DB to local paths under IMAGE_BASE_DIR."""
    if not original_path:
        return original_path
    try:
        for prefix in ORIGINAL_IMAGE_PREFIXES:
            if original_path.startswith(prefix):
                relative = original_path[len(prefix):]
                remapped = os.path.join(IMAGE_BASE_DIR, relative)
                if os.path.exists(remapped):
                    return remapped
        # Fallback: try basename match under IMAGE_BASE_DIR
        basename = os.path.basename(original_path)
        for root, dirs, files in os.walk(IMAGE_BASE_DIR):
            if basename in files:
                return os.path.join(root, basename)
    except Exception as e:
        logger.warning(f"Path remap failed for {original_path}: {e}")
    return original_path


def format_results(hits):
    """Format search results for API response."""
    formatted = []

    for hit in hits:
        try:
            result = {
                "id": hit.get('id', ''),
                "type": hit.get('type', ''),
                "score": hit.get('score', 0)
            }

            if hit.get('type') == 'text':
                result['text'] = hit.get('text', '')
            elif hit.get('type') == 'image':
                result['description'] = hit.get('text', '')

                img_path = remap_image_path(hit.get('img_path', ''))
                result['img_path'] = img_path

                if os.path.exists(img_path):
                    try:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                            result['img_base64'] = base64.b64encode(img_bytes).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Failed to read image {img_path}: {e}")
                        result['img_base64'] = None
                else:
                    result['img_base64'] = None

            formatted.append(result)
        except Exception as e:
            logger.warning(f"Error formatting result: {e}")
            continue

    return formatted

# ============== Flask API ==============

app = Flask(__name__)

@app.route('/insert_text', methods=['POST'])
def insert_text():
    """API endpoint to insert text data."""
    try:
        data = request.json or {}
        folder_path = data.get('folder_path')
        batch_size = data.get('batch_size', DEFAULT_TEXT_BATCH_SIZE)

        logger.info(f"Processing text folder: {folder_path}, batch_size: {batch_size}")

        with model_lock:
            total_inserted = process_text_folder_optimized(folder_path, jina_model, batch_size)

        logger.info(f"Finished text folder: {folder_path}, inserted {total_inserted}")

        return jsonify({
            "status": "success",
            "inserted": total_inserted,
            "batch_size": batch_size
        })

    except Exception as e:
        logger.error(f"Text insertion failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/insert_image', methods=['POST'])
def insert_image():
    """API endpoint to insert image data."""
    try:
        data = request.json or {}
        jsonl_path = data.get('jsonl_path')
        batch_size = data.get('batch_size', DEFAULT_IMAGE_BATCH_SIZE)

        logger.info(f"Processing image JSONL: {jsonl_path}, batch_size: {batch_size}")

        with model_lock:
            total_inserted = process_image_jsonl_optimized(jsonl_path, jina_model, batch_size)

        return jsonify({
            "status": "success",
            "inserted": total_inserted,
            "batch_size": batch_size
        })

    except Exception as e:
        logger.error(f"Image insertion failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """API endpoint to search the vector database."""
    try:
        data = request.json or {}
        query_list = data.get('query_list', [])
        data_type = data.get('type', 'text')
        topk = data.get('topk', 10)

        if not query_list:
            return jsonify({"error": "query_list must not be empty"}), 400

        if data_type not in ['text', 'image']:
            return jsonify({"error": "type must be 'text' or 'image'"}), 400

        logger.info(f"Search: {query_list}, type: {data_type}, topk: {topk}")

        with model_lock:
            try:
                query_embeddings = jina_model.embed_queries(query_list)
            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
                return jsonify({"error": "Query embedding failed"}), 500

        collection_name = f"{data_type}_collection"
        hits = MilvusDB.search(collection_name, query_embeddings, topk, data_type)
        reranked = rerank_results(hits, query_list, jina_model)[:topk]
        results = format_results(reranked)

        return jsonify({
            "status": "success",
            "type": data_type,
            "total_hits": len(results),
            "results": results
        })

    except Exception as e:
        logger.error(f"Search failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    from jina2 import JinaDenseRetriever

    os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else "logs", exist_ok=True)
    logger.add(LOG_FILE, rotation=LOG_ROTATION)

    try:
        use_gpu = not USE_CPU
        device_info = "CPU" if USE_CPU else "GPU"
        logger.info(f"Loading Jina model ({device_info})...")
        jina_model = JinaDenseRetriever(bs=16, use_gpu=use_gpu)
        logger.info(f"Jina model loaded ({device_info})")
    except Exception as e:
        logger.error(f"Jina model loading failed: {e}")
        logger.error(traceback.format_exc())
        raise

    try:
        db_dir = os.path.dirname(MILVUS_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Database directory ready: {db_dir}")
    except Exception as e:
        logger.error(f"Failed to create database directory: {e}")
        raise

    logger.info(f"Starting retriever service: {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, threaded=True)