import requests
import json
from loguru import logger

BASE_URL = "http://localhost:5555"

IMG_PATH = [
    # Add your image description JSONL file paths here
    # Example: "./results/aug_doc_image_descriptions.jsonl",
    "./results/aug_image_descriptions.jsonl"
]

MD_PATH = [
    # Add your markdown document folder paths here
    # Example: "./data/documents",
    "./backups/webpages_backup_20251019_140759"
]

def insert_text(folder_path, batch_size=16):
    """Insert text data via the retriever service."""
    url = f"{BASE_URL}/insert_text"
    payload = {
        "folder_path": folder_path,
        "batch_size": batch_size
    }
    response = requests.post(url, json=payload)
    logger.info(response.json())

def insert_image(jsonl_path, batch_size=16):
    """Insert image data via the retriever service."""
    url = f"{BASE_URL}/insert_image"
    payload = {
        "jsonl_path": jsonl_path,
        "batch_size": batch_size
    }
    response = requests.post(url, json=payload)
    logger.info(response.json())

if __name__ == "__main__":
    logger.info("Inserting text data...")
    for md_folder in MD_PATH:
        logger.info(f"Processing folder: {md_folder}")
        insert_text(md_folder)

    logger.info("Inserting image data...")
    for img_file in IMG_PATH:
        logger.info(f"Processing file: {img_file}")
        insert_image(img_file)

    logger.info("All data insertion complete.")