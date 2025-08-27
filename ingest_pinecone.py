#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from pinecone import Pinecone, ServerlessSpec
import orjson


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_index_if_not_exists(pc: Pinecone, index_name: str, dimension: int = 1536):
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}' with dimension {dimension}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created successfully")
    else:
        print(f"Index '{index_name}' already exists")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upsert_batch(index, vectors: List[Dict[str, Any]], namespace: str):
    """Upsert a batch of vectors to Pinecone."""
    return index.upsert(vectors=vectors, namespace=namespace)


def process_jsonl_file(jsonl_path: str, batch_size: int = 100) -> List[Dict[str, Any]]:
    """Process JSONL file and prepare vectors for Pinecone."""
    vectors = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = orjson.loads(line.strip())
                
                # Skip if no vector
                if not record.get("vector") or len(record["vector"]) == 0:
                    print(f"Skipping line {line_num}: empty vector")
                    continue
                
                # Prepare metadata
                metadata = record.get("metadata", {})
                
                # Add text to metadata if not present
                if "text" not in metadata and "text" in record:
                    metadata["text"] = record["text"]
                
                vector_data = {
                    "id": record["id"],
                    "values": record["vector"],
                    "metadata": metadata
                }
                
                vectors.append(vector_data)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return vectors


def main():
    parser = argparse.ArgumentParser(description="Upsert JSONL embeddings to Pinecone")
    parser.add_argument("--jsonl", required=True, help="Path to embeddings JSONL file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for upsert")
    args = parser.parse_args()
    
    # Environment variables
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    namespace = os.environ.get("PINECONE_NAMESPACE", "default")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable is required")
    
    print(f"Initializing Pinecone client...")
    pc = Pinecone(api_key=api_key)
    
    # Create index if needed
    create_index_if_not_exists(pc, index_name)
    
    # Get index
    index = pc.Index(index_name)
    
    # Process JSONL file
    print(f"Processing JSONL file: {args.jsonl}")
    vectors = process_jsonl_file(args.jsonl, args.batch_size)
    
    if not vectors:
        print("No valid vectors found to upsert")
        return
    
    print(f"Found {len(vectors)} vectors to upsert")
    
    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(vectors), args.batch_size):
        batch = vectors[i:i + args.batch_size]
        try:
            response = upsert_batch(index, batch, namespace)
            upserted_count = response.upserted_count if hasattr(response, 'upserted_count') else len(batch)
            total_upserted += upserted_count
            print(f"Batch {i // args.batch_size + 1}: Upserted {upserted_count} vectors")
        except Exception as e:
            print(f"Error upserting batch {i // args.batch_size + 1}: {e}")
            continue
    
    print(f"Total upserted: {total_upserted} vectors to namespace '{namespace}'")


if __name__ == "__main__":
    main()