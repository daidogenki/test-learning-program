#!/usr/bin/env python3
"""
Quick query test script for Pinecone RAG system
Tests basic queries to verify chunking and search functionality
"""

import os
from pinecone import Pinecone
from openai import OpenAI

def main():
    # Initialize clients
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    namespace = os.environ["PINECONE_NAMESPACE"]
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Test queries
    test_queries = [
        "時間外勤務の上限は？",
        "パスワードは何文字以上？", 
        "パスワードは何日ごとに変更？"
    ]
    
    print("=== Pinecone RAG Query Test ===\n")
    
    for query in test_queries:
        print(f"Q: {query}")
        
        # Get embedding for query
        try:
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small", 
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
        except Exception as e:
            print(f"  ERROR: Failed to get embedding: {e}")
            continue
        
        # Search Pinecone
        try:
            search_results = index.query(
                vector=query_embedding,
                top_k=5,
                namespace=namespace,
                include_metadata=True
            )
            
            print(f"  Hits: {len(search_results.matches)}")
            
            for i, match in enumerate(search_results.matches):
                metadata = match.metadata or {}
                text = (metadata.get("text", "") or "").replace("\n", " ")
                section = metadata.get("section", "Unknown")
                
                print(f"    {i+1}. ID: {match.id}")
                print(f"       Score: {match.score:.4f}")
                print(f"       Section: {section}")
                print(f"       Text: {text[:160]}...")
                print()
                
        except Exception as e:
            print(f"  ERROR: Search failed: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()