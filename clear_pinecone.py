#!/usr/bin/env python3
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
namespace = os.environ["PINECONE_NAMESPACE"]

print(f"Clearing namespace: {namespace}")
index.delete(delete_all=True, namespace=namespace)
print("Namespace cleared")