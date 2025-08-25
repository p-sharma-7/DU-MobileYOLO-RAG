import pickle
import json
from rag.quantum.quantum_retriever import retrieve_context

# Load BM25 index and ID map
with open("rag/index/bm25_index_real.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("rag/index/id_map.json", "r") as f:
    id_map = json.load(f)

# Run query
query = "What do we know about sea urchins found in coral reefs?"
results = retrieve_context(query, bm25, id_map, top_k=3)

# Show results
for i, context in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(context)
