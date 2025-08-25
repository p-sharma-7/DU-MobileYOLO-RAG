from rag.quantum.quantum_retriever import retrieve_context

query = "What do we know about sea urchins found in coral reefs?"
results = retrieve_context(query, top_k=3)

for i, ctx in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Title: {ctx['title']}")
    print(f"Content:\n{ctx['context']}")
