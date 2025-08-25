from rag.quantum.quantum_retriever import retrieve_context

# === INPUTS ===
img_path = "/workspace/Marine/DU-MobileYOLO-main/brackish_dataset/test/images/sample_image.jpg"
species = "Letepsammia"
label = "crab"  # ← general class for this species

# === QUERY + RETRIEVAL ===
query = f"What is known about the marine species '{species}'?"
contexts = retrieve_context(query, top_k=3)
joined_context = "\n---\n".join([ctx["context"] for ctx in contexts])

# === PROMPT CREATION ===
prompt = (
    f"You are a marine biology visual expert.\n"
    f"Knowledge Context:\n{joined_context}\n"
    f"<image>\n"
    f"USER: Based on the image and the knowledge above, explain why this image likely contains a {label}.\n"
    f"ASSISTANT:"
)

# === PREVIEW ===
print("\n🧪 Quantum-RAG Prompt Preview")
print(f"🖼️ Image Path: {img_path}")
print(f"🏷️ Class Label: {label}")
print("\n🧠 Retrieved Knowledge Context:\n", joined_context)
print("\n📜 Final Prompt:\n", prompt)
