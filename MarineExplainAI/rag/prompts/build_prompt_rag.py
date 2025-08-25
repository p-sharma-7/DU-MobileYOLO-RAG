import os
import json
import pandas as pd
from tqdm import tqdm

# === Use quantum retriever ===
from rag.quantum.quantum_retriever import retrieve_context

# === CONFIG ===
PROMPT_IN_CSV = "results/method_2/prompts_m2_abs.csv"
PROMPT_OUT_CSV = "results/method_2/prompts_rag.csv"
TOP_K = 3  # Number of contexts to retrieve

def main():
    print(f"📄 Loading base prompts from: {PROMPT_IN_CSV}")
    df = pd.read_csv(PROMPT_IN_CSV)
    print(f"🔢 Total samples: {len(df)}")

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🧠 Building Quantum-RAG Prompts"):
        img_path = row["original_image"]
        label = row["class_name"]

        try:
            # 🧠 Quantum query format
            query = f"What is known about the marine species '{label}'?"
            contexts = retrieve_context(query, top_k=TOP_K)
            joined_context = "\n---\n".join([ctx["context"] for ctx in contexts])

            # 📝 Final Prompt Template
            prompt = (
                f"You are a marine biology visual expert.\n"
                f"Knowledge Context:\n{joined_context}\n"
                f"<image>\n"
                f"USER: Based on the image and the knowledge above, explain why this image likely contains a {label}.\n"
                f"ASSISTANT:"
            )

            results.append({
                "original_image": img_path,
                "class_name": label,
                "retrieved_context": joined_context,
                "prompt": prompt
            })
        except Exception as e:
            print(f"❌ Error on row {idx}: {e}")
            continue

    # 💾 Save prompts
    os.makedirs(os.path.dirname(PROMPT_OUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(PROMPT_OUT_CSV, index=False)
    print(f"\n✅ Saved RAG prompts to: {PROMPT_OUT_CSV}")

if __name__ == "__main__":
    main()
