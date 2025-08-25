import pandas as pd, json, os

CSV_PATH = "rag/rag/5%_filtered_obis_species_processed_+common_species.csv"
OUTPUT_PATH = "rag/data/marine_kb_real.json"

df = pd.read_csv(CSV_PATH)
required = ["species_name", "color", "body_shape", "habitat", "size", "diet", "special"]
df = df.dropna(subset=required)

kb = []
for idx, row in df.iterrows():
    kb.append({
        "id": idx,
        "species_name": row["species_name"],
        "traits": {
            "color": row["color"],
            "body_shape": row["body_shape"],
            "habitat": row["habitat"],
            "size": row["size"],
            "diet": row["diet"],
            "special": row["special"]
        }
    })

os.makedirs("rag/data", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(kb, f, indent=2)

print(f"✅ Saved {len(kb)} entries to: {OUTPUT_PATH}")
