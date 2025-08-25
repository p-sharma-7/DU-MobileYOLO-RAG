import json
import os

KB_JSON = "rag/data/marine_kb.json"
IDMAP_OUT = "rag/index/id_map.json"

def load_kb(path):
    with open(path, "r") as f:
        data = json.load(f)
    id_map = {}
    for idx, entry in enumerate(data):
        id_map[idx] = {
            "title": entry["title"],
            "context": entry["context"]
        }
    return id_map

def main():
    os.makedirs("rag/index", exist_ok=True)
    id_map = load_kb(KB_JSON)
    with open(IDMAP_OUT, "w") as f:
        json.dump(id_map, f, indent=2)
    print(f"✅ Created ID map with {len(id_map)} entries.")

if __name__ == "__main__":
    main()

