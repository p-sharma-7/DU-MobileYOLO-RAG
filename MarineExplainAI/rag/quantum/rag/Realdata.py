import pandas as pd
import json
import os

CSV_PATH = "rag/5%_filtered_obis_species_processed_+common_species.csv"
OUT_PATH = "rag/data/marine_kb_real.json"

def main():
    print("📥 Reading real marine species dataset...")
    df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

    kb_entries = []
    for _, row in df.iterrows():
        response_raw = str(row["Response"]).strip()
        if not (response_raw.startswith("{") and "traits" in response_raw):
            continue  # Skip non-JSON rows

        try:
            parsed = json.loads(response_raw)
            traits = parsed["traits"]

            species_name = parsed.get("species_name", row.get("class_name_label", "Unknown Species"))
            context = (
                f"Color: {traits.get('color', 'unknown')}. "
                f"Shape: {traits.get('body_shape', 'unknown')}. "
                f"Habitat: {traits.get('habitat', 'unknown')}. "
                f"Size: {traits.get('size', 'unknown')}. "
                f"Diet: {traits.get('diet', 'unknown')}. "
                f"Notable feature: {traits.get('special', 'unknown')}."
            )

            kb_entries.append({
                "title": species_name,
                "context": context
            })
        except Exception as e:
            print(f"⚠️ Skipping row due to error: {e}")
            continue

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(kb_entries, f, indent=2)

    print(f"\n✅ Saved {len(kb_entries)} valid entries to: {OUT_PATH}")

if __name__ == "__main__":
    main()
