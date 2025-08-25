import json

# Real data sample
entries = [
    {
        "species_name": "Tamu fisheri",
        "traits": {
            "color": "translucent, colorless",
            "body_shape": "elongated, gelatinous body with a spherical umbrella and long, sticky tentacles",
            "habitat": "warm, shallow waters, typically found in estuaries or bays",
            "size": "up to 3 centimeters in diameter",
            "diet": "small invertebrates, plankton",
            "special": "bioluminescent, able to emit light"
        }
    },
    {
        "species_name": "Letepsammia",
        "traits": {
            "color": "pale yellow or cream with brownish spots",
            "body_shape": "spherical or ovoid body covered in short, moveable spines",
            "habitat": "deep-sea environments, often found in areas with soft sediments",
            "size": "up to 10 centimeters in diameter",
            "diet": "small invertebrates, detritus",
            "special": "body is highly flexible and can change shape"
        }
    }
]

# Create marine_kb.json format
kb_entries = []
for entry in entries:
    t = entry["traits"]
    context = (
        f"Color: {t['color']}. "
        f"Shape: {t['body_shape']}. "
        f"Habitat: {t['habitat']}. "
        f"Size: {t['size']}. "
        f"Diet: {t['diet']}. "
        f"Notable feature: {t['special']}."
    )
    kb_entries.append({
        "title": entry["species_name"],
        "context": context
    })

# Save to file
with open("rag/data/marine_kb.json", "w") as f:
    json.dump(kb_entries, f, indent=2)

print("✅ marine_kb.json created with real structured data!")
