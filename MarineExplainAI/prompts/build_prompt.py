import pandas as pd
import os

# === CONFIG ===
CROP_CSV = "results/crop_metadata.csv"
PROMPT_CSV = "results/prompts.csv"

def build_prompt(row):
    species = row["class_name"]
    crop_path = row["crop_path"]
    full_path = row["original_image"]  # ✅ fixed column name

    prompt = (
        f"explain why this object is likely classified as a '{species}'."
        f"Answer in 1-2 sentences."
    )
    return prompt

def main():
    if not os.path.exists(CROP_CSV):
        print(f"❌ Crop metadata not found at {CROP_CSV}")
        return 

    df = pd.read_csv(CROP_CSV)

    if "original_image" not in df.columns or "crop_path" not in df.columns:
        print("❌ Required columns ('original_image', 'crop_path') not found in CSV.")
        print("🧪 Columns available:", list(df.columns))
        return

    df["prompt"] = df.apply(build_prompt, axis=1)
    df.to_csv(PROMPT_CSV, index=False)
    print(f"\n✅ Prompts saved to: {PROMPT_CSV}")
    print(f"📝 Total prompts: {len(df)}")

if __name__ == "__main__":
    main()
