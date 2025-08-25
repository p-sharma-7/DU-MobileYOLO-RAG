import pandas as pd
import os

CROP_CSV = "results/crop_metadata.csv"
PROMPT_CSV = "results/method_2/prompts_m2.csv"

def build_prompt(row):
    label = row["class_name"]
    full_path = row["original_image"]

    prompt = (
        f"You are a marine biology visual expert.\n"
        f"Based on this underwater image, explain why this object is likely classified as a '{label}'.\n"
        f"Focus on features such as texture, color, shape, and context like background, water clarity, or marine terrain.\n"
        f"Answer in 1-2 sentences."
    )

    return pd.Series({
        "original_image": full_path,
        "class_name": label,
        "prompt": prompt
    })

def main():
    if not os.path.exists(CROP_CSV):
        print(f"❌ Crop metadata not found at {CROP_CSV}")
        return

    df = pd.read_csv(CROP_CSV)
    if "original_image" not in df.columns or "class_name" not in df.columns:
        print("❌ Required columns ('original_image', 'class_name') missing.")
        return

    df_prompt = df.apply(build_prompt, axis=1)
    df_prompt.to_csv(PROMPT_CSV, index=False)
    print(f"✅ Prompts saved to: {PROMPT_CSV}")

if __name__ == "__main__":
    main()
