import pandas as pd
import os

# === CONFIG ===
CROP_CSV = "results/crop_metadata.csv"
PROMPT_CSV = "results/method_2/prompts_m2_abs.csv"
FULL_IMG_ROOT = "/workspace/Marine/DU-MobileYOLO-main/brackish_dataset/test/images"  # 📌 absolute root

# === Build prompt for method 2 ===
def build_prompt(row):
    class_name = row["class_name"]
    full_img_rel = row["original_image"]
    full_img_abs = os.path.join(FULL_IMG_ROOT, full_img_rel)

    prompt = (
        f"You are a marine biology visual expert.\n"
        f"Explain why this image likely contains a '{class_name}'.\n"
        f"Focus on visual traits (color, texture, shape) and environment (like water, sand).\n"
        f"Image path: {full_img_abs}\n"
        f"Answer in 1-2 sentences."
    )

    return pd.Series({
        "original_image": full_img_abs,
        "class_name": class_name,
        "prompt": prompt
    })

# === Driver ===
def main():
    if not os.path.exists(CROP_CSV):
        print(f"❌ CSV not found: {CROP_CSV}")
        return

    df = pd.read_csv(CROP_CSV)

    if "original_image" not in df.columns or "class_name" not in df.columns:
        print("❌ Missing required columns.")
        return

    df_out = df.apply(build_prompt, axis=1)
    os.makedirs(os.path.dirname(PROMPT_CSV), exist_ok=True)
    df_out.to_csv(PROMPT_CSV, index=False)

    print(f"✅ Saved corrected prompts to: {PROMPT_CSV}")
    print(f"🔢 Total samples: {len(df_out)}")

if __name__ == "__main__":
    main()
