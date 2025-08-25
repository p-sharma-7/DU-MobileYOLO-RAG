import pandas as pd
import os

# Input and output paths
metadata_path = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/crop_metadata.csv"
output_path = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/real/data/abs_image_labels_real.csv"
base_img_dir = "/workspace/Marine/DU-MobileYOLO-main/brackish_dataset/test/images"

# Load metadata
df = pd.read_csv(metadata_path)

# Create full original image paths
df['image_path'] = df['original_image'].apply(lambda x: os.path.join(base_img_dir, x))

# Save only the required columns
df[['image_path', 'class_name']].to_csv(output_path, index=False)

print(f"✅ Saved {len(df)} entries to {output_path}")
