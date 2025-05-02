import os
import pandas as pd
# from kaggle.api.kaggle_api_extended import KaggleApi
# from zipfile import ZipFile
import shutil

# # Step 1: Download the dataset from Kaggle
# dataset_slug = 'michefqli/401-data-v-anzi'
# download_path = 'kaggle_download'
# os.makedirs(download_path, exist_ok=True)

# api = KaggleApi()
# api.authenticate()
# api.dataset_download_files(dataset_slug, path=download_path, unzip=True)

# Step 2: Read the label CSV file
image_folder = '/Users/poppy_puppet/Downloads/cat-memes-data-v-anzi'
csv_path = 'cat_memes - Sheet1.csv'
output_base = './dataset'

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df['image'] = df['image'].str.strip()
df['label'] = df['label'].str.strip().str.lower()

# === Create label directories ===
label_map = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}
for label in label_map.values():
    os.makedirs(os.path.join(output_base, label), exist_ok=True)

# === Copy images to respective folders ===
for _, row in df.iterrows():
    src = os.path.join(image_folder, row['image'])
    if os.path.exists(src):
        label = row['label']
        if label in label_map:
            dst_dir = os.path.join(output_base, label_map[label])
            shutil.copy(src, os.path.join(dst_dir, row['image']))
        else:
            print(f"⚠️ Unknown label '{label}' for image {row['image']}")
    else:
        print(f"⚠️ File not found: {src}")

print("✅ Done! Images are now in ./dataset/[positive|negative|neutral].")