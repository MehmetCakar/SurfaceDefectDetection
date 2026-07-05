"""One-off: extract NEU-CLS images from the Hugging Face parquet mirror
(newguyme/neu_cls_caption) into data/raw/<class_name>/*.jpg so that
prepare_data.py can consume them.
"""

import io
from pathlib import Path

import pandas as pd
from PIL import Image

RAW = Path("data/raw")

for split_file in ["train.parquet", "test.parquet"]:
    df = pd.read_parquet(RAW / "parquet" / split_file)
    print(f"{split_file}: {len(df)} rows, classes: {sorted(df['label_str'].unique())}")
    for _, row in df.iterrows():
        cls = row["label_str"].strip().lower().replace(" ", "_")
        dest = RAW / cls
        dest.mkdir(parents=True, exist_ok=True)
        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        Image.open(io.BytesIO(img_bytes)).save(dest / row["filename"])

counts = {p.name: len(list(p.glob("*.jpg"))) for p in RAW.iterdir() if p.is_dir() and p.name != "parquet"}
print("Extracted:", counts)
