import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

def setup_directories():
    outer_names = ['test', 'train']
    inner_names = ['angry', 'happy', 'neutral', 'sad', 'surprised']
    for outer_name in outer_names:
        for inner_name in inner_names:
            os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

def save_images(df):
    emotion_mapping = {0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprised"}
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        pixels = list(map(int, row['pixels'].split()))
        img = Image.fromarray(np.array(pixels).reshape(48, 48).astype('uint8'))
        subdir = 'train' if i < 28709 else 'test'
        img.save(os.path.join('data', subdir, emotion_mapping[row['emotion']], f"im{i}.png"))

def main():
    df = pd.read_csv('./fer2013.csv')
    setup_directories()
    save_images(df)
    print("Done!")

if __name__ == "__main__":
    main()