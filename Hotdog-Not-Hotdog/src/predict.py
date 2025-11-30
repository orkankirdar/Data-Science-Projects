from tensorflow import keras
import numpy as np
from PIL import Image
import sys

IMG_SIZE = (224, 224)

def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype="float32") 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_image(img_path):
    model = keras.models.load_model("models/hotdog_model.keras")
    img_array = load_image(img_path)

    pred = model.predict(img_array)[0][0]
    print(f"Raw prediction score: {pred:.4f}")

    if pred < 0.5:
        label = "HOTDOG 🌭 (class 0)"
    else:
        label = "NOT HOTDOG ❌ (class 1)"

    print(f"Model prediction: {label}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python3 src/predict.py path/to/image.jpg")
    else:
        img_path = sys.argv[1]
        print(f"Image path: {img_path}")
        predict_image(img_path)
